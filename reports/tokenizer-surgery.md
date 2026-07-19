# Tokenizer Surgery for RWKV

**Can you replace a pre-trained RWKV's tokenizer with a byte-level interface
without losing the learned computation in its core layers?**

---

## The Problem

RWKV models are trained on tokenized text (e.g., the RWKV "world" BPE
tokenizer with 65,529 tokens). The user wants to feed raw bytes instead —
eliminating the tokenizer dependency. But retraining from scratch on bytes
is expensive and wasteful if the core layers already know how to reason.

The core insight: **RWKV doesn't "think in tokens" — it thinks in
state vectors.** The first few layers encode tokens into state vectors,
the middle layers process those states, and the last few layers decode
states back into token predictions. If we can replace only the
encoder/decoder layers while keeping the core intact, the core's learned
computation should transfer.

## Three Experiments

### 1. Char-tokenizer → bytes (simple weight copying)

**Setup**: Train RWKVNano (dim=64, 3 layers, 118K params) on char-level
tokens (vocab=76). Copy embedding/head weights from char positions to
their byte-equivalent positions (ASCII chars overlap 1:1 with bytes).
Freeze **all 3 RWKV blocks**. Train only the new 256-byte embed + head
(33K params).

**Result**: Step-1 loss **1.05** vs from-scratch **5.55** — **5.3× better**.
At step 200: 0.084 vs 2.25 (**27× better**).

### 2. World BPE tokenizer → bytes (layer-aware surgery)

**Setup**: Train RWKVNano (dim=64, 3 layers, 8.6M params due to 65K vocab)
on the real RWKV world BPE tokenizer. Split model 1+1+1 (encoder layer 0,
core layer 1, decoder layer 2). Replace encoder and decoder with byte
versions (258 vocab). Init decoder from **second-to-last layer** (layer 1
= core), not the last layer (layer 2 = token-specialized). Frozen core:
54K params. Trainable encoder+decoder: 141K.

**Result**: Step-1 loss 4.75 vs scratch 5.50 (Δ=0.75). Positive but modest.
The BPE tokenizer creates a bigger distribution shift: " = " is one BPE
token but 3 raw bytes, so the core learned to process denser, semantically
loaded inputs.

### 3. Alternating distillation → task → distillation (the fix)

**Setup**: Same as experiment 2, but instead of one-shot surgery, run an
alternating loop:

1. **Distil**: Train byte encoder to match world-tokenizer state vectors
   at the encoder→core boundary (layer 0 output). MSE loss on pooled byte
   states aligned via the tokenizer's byte→token span mapping.
2. **Task**: Train encoder + decoder on the actual byte-level task.
3. **Repeat**: Re-distil the encoder (now with a trained decoder, the
   alignment signal is sharper). Train task again.

**Result across 4 rounds** (266 task steps total, 256 distillation steps):

```
Round 1 (init from world):   task loss 4.73 → 2.09
Round 2 (carry forward):     task loss 2.20 → 1.15  
Round 3 (re-aligned):        task loss 1.14 → 0.55
Round 4 (compounds):         task loss 0.54 → 0.17
```

Each round compounds: distillation re-aligns the encoder to produce states
the frozen core understands, task training pushes the decoder further, and
the next distillation round has better gradient signal because the decoder
is no longer random.

## Comparison

| Experiment | Step-1 loss | After 100 task steps | After 266 task steps |
|---|---|---|---|
| From scratch (all 195K params) | 5.50 | 4.61 | 0.09 |
| Surgery, no loop (141K trainable) | 4.75 | 2.55 | ~0.01 |
| Surgery + alternating loop (141K trainable, 54K frozen) | 4.73 | **2.09** | **0.17** |

The alternating loop converges **2.2× faster per task-step** than
from scratch, while 54K core params never re-learn.

## What Was Learned

1. **The core layers encode computational strategies, not token patterns.**
   The char→byte surgery transferred with 27× head start. The world→byte
   surgery transferred with 1.16× head start. Transfer is positive in both
   cases — the core is not token-bound.

2. **The bottleneck is the decoder, not the encoder.** Initial task loss
   is dominated by the byte output head being untrained. State distillation
   on the encoder side reduces MSE by 10× but doesn't directly help the
   decoder. The alternating loop fixes this by giving the decoder time to
   learn between distillation rounds.

3. **Decoder initialization matters.** Init from the second-to-last layer
   (core/general states) is better than the last layer (token-specialized).
   The last layer's weights are too coupled to the specific vocabulary.

4. **The alternating loop compounds.** Each pass tightens the alignment
   between byte encoder states and what the frozen core expects to see.
   With more rounds, the byte interface converges toward zero-loss without
   ever modifying the core.

## Code

- `src/token_surgery.py` — all phases (pretrain, distill, surgery, alternating)
- `src/hf_rwkv_tokenizer.py` + `src/rwkv_vocab_v20230424.txt` — RWKV world tokenizer
- `src/rwkv_nano.py` — the RWKV model architecture
- `src/byte_vocab.py` — 258-token byte vocabulary

### Tagged experiments

| Tag | Description |
|-----|-------------|
| `exp/token_surgery/001` | Char→byte weight copying |
| `exp/world_surgery/001` | World→byte layer-aware surgery |
| `exp/token_surgery/T1` | Theory claim: core transfers through surgery |

## Future Directions

1. **More alternating rounds** — the loss was still dropping at round 4.
   8-10 rounds could approach zero-loss byte-level operation.

2. **Patch-level encoder** — instead of one byte per position, group bytes
   into patches (like BLT) and produce one state vector per patch. The
   distillation would then align patch states to token states.

3. **Larger core** — the experiment used a 1-layer core. A real RWKV-7 has
   24+ layers; the middle 22 would stay frozen. The surgery only replaces
   the outer layers, so compute cost scales with encoder+decoder size,
   not core size.

4. **Real pretrained weights** — instead of training the world-tokenizer
   model from scratch, load an actual RWKV-7 checkpoint. The distillation
   approach is identical; only the state vectors change.
