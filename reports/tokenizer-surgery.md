# Tokenizer Surgery for RWKV

**Replace a pre-trained RWKV's tokenizer with a byte-level interface
without losing the learned computation in its core layers.**

---

## The Problem

RWKV models are trained on tokenized text (the RWKV "world" BPE tokenizer,
65,529 tokens). The user wants to feed raw bytes instead — eliminating the
tokenizer dependency. But retraining from scratch on bytes is expensive.

The core insight: **RWKV doesn't "think in tokens" — it thinks in state
vectors.** The first few layers encode tokens into state, the middle layers
think in state space, the last few layers decode state back to tokens.

If we can replace only the interface layers (encoder/decoder) while keeping
the core intact, the model's learned computation transfers.

---

## Five Approaches (in increasing order of elegance)

### 1. Weight copying — char tokenizer → bytes

Train on char-level tokens (vocab=76). Copy embedding/head weights to byte
positions (ASCII overlaps 1:1). Freeze all layers. Train only new embed+head
(33K params).

**Result**: Step-1 loss **1.05** vs scratch **5.55** — 27× better at step 200.
Works because the char vocabulary is a subset of the byte vocabulary.

**Downside**: Only works for overlapping vocabularies. Doesn't generalize to
BPE tokenizers.

### 2. Layer-aware surgery — world BPE → bytes

Train on the real world BPE tokenizer (65K vocab). Split model 1+1+1
(encoder=layer 0, core=layer 1, decoder=layer 2). Replace encoder and
decoder with byte versions. Init decoder from **second-to-last layer**
(layer 1 = core), not the last layer (layer 2 = token-specialized).

**Result**: Step-1 loss 4.75 vs scratch 5.50 (Δ=0.75). Positive but modest.

**Downside**: Replaces the output head — the model has to learn a new
258-vocab output projection from scratch. The decoder bottleneck dominates
initial loss.

### 3. Alternating distillation + task — compounds the transfer

Same as #2, but run an alternating loop: distil encoder (MSE against world
states) → train task → repeat. Each round tightens alignment because the
decoder improves between distillation passes.

**Result** (4 rounds, sum_threshold rule):
```
Round 1: task loss 4.73 → 2.09
Round 2: task loss 2.20 → 1.15
Round 3: task loss 1.14 → 0.55
Round 4: task loss 0.54 → 0.17
```
Converges 2.2× faster per task-step than from scratch, 54K core params frozen.

**Multilingual version** (5 languages, TinyStories):
```
Round 1: eval 3.25  →  Round 3: eval 2.08 (best)
```
Works on real data too, but gains are smaller (harder task, diverse token
distributions).

**Downside**: Still replaces the decoder. The head has to learn byte
predictions from scratch.

### 4. Shunt — bypass layer 0, train encoder for layer 1

Instead of replacing layers, keep the **entire model 100% frozen**. Train a
shunt encoder that feeds directly into layer 1, bypassing layer 0 entirely.
The shunt is initialized from layer 0's weights and learns to produce state
vectors layer 1 can consume. The original head (65K vocab) stays — it
already maps the first 256 tokens to single bytes, so byte prediction works
from the start.

```
bytes → shunt_encoder → [frozen layer 1..L-1 + ln_out + head] → logits
```

**Result**: Task loss **0.02 from round 1** (vs 4.75 for surgery). No decoder
learning needed — the original head already handles byte tokens.

**Why it works**: The world tokenizer's first 256 tokens are single bytes
0x00-0xFF. When byte input maps to these token IDs (off by a small offset),
the original head can predict them correctly without retraining. The shunt
only needs to learn the encoder transformation (what layer 0 normally does).

**Downside**: The output head is still 65K-vocab, not 258-byte. The model
emits world token IDs that happen to be bytes — it's not a true byte-level
model yet.

### 5. Loopy tokenizer — learned byte→token front-end

A loopy RNN that reads raw bytes one at a time, accumulates state, and emits
token IDs into the completely untouched frozen model. The RNN learns to
emulate the TRIE's greedy byte→token mapping as a differentiable recurrent
process.

```
bytes → [loopy RNN: reads, accumulates, triggers] → token_id → [frozen RWKV-7]
         ┌────────────────────────────────────────┐
         │  AccumulatorCell:                       │
         │    byte_embed → RWKV7Block → token_head │
         │                           → trigger     │
         └────────────────────────────────────────┘
```

The cell runs once per byte, updates a matrix state (H × N × N), and
decides: emit a token now (trigger) or keep reading. When triggered, the
accumulated token ID feeds into the frozen model.

**Result**: 106K trainable params, everything else frozen. The frozen model
never knows bytes exist — it sees token IDs as if from the normal tokenizer.

**Why it matters**: 
- The tokenizer is modeled as a learned program (loopy RNN), not a separate
  hard-coded component
- In future, the whole stack (loopy front-end + core) can be retrained
  end-to-end without changing the architecture
- The loopy RNN's trigger mechanism naturally learns token boundaries from
  the real tokenizer's byte spans (via supervised alignment)

---

## Comparison

| Approach | Models changed | Head replaced | Initial loss | Trainable params |
|----------|---------------|---------------|-------------|-----------------|
| 1. Weight copy | All frozen | No (76→258) | **1.05** | 33K |
| 2. Layer surgery | Encoder+decoder | Yes (65K→258) | 4.75 | 141K |
| 3. Alternating | Encoder+decoder | Yes | 4.73 | 141K |
| 4. Shunt | **None frozen** | **No** | **0.02** | 100K |
| 5. Loopy | **None frozen** | **No** | TBD | **106K** |

---

## What Was Learned

1. **The core layers encode computational strategies, not token patterns.**
   Transfer is positive across every approach. The core is not token-bound.

2. **The decoder is the bottleneck.** Approaches that keep the original head
   (#4 shunt, #5 loopy) start far ahead of those that replace it (#2, #3).

3. **The alternating loop compounds.** Each pass tightens encoder alignment.
   Effective on both synthetic rules and real multilingual data.

4. **The tokenizer is just layers 0 and L-1.** In vector space, layer 0 is
   the tokenizer encoder (token IDs → state), and the last layer + head is
   the tokenizer decoder (state → token IDs). Replace or bypass these, and
   the tokenizer disappears as a separate component.

5. **RWKV-7 architecture is essential.** The matrix state and data-dependent
   decay make the core general enough to transfer. Old RWKV-4 vector state
   was too rigid.

---

## Code

| File | What |
|------|------|
| `src/rwkv_nano.py` | RWKV-7 architecture + legacy RWKV-4 compat |
| `src/token_surgery.py` | Approaches #2 and #3 (surgery + alternating) |
| `src/shunt.py` | Approach #4 (bypass layer 0, train encoder for layer 1) |
| `src/loopy_tokenizer.py` | Approach #5 (learned byte→token front-end) |
| `src/train_tokenizer_multilingual.py` | Multilingual TinyStories training |
| `src/multilingual_tinystories.py` | 29-language TinyStories data loader |
| `src/hf_rwkv_tokenizer.py` | RWKV world tokenizer (TRIE-based) |
| `tests/test_rwkv_tokenizer.py` | 61 unit tests for the tokenizer |

## Data

Multilingual TinyStories from
[Dxniz/TinyStories-Multilingual](https://huggingface.co/datasets/Dxniz/TinyStories-Multilingual):
29 languages, 10K-100K stories, JSONL format with `language_code`,
`output` (story), `score` (quality 0-10).

## Tagged Experiments

| Tag | Description |
|-----|-------------|
| `exp/token_surgery/001` | Weight copy (char→byte) |
| `exp/world_surgery/001` | Layer surgery (world→byte) |
| `exp/token_surgery/T1` | Theory claim: core transfers through surgery |

## Future Directions

1. **Train the loopy tokenizer's trigger** — supervised alignment to real
   tokenizer byte spans. The AccumulatorCell already has a trigger head;
   it needs to learn when to fire (matching the TRIE's greedy longest-match
   boundaries).

2. **End-to-end retraining** — once the loopy front-end stabilizes, unfreeze
   the core and train the whole stack jointly. The tokenizer disappears into
   the model.

3. **Full 29-language training** — only 5 tested so far. More languages =
   more diverse token patterns = stronger alignment.

4. **Larger core** — the experiment used 1 frozen layer. A real RWKV-7 has
   24+ layers; the middle 22 stay frozen. Shunt/loopy cost scales with
   encoder size only.
