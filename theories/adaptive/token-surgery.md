# Token Surgery (Layer-Aware)

Replace the token → state encoder and state → token decoder layers of a
pre-trained RWKV with byte-level counterparts. The frozen "thinking" core
stays intact.

## Motivation

RWKV doesn't "think in tokens" — it thinks in state vectors. The first few
layers encode tokens into state vectors, the middle layers think in state
space, and the last few layers decode state vectors back into token
predictions.

Surgery replaces the encoder (first K layers + embedding) and decoder
(last K layers + head) while keeping the core intact. The byte encoder
learns to produce state vectors the core can understand; the byte decoder
learns to turn core state vectors into byte predictions.

## Experiment Design

### Single variable

**Layer initialization**: pre-trained world-tokenizer layers copied as
initialization for byte encoder/decoder vs. random initialization.

### Arms

| Arm | Pre-training | Surgery | Frozen? |
|-----|-------------|---------|---------|
| A (surgery) | 500 steps, world tokenizer (65529 vocab) | Replace enc+dec layers with byte versions (258 vocab) | Core layers frozen |
| B (scratch) | None | Same ByteLevelRWKV architecture | Nothing frozen |

### Architecture split (3-layer model)

```
Original: embed → [RWKV-0, RWKV-1, RWKV-2] → head
Surgery:  byte_embed → byte_encoder → [frozen RWKV-1] → byte_decoder → byte_head
                    (RWKVBlock init from RWKV-0)        (RWKVBlock init from RWKV-2)
```

### Data

`sum_threshold` rule. Phase 1 uses the RWKV world BPE tokenizer
(65529 vocab). Phases 2-4 use raw bytes (258 vocab: PAD + UNK + 256 bytes).

### Size

- Phase 1 model: ~4.4M params (embed=65529×64 + 3×RWKVBlock + head=65529×65)
- Surgery: ~66K trainable (byte_embed=258×64 + 2×RWKVBlock + byte_head=258×65)
- ~4.3M frozen (core layer + ln_out)

## Claim T1

Pre-trained RWKV core (trained on world BPE tokens) transfers its learned
computation to a surgically attached byte-level interface. Core layers
learn *computational strategies* — not token-specific patterns.

## Results

`exp/token_surgery/001` (f0e6434): token→byte interface surgery preserves
>20× head start over from-scratch byte training.

## See also

- `src/token_surgery.py` — experiment script
- `src/hf_rwkv_tokenizer.py` + `src/rwkv_vocab_v20230424.txt` — RWKV world tokenizer
- Tag `exp/token_surgery/001-pre` / `exp/token_surgery/001-post`
- [`research/rwkv_overview.md`](../research/rwkv_overview.md) — DREAMSTATE, RWKV state editing
