# tiny-blt-rwkv — proposal

## Goal

A byte-level RWKV story LM, scraped down to a CPU-runnable experiment.

## Why TinyStories

Open-domain children's-story text. ~2GB total; a *tiny* subset (1–5 KB of
text, i.e. a handful of stories) is enough to see whether the model
captures *any* story statistics — even a few KB will expose byte n-gram
behavior. The interesting check isn't coherent generation; it's whether
a byte-RWKV picks up *local* patterns the way a char-level RWKV does,
which is the BLT-vanilla case (patches shorten but do not learn).

## V0: byte-only, no patcher

The cheapest first experiment is just `train_byte_rwkv.py` retargeted
to a text stream. No entropy predictor, no three-stage. Just:

1. Pull a small slice of TinyStories raw text via HF.
2. Tokenise to bytes (already in `src/byte_vocab.py`).
3. Train `RWKVNano(vocab_size=258, dim=64, layers=2)` on a sliding window
   of bytes with next-byte prediction loss.
4. Periodically `generate_one(prompt)` and look at the sample.
5. Record loss curve → `experiments/byte_ts_001/metrics.json`.

Single variable across this run vs `prog_exp_001`: task distribution,
not architecture. Same model, same vocab size class, same step budget.
If byte-RWKV's loss on TinyStories converges to ~the same level as
char-RWKV's loss on logic-niiah, that proves the architecture
*generalises* across domains — meaningful because char-RWKV was trained
only on rails-like generator output.

## V1: lightweight patcher

A single change from V0: prefix a tiny entropy head (1-layer byte MLP
predicting next-byte logits) → threshold binarise its predicted
entropy → use the resulting patch boundaries as `patch_lengths` for an
RWKV computed over patch tokens.

What V1 measures: does entropy-driven patching change overfit/local-fit
behaviour vs V0 on the same data? Patch-heterogeneity should show up
in the loss curve or, more strikingly, in the *distribution* of
generated lengths (V0 chokes on a uniform window; V1 follows the
entropy).

## What we are NOT doing

- Not building BLT's three-stage encoder/decoder with cross-attention.
  That is a much larger architecture change. The user's "tiny BLT" is
  V0 (or V1 at most).
- Not downloading the full 2GB dataset.
- Not scaling up to ~1M params. We're at <150K. The "tiny" choice is
  deliberate; the gap between V0 and V1 is what matters, not absolute
  quality.

## Risks (calibrated)

- Byte-RWKV at <150K params on a few KB of stories is expected to
  learn *nothing story-like*. That's fine — the question is whether
  it's *structurally* different from a purely-overfit memoisable
  baseline. If yes, V1 is interesting.
- TinyStories is plain ASCII for the most part; there is no
  multi-byte UTF-8 stress. The 256-byte mapping is safe.
- HF access can fail mid-experiment. Mitigation: cache the slice to
  `experiments/byte_ts_001/text.txt`.
