# tiny-blt-rwkv — proposal (REVISED, awaiting scope confirmation)

## Goal (unchanged)

A byte-level RWKV story LM, scoped down to a CPU-runnable experiment.

## Why TinyStories

Open-domain children's-story text. ~2GB total. A *tiny* slice (~5KB
of text, roughly a few hundred short stories) is enough to expose
local byte-level statistics — even fixed-window models will pick up
character/word patterns at this scale.

## V0: byte-only, no patcher (smallest viable)

What changes from the existing train_byte_rwkv.py:
1. Replace the LogicNiiahGenerator loader with a raw byte-stream
   dataloader that reads from `experiments/byte_ts_001/text.txt`.
2. Use sliding window next-byte prediction (no answer-span masking)
   across the whole corpus.
3. Periodically call `model.generate_one(prompt, max_new_tokens=120)`
   and log the sample.
4. Same `RWKVNano(vocab_size=258, dim=64, layers=2)` — ~140K params.

What this proves: byte-RWKV can train end-to-end on real text data
(at all — the logic-niiah generator is synthetic). If the generated
samples after training look like *anything* story-like, the
per-channel WKV time-mixing is doing real work.

## V1: lightweight patcher (one change from V0)

Prefix a 1-layer byte-MLP entropy head → threshold binarised
predicted entropy → use as `patch_lengths` for an RWKV computed over
patch tokens.

What this proves: whether entropy-driven patching produces a
different loss curve or sample distribution than V0 at the same
data and step budget. Single change (entropy head) vs V0 baseline.

## Why we are stopping at V0/V1 (not full BLT)

The original BLT has three stages (encoder → pool → global → cross-
attention → decoder). At <150K params this is silly compute overhead
relative to the question being asked: *does the byte-stream part of
the design improve on plain byte-RWKV?* If V1 doesn't show a
clear divergence from V0, full BLT isn't worth the engineering
cost for this project's scale.

## Open scope questions to confirm before coding

| Question | Default if left default | Alternative |
|----------|--------------------------|--------------|
| V0 only, or V0+V1? | V0 first, V1 only if V0 succeeds | skip V1 entirely |
| How many KB of stories? | ~5KB (~500 short stories) | 50KB or 500KB |
| Compare to char-version as a baseline run? | yes, byte vs char at matched step count | only if V0 succeeds |
| Are we aiming for coherent stories, or just loss curves? | loss + sample inspection | loss-only |

## Risks

- Byte-RWKV at <150K on 5KB will not produce coherent stories. Loss
  curve + sample inspection is the success criterion, not ROUGE/BLEU.
- HF bandwidth: TinyStories-train.txt is 2GB; we only pull a prefix.
  HF rate-limits aggressive downloads. Mitigation: cap at
  `range:bytes=0-5000` in the HEAD request first to validate, then
  pull the slice and cache locally.
- The byte-only path on plain ASCII gives the model a structural
  advantage (English byte frequencies are very imbalanced) — it
  essentially learns to copy common bigrams. This is fine for V0 but
  V1 has to demonstrate *something* beyond that baseline.

## What this is NOT

- A reproduction of BLT.
- A byte-LM benchmark.
- A claim that we built BLT in 140K params. We didn't.

It's a *testbed* for whether the entropy-patcher machinery has
detectable effects on training behaviour at all. That's the project's
question.
