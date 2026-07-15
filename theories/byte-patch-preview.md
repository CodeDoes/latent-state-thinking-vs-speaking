# byte-patch-preview

Link: https://github.com/facebookresearch/blt

## What BLT does

BLT replaces fixed subword tokenization with **dynamic byte patching**.
Instead of splitting text into vocabulary atoms in advance, the model
learns to merge adjacent bytes into "patches" on the fly, using an
entropy signal: high-uncertainty byte spans get merged (one shared
representation), low-uncertainty spans stay granular.

The key operation is a *patch predictor*: a small network that scans
the entropy distribution across a sequence and outputs a binary merge
decision at each position. Patches become the new sequence tokens.

Result: no tokenizer bottleneck, no OOV, and adaptive sequence length
— rare/cryptic spans get one token, common spans stay byte-fine.

## Why this matters for RWKV (and for progressive expansion)

RWKV's recurrence is the right substrate for adaptive patching because:

- **Time-mixing already operates over the sequence dimension.** The WKV
  accumulation naturally handles inputs of varying semantic grain. A
  patch boundary is just another position where information must be
  routed forward. The existing gating mechanism already does this.
- **No new topology required.** Patch merging is a pre-processing step
  over the input stream; the RWKV block itself doesn't change. You replace
  the tokenizer with a patch encoder → the rest of the model is unchanged.
- **Gradual rollout possible.** You can start with a coarse fixed patch
  size (e.g. 4-byte patches), then progressively introduce a learned
  predictor. That's a single-change ablation (patch size) with a clear
  comparison point.

## Hypothesis

> A byte-patched RWKV matches or exceeds a subword-tokenized RWKV on
> tasks with rare vocabulary items (entity names, numbers, code tokens)
> at matched parameter count and training steps, and does so without
> any tokenizer vocabulary design choices.

Corollary for progressive expansion:

> The entropy signal that drives patching *is itself* a bottleneck
> indicator. Regions where the patch predictor aggressively merges bytes
> are exactly the regions where the model has strong, reliable priors.
> Regions where it keeps the patch fine are regions of high uncertainty —
> the same regions capacity-pressure metrics should flag in activation
> space.

If the two signals agree, you have cross-validated bottleneck detection:
patches tell you *what the model doesn't trust*, activations tell you
*where the model is straining*. They should overlap.

## Minimal test

1. Take a trained RWKV (subword-tokenized). Run it on byte-encoded
   inputs of the same task. Compare per-sequence accuracy.
2. Add a *fixed-size* byte patcher (no learned entropy predictor yet —
   just split every N bytes). Re-train at matched params.
3. If fixed-size patches are worse: the patching granularity matters and
   you need the learned predictor.
4. If fixed-size patches are equal or better: RWKV's recurrence is
   already extracting representational value from byte locality, and the
   tokenizer was masking that.

Then and only then: introduce the learned entropy predictor (BLT's
actual mechanism) as a *single change* from the fixed-patch baseline.

## What it doesn't yet prove

- That byte patching + RWKV beats byte patching + a non-recurrent model.
- That the patching mechanism generalises beyond the task it was trained
  on.
- That progressive expansion (new layers at detected bottlenecks) works
  *inside* a byte-patched model rather than a subword one.

Those are下游 experiments, not the first proof of mechanism.
