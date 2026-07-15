# status
Live project state. Last refreshed: 2026-07-15 (updated after prog_exp_001).

## Frame
Governed by [`theories/ultimate.md`](ultimate.md) + [`theories/ultimate.infer.md`](ultimate.infer.md):
small systems, small experiments, prove one thing at a time, no emergence.

## What exists on disk
### Scaffold (this branch, base `909cf23`)
- `src/` — model, dataset, training + analysis as separate modules.
- `experiments/` — gitignored, one dir per run.
- `devenv.nix` — sole dependency manager.

### Already-run experiments
- **`experiments/exp001/`** (commit `bb44c04`) — think-once vs re-encode baseline,
  ~7.6k params. Latent 0.569 > baseline 0.427 overall; WHERE breakthrough
  0.336 vs 0.007. Proves recurrent latent state > per-query re-encode at
  matched param count.

- **`experiments/prog_exp_001/`** (commit `5161ad7`) — RWKV 2-layer (dim=64,
  117K params), 2000 steps on easy logic-niiah (2 vars, 2–4 transforms).
  Final accuracy 71% exact-match, best 67%. Bottleneck map captured across
  5 layer groups (330 channel-layer pairs).
  - **A1 validated (first-pass)**: capacity pressure *is* observable.
    The only signal with dynamic range here is **effective_dimensionality**
    (per-channel SVD loading shift). Saturation and range metrics hit ceiling
    on this hard/easy gap.
  - Layer hierarchy by anomaly: `head` > `ln_out` ≈ `embed` ≈ `blocks.*`.

## Active thread: progressive-expansion (integrated into towerverse)
See [`theories/progressive-expansion.md`](progressive-expansion.md),
[`theories/progressive-expansion.infer.md`](progressive-expansion.infer.md),
[`theories/progressive-expansion.assumptions.md`](progressive-expansion.assumptions.md),
[`theories/progressive-expansion.metrics.md`](progressive-expansion.metrics.md).

Current proof chain:
- A1 (signal existence) — partially proven. Next step: calibrate easy/hard
  gap to produce partial degradation, not total saturation.
- A2 (location relevance) — needs per-model ablation (insert layer at detected
  bottleneck vs random vs no insertion).
- A4 (non-obstruction) — pending first expansion experiment.
- A3 (faster convergence) — pending A2.

Key design decision recorded in
[`theories/progressive-expansion.assumptions.md`](progressive-expansion.assumptions.md):
channel *numbers* are not stable across training seeds (dataset programs
capacity allocation, but the exact channel assignment is a symmetry of the
loss landscape). A2 must be tested within a single model instance, not
cross-run.

## Open proposals (filed, not yet run)
- [`theories/generation-loss.md`](generation-loss.md) — train on own-generation
  logits, gradient only on wrong-token positions. Tied to `exp003`.
- [`theories/read-twice.md`](read-twice.md) — recurrent passes instead of
  new layers for progressive expansion.

## What we are *not* doing right now
- Scale-up debugging of RWKV without a named hypothesis — forbidden.
- Multi-change experiments that conflate causes.
- Re-opening the archived world-model line until the scaffold's own story
  is on solid ground.

## Archive
Old single-theme status focused narrowly on the RWKV-nano bug hunt has been
superseded by this file.
