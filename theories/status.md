# status

Live project state. Last refreshed: 2026-07-15.

## Frame
Governed by [`theories/ultimate.md`](ultimate.md) +
[`theories/ultimate.infer.md`](ultimate.infer.md):
small systems, small experiments, prove one thing at a time, no emergence.

Anything here that doesn't isolate a single property is wrong by the project's
own rules and should be cut.

## What exists on disk

### Scaffold (this branch, base `909cf23`)
- `src/` — model, dataset, training loop as separate modules.
- `train.py` — entry point for the think-once line.
- `experiments/` — gitignored, one dir per run.
- `devenv.nix` — sole dependency manager.

### Already-run experiments
- **`experiments/exp001/`** (commit `bb44c04`) — proof-of-method for the
  think-once vs re-encode baseline. ~7.6k params, 2000 worlds, 12038 queries,
  15 epochs in 30.8s on CPU. Latent 0.569 vs baseline 0.427 overall;
  breakthrough on WHERE (0.336 vs 0.007) at 4.2x lower per-query compute.
  - **Proves**: recurrent latent state > per-query re-encode at matched
    param count.
  - **Does not yet prove**: *which* part of the latent state does the work
    (no ablation). That is the next prove-one-thing in this frame.

### Uncommitted work in the tree
Stubs for a separate sub-thread (RWKV-nano / logic-niiah):
- `src/rwkv_nano.py`, `src/train_rwkv.py`, `src/logic_niiah_generator.py`
- These are paused (see "Stalled sub-thread" below).

### Archived history
Anything from before commit `909cf23` (down to `028f2a3` and earlier) is the
deeper world-model / thinking-vs-speaking line. Not on this branch, but fully
recoverable: `git show 028f2a3:<path>` or `git checkout 028f2a3 -- <path>`.

## Stalled sub-thread (RWKV-nano + logic niiah)
Paused at the **mode-collapse bug** as previously recorded:
- Single-example overfit works (loss 0.02 in ~200 steps).
- Varied training: loss plateaus ≈ 2.3, exact_acc ≈ 0%, digit_acc ≈ 6%.
- Model collapses to a mode token regardless of context.

Per `theories/ultimate.md`, the right move is **not** to debug via scale-up
("make it bigger, hope it works"). The right moves are each one
prove-one-thing:

- A1. Run a tiny GRU baseline at the same param count. If GRU learns it
  where RWKV nano doesn't, the failure is RWKV-specific, not nano-specific.
- A2. Instrument channel-wise decay + gradient norms to localise *where*
  the collapse lives. One number per channel, one attribution.
- A3. Add RWKV-7-style data-dependent decay (`lerp`/`gated`) as a single
  change, hold everything else constant. If it learns the task, that is a
  proof of mechanism (data-dependent decay is sufficient for the task at
  this scale).

Each is a clean ablation. The bug should be re-opened under the ultimate
frame, not as a free-form architecture exploration.

## Proposed next prove-one-thing experiments

Order them to maximise information per minute of CPU, not per parameter.

1. **`exp002` — WHERE ablation (think-once line).** From `exp001`, the
   latent state already wins WHERE by 30 points. Disable the recurrent
   pass entirely (single forward per query, no state carry). Re-run at
   matched params. If WHERE collapses to ~baseline, that proves the
   *recurrence is the mechanism* for WHERE-style long-horizon recall.
   If WHERE holds, the mechanism is somewhere else in the latent
   head — and that is itself a finding.

2. **`exp003` — generation-loss signal (single-token answer task).**
   See [`theories/generation-loss.md`](generation-loss.md) and the full
   single-variable design in
   [`theories/generation-loss.infer.md`](generation-loss.infer.md).
   Train a small GRU on a one-digit logic-niiah task (1 needle, 1 transform,
   no noise). Three runs at matched params, model, data, and step count —
   only the supervision regime changes:
     (a) **T** — teacher-forcing CE on every answer position. Baseline.
     (b) **U** — own-generation CE on every position. (One variable vs T:
         self-conditioning.)
     (c) **M** — own-generation CE, gradient only on positions where the
         generated token ≠ the target. (One variable vs U: the mask.)
   The pair **U → M** isolates the specific contribution of the
   generation-loss idea; **T → U** and **T → M** are controls. One number
   per run: final exact-match accuracy + loss curve + gradient norm.

3. **`exp004` — RWKV vs GRU on logic niiah (1 needle, no noise).** Strip
   the logic-niiah task down to its minimum (1 variable, 1 transform, no
   noise). Two models, matched params. Single change: recurrence type.
   If neither learns, the task is too sparse — change the data generator,
   not the model. If only one learns, that is the proof-of-mechanism
   result.

## What we are *not* doing right now
- Scaling RWKV-nano in hope of non-emergence — explicitly forbidden by
  `ultimate.md`.
- Multi-change experiments that conflate causes.
- Re-opening the archived world-model line until the scaffold's own single-
  property story is on solid ground.

## Open proposals (filed, not yet run)

- [`theories/generation-loss.md`](generation-loss.md) — train on own-generation
  logits, gradient only on wrong-token positions. Filed 2026-07-15.
  Tied to `exp003`.

## Archive
Old single-theme status focused narrowly on the RWKV-nano bug hunt has been
superseded by this file. It is not deleted in case anything still wants to
read the prior framing.
