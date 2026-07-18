# status

Live project state. Run `python src/_status.py` for a one-screen dump of theories, experiments, code, and git state.

The proof ledger lives in [`theories/proofs.md`](proofs.md). Per-experiment status lives in each `experiments/<id>/` directory (config.json + train.log + metrics.json).

## Frame

Governed by [`theories/ultimate.md`](ultimate.md) +
[`theories/ultimate_thesis.md`](ultimate_thesis.md):
small systems, small experiments, prove one thing at a time, no emergence.

Read order for a newcomer: `ultimate.md` → `ultimate_thesis.md` → any individual theory → the experiment dir.

## Conventions

- **Theories**: one file per topic in `theories/`. Each is the prose, hypothesis, and status for one thread.
- **Experiments**: one dir per run in `experiments/`. Each dir contains `config.json` (with the git hash at run time), `train.log`, and `metrics.json` or `samples.json`. Check a run with `python src/_status.py --experiment`.
- **Code**: `src/` holds model, dataset, training, analysis modules. `train_*.py` at repo root are the runnable entry points.
- **Reports**: `reports/` is for summaries written for people who do not need the full theory.

## Active thread

[`theories/byte-state-byte.md`](byte-state-byte.md) — the main architecture family: byte → encoder-state → patch-model → decoder-state → byte.

## Memory / growth thread

- [`theories/dendrite_memory.md`](dendrite_memory.md) — frozen trunk + LoRA-pack branches with hash-gated lifecycle (superseded by dendrite_growth).
- [`theories/dendrite_growth.md`](dendrite_growth.md) — frozen trunk + new architectural modules per memory (current framing).
- [`theories/delta-mem.md`](delta-mem.md) — stateful adapters inside Transformer attention.
- [`theories/progressive-expansion.md`](progressive-expansion.md) — surgical addition of new layers at detected bottlenecks.
- [`theories/read-twice.md`](read-twice.md) — recurrent re-passes instead of new layers.

## Adaptive compute thread

- [`theories/adaptive-exit-entropy.md`](adaptive-exit-entropy.md) — entropy regularization on adaptive-loop gates.
- [`theories/injection-frequency.md`](injection-frequency.md), [`theories/dynamic-patch-vs-fixed.md`](dynamic-patch-vs-fixed.md), [`theories/token-vs-byte-head.md`](token-vs-byte-head.md) — derivative ablations on encoder / core / decoder interface.

## RWKV core thread

- [`theories/rwkv.md`](rwkv.md) — small-model stress test.
- [`theories/rwkv-state-carry.md`](rwkv-state-carry.md) — long-memory mechanism for RWKV.

## Diffusion / terminal thread

- [`theories/b3d-rwkv-nano.md`](b3d-rwkv-nano.md) — Triplet-Block Diffusion RWKV at nano scale.
- [`theories/diffusion-grid-terminal.md`](diffusion-grid-terminal.md) — RWKV diffusion over H×W byte grids.

## Scratchboard / viewport thread

- [`theories/movable-grid-scratchboard.md`](movable-grid-scratchboard.md) — agent reads, moves a 2D grid, uses output as scratchboard.
- [`theories/screen-viewport-zoom-pan.md`](screen-viewport-zoom-pan.md) — screen viewport with zoom/pan/quantization.

## Application thread

- [`theories/realtime_ai.md`](realtime_ai.md) — AI that constantly learns from device streams (wifi/bluetooth/PC telemetry).

## Operating methods

- [`theories/working_method.md`](working_method.md) — rules for how the project is run.
- [`theories/smoke_test_methodology.md`](smoke_test_methodology.md) — 1-minute proof rule.
- [`theories/generation-loss.md`](generation-loss.md) — proposed self-generated-logit loss.

## Other threads

- [`theories/decoder-ablations.md`](decoder-ablations.md) — one arm of a planned 4-variable ablation grid.
- [`theories/byte-patch-preview.md`](byte-patch-preview.md) — design notes on BLT/RWKV mapping. Inspected source code of `facebookresearch/blt`.

## What we are *not* doing right now

- Scale-up debugging of RWKV without a named hypothesis.
- Multi-cause experiments.
- Re-opening the archived world-model line until active threads are stable.

## Archive

`theories/archive/` holds retired drafts. Superseded ideas get `git mv`'d there; history preserved.
