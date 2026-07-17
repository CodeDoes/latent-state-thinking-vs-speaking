# status
Live project state. Proof ledger lives in
[`theories/proofs.md`](proofs.md). Per-theory proof status lives in
each theory's `<topic>.status.md`.

## Frame
Governed by [`theories/ultimate.md`](ultimate.md) +
[`theories/ultimate.infer.md`](ultimate.infer.md):
small systems, small experiments, prove one thing at a time, no emergence.

## What exists on disk
### Scaffold
- `src/` — model, dataset, training + analysis as separate modules.
- `experiments/` — tracked in git (configs, metrics, samples); checkpoints
  (`*.pt`) stay ignored. One dir per run.
- `devenv.nix` — sole dependency manager.

## Active thread
- [`theories/byte-state-byte.md`](byte-state-byte.md) — byte → encoder-state →
  patch-model → decoder-state → byte, with two parallel architectures on disk
  (step-function and RWKV-based). Proof status:
  [`theories/byte-state-byte.status.md`](byte-state-byte.status.md).

## B5 follow-up threads (single-variable ablations)
- [`theories/injection-frequency.md`](injection-frequency.md) — front vs per-layer
  core→decoder fusion. Status:
  [`injection-frequency.status.md`](injection-frequency.status.md).
- [`theories/dynamic-patch-vs-fixed.md`](dynamic-patch-vs-fixed.md) — fixed stride
  vs surprise-threshold dynamic patching.
  Status: [`dynamic-patch-vs-fixed.status.md`](dynamic-patch-vs-fixed.status.md).
- [`theories/token-vs-byte-head.md`](token-vs-byte-head.md) — byte head 258 vs
  token head 1k (dense vs sparse supervision).
  Status: [`token-vs-byte-head.status.md`](token-vs-byte-head.status.md).
- [`theories/rwkv-state-carry.md`](rwkv-state-carry.md) — zero init vs stateful
  carry vs learned init for RWKV long memory.
  Status: [`rwkv-state-carry.status.md`](rwkv-state-carry.status.md).
- [`theories/adaptive-exit-entropy.md`](adaptive-exit-entropy.md) — entropy weight
  sweep controlling loop collapse.
  Status: [`adaptive-exit-entropy.status.md`](adaptive-exit-entropy.status.md).

## Diffusion / terminal threads
- [`theories/b3d-rwkv-nano.md`](b3d-rwkv-nano.md) — Triplet-Block Diffusion RWKV
  at nano scale. Status:
  [`b3d-rwkv-nano.status.md`](b3d-rwkv-nano.status.md).
- [`theories/diffusion-grid-terminal.md`](diffusion-grid-terminal.md) — RWKV
  diffusion over H×W byte grid, stochastic output commits when certainty > τ.
  Status: [`diffusion-grid-terminal.status.md`](diffusion-grid-terminal.status.md).

## Scratchboard / viewport threads
- [`theories/movable-grid-scratchboard.md`](movable-grid-scratchboard.md) — text→pos
  association, movable viewport, output-as-scratchboard.
  Status: [`movable-grid-scratchboard.status.md`](movable-grid-scratchboard.status.md).
- [`theories/screen-viewport-zoom-pan.md`](screen-viewport-zoom-pan.md) — screen
  viewport with zoom/pan/quantization for massive content.
  Status: [`screen-viewport-zoom-pan.status.md`](screen-viewport-zoom-pan.status.md).

## Memory / growth threads
- [`theories/dendrite_memory.md`](dendrite_memory.md) — frozen RWKV + LoRA-debra branches with hash-gated lifecycle. **Superseded by `dendrite_growth` (architectural-extensions framing).** Status: [`dendrite_memory.status.md`](dendrite_memory.status.md).
- [`theories/dendrite_growth.md`](dendrite_growth.md) — RWKV trunk that grows new architectural branches (cross-attn/residual/output-head/vocab/state). Status: [`dendrite_growth.status.md`](dendrite_growth.status.md).
- [`theories/delta-mem.md`](delta-mem.md) — stateful adapters inside transformer attention (Delta-rule + RWKV-state). Status: no status file yet.

## Application threads
- [`theories/realtime_ai.md`](realtime_ai.md) — AI that constantly learns from device streams (wifi/bluetooth/PC telemetry). Status: [`realtime_ai.status.md`](realtime_ai.status.md).
- [`theories/ultimate_thesis.md`](ultimate_thesis.md) — consolidated "small systems, fast experiments" thesis tying together all threads. Status: [`ultimate_thesis.status.md`](ultimate_thesis.status.md).

## Working method
- [`theories/working_method.md`](working_method.md) — the operating principles (observable training, git-bound, single-variable, etc.). Status: [`working_method.status.md`](working_method.status.md).
- [`theories/smoke_test_methodology.md`](smoke_test_methodology.md) — 1-minute smoke tests with learnable-pattern synth data. Status: [`smoke_test_methodology.status.md`](smoke_test_methodology.status.md).

## Other threads (filed, not primary)
- [`theories/progressive-expansion.md`](progressive-expansion.md) —
  proof status: [`theories/progressive-expansion.status.md`](progressive-expansion.status.md).

## Open proposals
- [`theories/generation-loss.md`](generation-loss.md) — train on
  own-generation logits, gradient only on wrong-token positions.
- [`theories/read-twice.md`](read-twice.md) — recurrent passes instead
  of new layers for progressive expansion.
- [`theories/byte-patch-preview.md`](byte-patch-preview.md) — BLT/RWKV
  mapping, channel-decay-as-entropy insight. (Now read as
  source-research context for byte-state-byte.)
- [`theories/decoder-ablations.md`](decoder-ablations.md) — one arm
  of a stated 4-variable ablation grid (n_dec_layers).

## What we are *not* doing right now
- Scale-up debugging of RWKV without a named hypothesis — forbidden.
- Multi-change experiments that conflate causes.
- Re-opening the archived world-model line until the scaffold's own
  story is on solid ground.

## Archive
`theories/archive/` holds retired drafts. Superseded ideas get
`git mv`'d there; history preserved via rename detection.
