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
- [`theories/progressive-expansion.md`](progressive-expansion.md) —
  proof status: [`theories/progressive-expansion.status.md`](progressive-expansion.status.md).

## Open proposals (filed)
- [`theories/generation-loss.md`](generation-loss.md) — train on
  own-generation logits, gradient only on wrong-token positions.
- [`theories/read-twice.md`](read-twice.md) — recurrent passes instead
  of new layers for progressive expansion.
- [`theories/byte-patch-preview.md`](byte-patch-preview.md) — BLT/RWKV
  mapping, channel-decay-as-entropy insight.
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
