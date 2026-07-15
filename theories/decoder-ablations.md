# decoder-ablations

## Ablation: do the byte-level decoder RWKV blocks carry signal?

Setup: identical encoder + patch slot + data, only `n_dec_layers` varies.
- 9 KB TinyStories slice, 3 epochs (~291 steps), batch=8, len=128
- dim=96, n_enc_layers=2, n_patch_layers=1, patch_size=4
- seed=42 fixed for reproducibility

| arm | params | final loss (epoch 3 mean) |
|---|---|---|
| `n_dec_layers=2` (current ProtoBLT_RWKV) | 673,218 | **1.2032** |
| `n_dec_layers=0` (LN+Linear stub equivalent) | 431,490 | 1.4021 |

Δ = +0.199, or **+14.2%** worse without the decoder RWKV blocks.

This validates the architectural choice at `eb31908` — making the
decoder a real RWKV stack (not just LN+Linear) buys 14% loss
reduction at matched data, seed, and patch-slot. The decoder is doing
something the patch context alone doesn't reach: it time-mixes the
patch-context-per-byte information back into the byte-level recurrence
before being projected to logits.

Note: with n_dec_layers=0, we *still* have patch context injected via
`fuse` (the gated linear fusion before the head). So the gap is
specifically the byte-RWKV blocks *integrating* the fused representation
through the byte recurrence, vs. just adding it once.

Saved per-run log lives under `experiments/proto_blt_ts_001/`
(gitignored).
