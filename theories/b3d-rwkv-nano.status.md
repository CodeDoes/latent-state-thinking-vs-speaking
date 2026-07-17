# b3d-rwkv-nano.status

Proof chain for triplet-block diffusion RWKV at nano scale.

## Claims

- **BD1** — *triplet-block layout trains at ~228K params without loss vs causal AR at matched physical tokens.* Same RWKVNano dim=128/3L, B=8 logical, mask 0.3, 2k steps. Proven if diffusion CE loss ≤ AR CE +0.1 on byte_ts_001 or logic_niiah eval. Status: **open**. Exp: `b3d_ar_001` vs `b3d_triplet_001`.

- **BD2** — *triplet-block enables parallel decode (fewer forward passes than AR) at nano.* Measure steps to decode 128-byte block: AR needs 128 passes (one per token). Diffusion needs iter * 1 pass per iteration but commits multiple tokens per iteration if τ high-conf. Proven if avg iter ≤ 80 (1.6× speedup as paper) at τ=0.9 with accuracy ≥ AR -0.05. Status: **open**. Exp: same as BD1 inference with τ sweep `b3d_decode_tau_001`.

- **BD3** — *clean copy b3 is load-bearing.* Ablate b3: train variant with only b1+b2 (no clean refresh). Proven if no-b3 variant loss > with-b3 +0.2 or shows compounding errors across blocks (later blocks worse). Status: **open**. Exp: `b3d_no_b3_001`.

- **BD4** — *pseudo-bidirectional via b1→b2 state carry works at nano.* Measure that model in b2 can predict masked token that depends on future unmasked token within same block (from b1). Create synthetic block where answer requires future token. If b2 accuracy > b1 accuracy +0.2, bidirectional effect proven. Status: **open**.

## Open follow-ups (cheap→expensive)

1. Sweep B=4,8,16,32 – throughput vs accuracy tradeoff at nano
2. Combine with adaptive_exit: let gate decide τ per block (adaptive diffusion)
3. Apply to 2D grid (diffusion-grid-terminal.md) – grid is just blocks
4. Scale to 1M params to see if gap to AR closes
5. Test diffusion + injection-frequency (per-layer core fusion) together

## Related theories

- `rwkv-state-carry.md` – state carry is needed for b1→b2 pseudo-bidirectional; proves long-memory channels
- `diffusion-grid-terminal.md` – grid diffusion uses B3D as substrate for screen
