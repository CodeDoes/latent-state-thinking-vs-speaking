# Decoder ablation analysis

Using commit `317f4a5` as cited. Re-check the actual numbers from the in-thread runs.

The ablation was: same encoder + patch slot + data, only `n_dec_layers` varied.
- n_dec_layers=2: 673,218 params
- n_dec_layers=0: 431,490 params
- Final loss mean (epoch 3): 1.2032 vs 1.4021 — Δ = +0.199, or +14.2% worse
  without decoder blocks.

Conclusion: decoder RWKV blocks are doing real work. Documented at
`theories/decoder-ablations.md`.

## What the ablation does NOT claim

- Decoder RWKV is the *only* signal contributor. Same data, same encoder,
  same patch slot — decoder is the *only* difference — but the model
  is so small (~673K) that we can't attribute 14% to the full RWKV stack
  vs. just the additional capacity of n layered blocks (any architecture
  would gain from doubling depth).
- Decoder is the right *place* to integrate patch context. We compared
  decoder-with-RWKV-blocks vs decoder-with-LN+Linear. Both still consume
  h_byte + h_patch_per_byte at the same single point (one residual
  fusion). What we need to test is *interleaving* patch context through
  every block vs. fusing once at the front.

## Followups worth running

When the GPU is free:
1. **Decoder with patch context injected per-block.** Like cross-attention
   at every decoder layer instead of one-shot residual.
2. **Encoder ablation.** Same idea but for `n_enc_layers=0`. Encoder is
   presumably also load-bearing; if it isn't, we can shrink the model.
3. **Patch-slot ablation.** Fix window for n_patch_layers, but vary
   n_patch_layers from 0 to 3 with everything else matched.

These three together give us the full decomposition of where loss comes
from in the architecture. Until then, +14% is *one* arm of a 4-variable
ablation grid.
