# injection-frequency.status

Proof chain for injection frequency ablation.

## Claims

- **I1** — *per-layer fusion of core state into decoder improves recon loss over front-only fusion at matched ~228K params.* Train AdaptiveLoopModel front_fusion vs per_layer_fusion on byte_ts_001 for 2k steps, same seed, lr, patch_size=4. If per-layer loss < front loss - 0.05 and samples non-blank, proven. Status: **open**. Exp: `inj_freq_front_001` vs `inj_freq_perlayer_001`.

- **I2** — *gating is load-bearing for per-layer variant.* Same as I1 but compare per_layer with learned gate vs per_layer with fixed addition (h + core). If gated loss < ungated - 0.03, gate matters. Status: **open**. Exp: `inj_freq_gate_ablation_001`.

- **I3** — *per-layer injection causes decoder to use more adaptive loops.* Measure dec_loops stats: front should stay 1 (B5), per-layer may increase to 2-3 if global context needs deeper processing. Status: **open**, cheap metric from existing logs.

## Open follow-ups (cheap→expensive)

1. Test injection at encoder side too (core → encoder feedback)
2. Attention-style injection (query=decoder hidden, key/value=core) vs addition
3. Scale to >=1M params to see if gap widens
