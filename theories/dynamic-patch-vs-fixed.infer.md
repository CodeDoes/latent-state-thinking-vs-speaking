# dynamic-patch-vs-fixed.infer

Interpretation of dynamic-patch-vs-fixed.md.

What .md leaves implicit:

**Matching compression is non-trivial**: Fixed patch_size=4 gives deterministic rho=4. Dynamic threshold=0.7 was measured in gpu_proto_blt_001 to produce ~32 latents for 128 bytes (rho=4). But threshold → rho mapping is data-dependent and shifts during training as encoder improves. So to truly match budget, we must either (a) sweep thresholds post-hoc to achieve target rho, or (b) use adaptive threshold that targets rho (e.g., percentile of surprise). .md chooses (a) for simplicity.

**Surprise signal source matters**: surprise_per_step uses encoder hidden delta |h(t)-h(t-1)|, not loss-based entropy. This is already available from B5 encoder (no extra model). BLT trains separate entropy model; we avoid that by reusing encoder's own receptance. Risk: early in training encoder is random, so surprise signal is noise → patch boundaries random → unstable. That predicts dynamic should be paired with warmup (first 200 steps fixed, then switch to dynamic).

**Why this was never run before**: src/surprise_patcher.py exists but no training script sets dynamic_patch=True in successful runs. byte_ts_001 experiment tracked fixed only. The .md theory is resurrecting code that is currently dead.

**What negative result would mean**: If dynamic loss > fixed +0.05, it suggests parameter-free surprise is insufficient; need learned patcher (small LM) as BLT does. That would pivot to training separate entropy model (costly) — so negative is informative.

**Metrics beyond loss**: Should log patch_lengths histogram, surprise mean/var per step, rho variance. If rho var high (>2.0 std), dynamic is unstable even if mean matches.

**Observable**: The theory predicts word boundaries get shorter patches (good for decoder). Can verify by sampling: does dynamic create patch break at spaces? Check via sample dump.
