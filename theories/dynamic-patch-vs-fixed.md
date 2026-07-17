# dynamic-patch-vs-fixed

Architecture has second free variable: **patch boundary source**.

Current B5 uses fixed stride patch_size=4. BLT and byte-state-byte theory propose dynamic boundaries from surprise: `surprise_per_step` > threshold starts new patch.

Hypothesis: dynamic patching reduces latent count (better compression) without increasing recon loss, when core has enough capacity.

Why fixed might be suboptimal: natural byte sequences have variable entropy — word boundaries, punctuation are high-surprise (1.8+ per earlier smoke), double-letters low surprise (0.1-0.2). Fixed patch wastes latents on predictable spans and undersamples surprising spans.

Why dynamic might fail: `surprise_patcher.py` is parameter-free but non-differentiable (threshold + mean pooling). If threshold mis-tuned, patch_lengths variance explodes, leading to unstable training or rho collapse. Fixed is stable baseline.

Minimal test (single variable = patcher):
- Arm A: fixed patch_size=4 (existing, rho=4.0)
- Arm B: dynamic, threshold=0.7, min_patch=2, max_patch=8, tuned to avg rho ~4.0 to match compression budget
- Both: same AdaptiveLoopModel dim=64, enc/core/dec 2 layers, 228K params, 2k steps on byte_ts_001
- Metric: recon loss AND rho (compression). Win if B loss <= A loss + 0.02 AND rho >= A rho (or B rho 20% better at same loss)

If dynamic wins, claim: surprise-based patching adds value beyond fixed when core can use it (open follow-up #6 in byte-state-byte.status.md).

Single variable: patching logic. All else locked.
