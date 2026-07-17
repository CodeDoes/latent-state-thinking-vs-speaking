# dynamic-patch-vs-fixed.status

Proof chain for dynamic vs fixed patching.

## Claims

- **D1** — *dynamic surprise patching maintains recon loss at matched compression rho~4.0 vs fixed.* Arms: fixed patch_size=4 vs dynamic threshold 0.7, both at 228K params, 2k steps. Proven if dynamic loss <= fixed +0.02. Status: **open**. Exp: `dyn_patch_fixed_001` vs `dyn_patch_surprise_001`.

- **D2** — *dynamic patching improves compression (higher rho at same loss) when threshold tuned.* Same setup but sweep threshold 0.5,0.7,0.9 to find rho where loss matches fixed. Proven if there exists threshold where rho >= 5.0 at loss <= fixed+0.02 (25% better compression). Status: **open**. Exp: `dyn_patch_threshold_sweep_001`.

- **D3** — *warmup stabilizes dynamic.* Compare dynamic from step 0 vs fixed→dynamic switch at step 200. If warmup loss < cold-start -0.05, proven warmup matters. Status: **open**.

## Open follow-ups

1. Learned entropy model vs parameter-free surprise delta
2. Dynamic patching at scale >=1M (does core use extra flexibility?)
3. Patch visualization: correlate patch boundaries with tokenization
