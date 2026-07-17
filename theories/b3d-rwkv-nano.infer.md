# b3d-rwkv-nano.infer.md

Interpretation of b3d-rwkv-nano.md.

What the .md leaves implicit:

**Why clean copy b3 is needed**: RWKV's recurrent state `num, den, xx` accumulates history. After processing b1 and b2 (both masked), state is contaminated by mask tokens. b3 is clean ground-truth that "refreshes" state before next block, so next block's context starts from correct history, not masked history. Without b3, diffusion errors compound across blocks. This is the core trick that lets causal RNN simulate bidirectional block.

**Pseudo-bidirectional explained**: In a causal model, token at position j can only see <j. In triplet layout, b1 has some unmasked tokens scattered. By the time we reach same position j in b2, the recurrent state has already seen all unmasked tokens from b1 (including those *after* j within block but from b1). So when predicting j in b2, model indirectly sees future unmasked tokens via state carried from b1. That's why b2 gets bidirectional context.

**Matching compute is tricky**: Triplet expands sequence 3× (b1+b2+b3). If we naively train B for same steps as A, B sees 3× fewer logical blocks for same physical tokens. Paper compensates via shorter physical block and more steps. The .md says "reduce steps so total bytes processed equal" – meaning if A processes N bytes in S steps, B should process N/3 logical blocks in S steps but 3× physical tokens = same N physical. So to isolate layout, we must compare at equal physical tokens, not logical.

**Tau-threshold decoding**: At inference, commit when top-1 prob > τ. This is confidence-based. Low τ → commit fast but errors. High τ → slow but accurate. Paper uses τ tuned per task. At nano, we should sweep τ 0.8/0.9/0.95 and measure accuracy vs steps.

**Why smaller is non-trivial**: At 7.2B, RWKV has 4096 dim, 64 heads, 32 layers – large capacity to store b1 unmasked context in state. At nano dim=128, 3 layers, state is tiny (128*3). Can it still remember 8 unmasked tokens from b1 across b1→b2 distance (B=8)? This tests if per-channel decay learns long-memory at nano (see rwkv-state-carry theory). If fails, need larger dim or smaller B.

**Relation to existing B5 adaptive_loop**: AdaptiveLoop also has encode→core→decode with loops. Triplet-block is different: it's *same* RWKV reused for diffusion, not encoder/core split. But both share "think longer before commit" idea. Could combine: adaptive exit decides when to commit vs diffuse again.

**What negative means**: If B loss > A +0.1 at same compute, triplet layout hurts at nano – diffusion needs capacity to store bidirectional context. That would suggest B3D speedup is scale emergent, not architectural. Still publishable negative.

**Metrics beyond loss**: Should log mask ratio per iteration, steps per block (throughput), and per-position entropy H(p). Commit schedule (how many tokens committed per iter) tells if diffusion is parallel or degenerates to sequential (commits 1 token per iter = no speedup).

**Tool calling link**: Diffusion grid terminal theory (diffusion-grid-terminal.md) can reuse this model as substrate – grid is just 2D block.
