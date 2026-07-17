# adaptive-exit-entropy.infer

Interpretation of adaptive-exit-entropy.md.

What .md leaves implicit:

**What AdaptiveExitGate actually does**: At each loop step r, outputs λ_r = sigmoid(W·h). Cumulative exit prob π_r = λ_r * prod_{k<r}(1-λ_k). Training loss is expected loss Σ π_r * L_r minus entropy H = -Σ π_r log π_r * weight. This is REINFORCE-ish without sampling (expected). Inference exits when mean λ >0.5.

**Why entropy needed**: Without entropy bonus, model can set λ_1≈1 immediately (exit after 1 loop) if first loop already decreases loss somewhat, even if second loop would help more. Entropy forces distribution to stay somewhat uniform initially, exploring deeper loops.

**What collapse looks like**: If λ_1→1, π_1→1, all other π→0, H→0. Loss = L_1. Model never sees gradient for deeper loops because their π≈0 (no gradient). Irreversible.

**Connection to B5**: B5 report shows enc_loops 1→3 adapt during training, suggesting entropy worked. But decoder stayed 1. Is that because decoder task easy (byte reconstruction single pass enough) or because decoder gate collapsed early? This sweep answers: if increasing weight causes decoder to use >1 loop, then decoder 1-loop is not task-necessary but optimization artifact.

**What to log beyond loops**: gate weight norm (W norm) tells if gate is learning. Exit_lambdas per layer mean var. If mean var low (all gates ~same), gate not input-dependent (failure). Good gate should be input-dependent: lambda higher for easy patches, lower for hard.

**Cheap metric**: Can measure correlation between surprise_per_step and exit lambda: high surprise should cause higher loop count (lower early λ). If correlation positive, gate learned to "think longer" on surprising bytes – validates thinking/speaking frame.

**Risk**: Sweeping entropy changes loss scale, so need to compare recon loss separate from entropy term. Log recon_loss alone, not total.

