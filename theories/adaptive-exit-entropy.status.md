# adaptive-exit-entropy.status

Proof chain for adaptive exit regularization.

## Claims

- **E1** — *entropy weight 0.01 prevents collapse to 1 loop, enables encoder loop adaptation 1→3 at 228K params.* Sweep 0.0 vs 0.01 vs 0.1. Proven if 0.0 collapses to 1 loop >90% steps, 0.01 shows >2 depths used >20% each, and loss 0.01 < 0.0 -0.05. Status: **open**. Exp: `adapt_ent_0.0_001`, `adapt_ent_0.01_001`, `adapt_ent_0.1_001`.

- **E2** — *decoder loop usage responds to entropy weight.* If at high weight (0.1) decoder uses >=2 loops vs 1 at low weight, proves decoder not task-limited but regularization-limited. Status: **open**, metric from E1 logs.

- **E3** — *gate learns input-dependent exit, not constant.* Measure lambda variance across batch and correlation with surprise. Proven if var>0.01 and corr(surprise, dec_loops) >0.2. Status: **open**.

## Follow-ups

1. Learned per-layer entropy weight (not global)
2. Gumbel-softmax sampling vs expected loss (exploration difference)
3. Scale to >=1M params where deeper loops may matter more
