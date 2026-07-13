# Theory 01 — Latent thinking beats tokens (core hypothesis)

## Claim
A model with `latent_state_update() × N + decode_token() × M` can match or
outperform an **equal-capacity** token-by-token model on long-horizon
reasoning, because *thinking* (the SSM/recurrent state) is amortized over
many queries while *speaking* (token generation) is cheap.

## Rationale
- A token is a poor clock cycle for reasoning; the latent state is the
  scratchpad / private computational space, and language is only an output device.
- **Efficiency claim:** latent thinks ONCE from the source, then answers N
  queries from a fixed-size state; the baseline re-encodes the full source for
  every query (N× encode cost). The win is supposed to be *efficiency at scale*
  + *resilience of a persistent state*, not raw accuracy on a single query.

## Supporting evidence
- ITER-1/2 (`exp_converged_lh_2026-07-13`): latent reaches ~0.63 exact-match,
  within ~3 pts of an autoregressive baseline **despite a 40% parameter
  handicap and a lossy compression disadvantage**.
- Per query type (ITER-2): latent **wins** AT (0.853>0.839) and SAME
  (0.918>0.906); loses WHERE (0.018 vs 0.136).

## Predictions (testable)
1. At many queries/world, latent's amortization edge should make it pull ahead
   on cost/accuracy.
2. On genuinely long-horizon tasks the latent state should beat the baseline
   (whose per-query re-encoding degrades with source length).
3. At the "first-night" scale (~20M params, 4 thinking steps) the gap should
   favor latent.

## Proposed experiments
- iter-3: integration-heavy query mix (tilt toward AT/SAME) → expect latent to
  win overall (confirms "latent wins on reasoning").
- Scale to ~20M params on GPU (currently blocked historically by P100 torch
  hang; now unblocked by local RTX 2050 — see T08).

## Status
🔶 **Nuanced, not refuted.** Level 1 in flight. Latent is competitive
(≈tie with baseline-edge) and wins on reasoning-type queries.

## Related
T02 (ssm-vs-tape split), T07 (capacity not bottleneck).
