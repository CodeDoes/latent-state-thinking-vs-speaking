# Theory 05 — Uniqueness-weighted loss

## Claim
Weight each query's loss by `w(a) = -log2 p(a)` (the **self-information** of
its answer). This near-zero-weights `"NONE"` (p≈0.87 → ~0.2 bits) and
up-weights WHERE (p=1/32 → 5 bits) and rare answers, **forcing the model to
learn the informative queries without hand-balancing the dataset.**

## Rationale
- Derived directly from T03/T04: the information a label carries equals its
  uniqueness. A uniform CE over-represents the frequent (non-unique) labels,
  which is exactly why the model drowns in NONE.
- Weighting by `-log2 p(a)` makes every bit of world information cost equally,
  so the optimizer cannot "save loss" by always-NONE.

## Supporting evidence
- Empirical: the unweighted run collapsed **exactly** to the NONE-cheat ceiling
  (0.619); latent scored 0.634. The loss had no pressure toward non-NONE.
- `p(NONE)≈0.87 → w≈0.2`; `p(WHERE location)=1/32 → w=5.0`. The weight ratio
  (~25×) is precisely the debiasing needed.

## Predictions (testable)
- With uniqueness weighting, overall accuracy should rise **above 0.619** (latent
  actually learns WHERE / non-NONE).
- AT/SAME frozen-accuracy should *drop* (the model stops cheating on NONE).

## Proposed experiments
- Compute `p(a)` over the generated corpus at startup; multiply each query's CE
  by `w(a) = -log2 p(a)`. Keep WHERE as the headline metric.
- Pair with auxiliary reconstruction (T06).

## Status
🟡 **Proposed fix — untested.** (The `p(a)` table and weighting are
straightforward to add to `train_converged.py` / `bench.py`.)

## Related
T03 (label non-uniqueness), T04 (normally-empty state), T06 (auxiliary loss).
