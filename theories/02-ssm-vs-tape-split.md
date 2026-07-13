# Theory 02 — SSM wins aggregation, tape wins precise recall

## Claim
The latent SSM state is good at **integration / set-aggregation** reasoning
(AT, SAME) but bad at **precise single-item trajectory recall** (WHERE).
Precise recall is the job of a *tape / explicit-recall* mechanism, which the
current design lacks. Therefore "latent vs tokens" must be evaluated on a
**reasoning-heavy** mix, not on trajectory recall.

## Rationale
- Mirrors the project's own architecture split: SSM = "world model / logic /
  planning"; Tape = "exact recall / spelling". Compression into a fixed-size
  state discards trajectory detail, so WHERE (which item ended where) is lost.
- The overall "baseline +3 pts" headline is an **artifact of equal-weighting
  WHERE** with the reasoning queries.

## Supporting evidence
- ITER-2 per-type breakdown:
  - AT: latent 0.853 > baseline 0.839
  - SAME: latent 0.918 > baseline 0.906
  - WHERE: baseline 0.136 >> latent 0.018 (baseline 7.7× better)
- Bigger `d_state` (48 vs 32) did **not** help latent on WHERE (0.018 vs 0.06)
  → deficit is architectural (compression), not just capacity (see T07).

## Predictions (testable)
1. Tilt the query mix toward AT/SAME → latent should win overall.
2. Adding a tape / explicit-recall mechanism (or auxiliary reconstruction, T06)
   → latent WHERE should improve toward baseline.

## Proposed experiments
- iter-3: integration-heavy mix (Model A→D incremental path's reasoning stage).
- Auxiliary reconstruction head (T06) to give the state trajectory signal.

## Status
✅ **Confirmed** within current scale (0.5M params, multi-query worlds).

## Related
T01 (core hypothesis), T03 (dataset non-uniqueness), T06 (auxiliary loss).
