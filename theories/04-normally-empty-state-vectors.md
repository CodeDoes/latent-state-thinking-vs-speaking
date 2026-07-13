# Theory 04 — Normally-Empty latent-state vectors

## Claim
In a *structured* latent state (e.g. `[location-slots]` or `[item-slots]`),
~87% of slots are **empty** in a typical world, so the ground-truth
slot-targets are 87% "empty". The slot-prediction gradient is therefore
dominated by emptiness, and the model collapses toward an empty / ambiguous
representation — it cannot distinguish *"this slot is genuinely empty because
item X is elsewhere"* from *"I collapsed and don't know"*.

## Rationale
- The 87% slot-emptiness **mirrors** the 87% NONE-label rate: same structural
  sparsity viewed at the *state* level vs the *label* level.
- Without explicit empty-handling, the majority target ("empty") dominates the
  loss exactly as `"NONE"` dominates the answer loss (T03).

## Supporting evidence
- Measured location-slot emptiness = **0.870** over 5000 worlds (32 location
  slots, ~5 items each → each location holds ~5/32 items).
- This matches AT/SAME NONE rates (0.869 / 0.894) — the answer distribution
  reflects the world's structural sparsity.

## Predictions (testable)
- A slot-structured model trained without empty-handling will collapse to empty
  predictions; the collapse rate will track the structural emptiness rate.
- Giving the state an explicit "empty / occupied" representation (or an
  auxiliary reconstruction loss, T06) should turn the 87% emptiness from a
  collapse-trap into learnable negative structure.

## Proposed experiments
- Auxiliary per-item final-location reconstruction from `s` (T06).
- Compare slot-structured vs single-vector state under identical loss.

## Status
✅ Rate **confirmed**; architectural implication **untested** (no slot-structured
model trained yet).

## Related
T03 (label non-uniqueness), T05 (uniqueness-weighted loss), T06 (auxiliary loss).
