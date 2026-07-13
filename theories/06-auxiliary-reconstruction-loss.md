# Theory 06 — Auxiliary state-tracking (reconstruction) loss

## Claim
Add an auxiliary head that predicts each item's **final location** from the
latent state `s` (JEPA-style state-evolution / reconstruction loss). This forces
`s` to actually encode per-item trajectories, fixing the **WHERE failure**
(Bug B: the model can't learn the informative query even when its label is
unique).

## Rationale
- WHERE answers are *always* unique (0% NONE, 5 bits) yet latent scores
  ≈random → the state **discards** trajectory info during "think once".
- An auxiliary reconstruction loss gives a gradient that populates even "empty"
  slots with *negative* information ("item X is at L7, not here"), turning the
  Normally-Empty collapse-trap (T04) into learnable structure.
- Matches AGENTS.md training objectives: latent consistency / reconstruction /
  state-evolution losses — not just next-token prediction.

## Supporting evidence
- WHERE is high-entropy (5.00 bits) but latent WHERE ≈ 0.02–0.05 (random);
  baseline WHERE 0.14–0.22 (it re-reads the source).
- Bigger `d_state` (32→48) did **not** help WHERE (0.018) → not a capacity
  problem (see T07); the state simply isn't *trained* to track trajectories.

## Predictions (testable)
- With reconstruction loss, latent WHERE should climb above random and approach
  baseline's 0.14–0.22.
- State should become discriminable even for "empty" locations.

## Proposed experiments
- Auxiliary `location_predictor(s) → per-item final location`, trained alongside
  the main objective. Combine with uniqueness weighting (T05).

## Status
🟡 **Proposed fix — untested.**

## Related
T02 (ssm-vs-tape split), T04 (normally-empty state), T07 (capacity refuted).
