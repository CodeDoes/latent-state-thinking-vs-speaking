# Theory 03 — Label non-uniqueness drowns the signal (NONE collapse)

## Claim
The dominant training failure is **label non-uniqueness**: AT/SAME answers are
86.9% / 89.4% `"NONE"`, a *non-unique* label carrying ≈0 bits of
world-discriminating information. Cross-entropy is therefore minimized by
always saying `"NONE"`, so the model collapses to majority-class cheating with
**no gradient toward real tracking**.

## Rationale
- `"NONE"` is the correct answer for the vast majority of worlds, so predicting
  `"NONE"` is right for most worlds but teaches nothing about the *specific*
  world. The label itself is degenerate.
- This is the AGENTS.md "low loss but useless output" trap, made measurable.

## Supporting evidence
- NONE-rates: AT 0.869, SAME 0.894, WHERE 0.000 (measured over 5000 worlds).
- Answer entropy (bits): WHERE 5.00, AT 1.43, SAME 1.17.
- **NONE-cheat accuracy ceiling = 0.619** (always-NONE). The `train_converged`
  latent model scored **0.634 ≈ cheat** → learned nothing beyond majority class.
- Latent `L·AT` was *frozen at 0.8701 across all 9 epochs* — exactly the
  always-NONE value → constant-output collapse confirmed.

## Predictions (testable)
- A model trained on this data hits ≈0.619 by always-NONE; AT/SAME accuracy is
  frozen across epochs; only WHERE (0% NONE) carries learnable signal.

## Proposed experiments
- Measure NONE-rate / entropy per type (done, 2026-07-13).
- Apply uniqueness-weighted loss (T05) → accuracy should rise above 0.619.
- Balance AT/SAME to non-empty sets → remove the cheat.

## Status
✅ **Confirmed** (rates, entropy, cheat-ceiling, frozen-accuracy all align).

## Related
T04 (normally-empty state), T05 (uniqueness-weighted loss), T06 (auxiliary loss).
