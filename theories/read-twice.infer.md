# read-twice.infer.md

> **Source**: `read-twice.md` (verbatim).  
> **Date**: 2025-07-17

---

## What this theory is doing

A **alternative to parameter expansion** when capacity is pressured:

> *"i think RNN's can do 'read twice' instead"*

Instead of *inserting new layers* (progressive expansion), **run the input through additional recurrent forward passes**. Same weights. Same model. Extra compute depth, not new params.

This is "thinking longer" instead of "thinking wider."

## What the .md assumes vs. declares

| Claim | Status in .md |
|---|---|
| Two recurrent passes can substitute for one new layer (at matched compute) | Hypothesis (math unproven) |
| Bottleneck metrics from progressive-expansion *predict* the number of extra passes needed | Hypothesis (correlation claim, not measurement) |
| Read-twice does not disturb existing capabilities | Hypothesis (no new weights → no interference claim) |

## Latent assumptions the .md leaves implicit

- "Passes" are well-defined semantically (forward → state → forward → state → ...).
- The bottleneck-detection metrics are **applicable** to this case (re-using PE metric for read-twice).
- "Mathed compute" means *total FLOPs*, not *total wallclock*. Read-twice uses the *same parameters twice*; new-layer replaces one pass with one larger-pass + one new-layer-pass — these are *not equivalent in wall time* unless carefully counted.
- Pass-depth control is *gating* the model at inference time, not retraining. The model should learn *when to stop*? Or always 2 passes? Or always Kmax passes?

## Connection to existing work in repo

| Theory | Relationship |
|---|---|
| `progressive-expansion.md` | Sister theory: pass-depth (this) vs layer-depth (that). Both are surgical interventions when capacity is pressed. |
| `adaptive-exit-entropy.md` | Read-twice is the *opposite* exit signal: instead of *exit early when confident*, hold *longer when necessary*. |
| `dendrite_growth.md` | Perspective check: pass-depth ammortizes over the *same trunk*; dendrite-growth has *modular trunks* per branch. |
| `adaptive-exit-entropy.infer.md` (U4 hypothesis) | U4 is "adaptive looping > fixed length." Read-twice is the *special case* where the loop bound is parameter-free. |

## What I would test first

1. **Sanity.** Train one RNN on task A (single pass). Then test 1, 2, 3, 4 passes on task B. Plot accuracy vs pass-depth. If monotonic-increase goes too long before plateauing, the hypothesis survives trivially. (Doesn't yet prove "vs new layer.")
2. **Matched comparison.** Same compute budget. Round 1: progressive expansion adds 1 layer. Round 2: passes the input twice. Compare final accuracy on the *hard task*. This is the key experiment.
3. **Per-example gating.** Set pass-depth on a per-input basis, controlled by bottleneck-metric prediction. Compare to uniform-pass-depth. Verifies or falsifies the "metric predicts depth" claim.

## Open questions (from .md)

1. **Monotonicity**: does more passes always help? Risk: over-iteration starts unlearning.
2. **Per-example early exit**: can the model learn when to stop?
3. **GRU vs RWKV**: works for both? Or recurrency-specific?

## Status

Pure theory. No implementation yet. Minimal test skeleton proposed in .md. Cheap: CPU-only, small models, hours-of-effort per arm. Recommend running the matched comparison **before** committing to the per-example early-exit variant.

## Hypothesis Statement (Refined)

**R1:** A model with read-twice can match the accuracy of a model with one additional layer, at matched-compute. *True:* read-twice wins (no new params, no new training, no risk to old capabilities). *False:* parameterized depth wins per-FLOP.

**R2:** The progressive-expansion bottleneck metrics *predict* the number of extra passes needed. *True:* these metrics are interpretive signals, not just diagnostic — they directly prescribe compute budget.

**R3:** Matched-conditions: read-twice without per-input gating strictly ≥ single-pass by enough margin to justify the extra forward pass.
