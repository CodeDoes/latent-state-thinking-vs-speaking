# progressive-expansion — underlying assumptions

This file extracts and enumerates every implicit premise behind the
progressive-expansion idea. Each assumption is tagged with its status so we
can decide which to prove first, which to accept provisionally, and which
to cut entirely.

## A1. Capacity pressure leaves observables

When a trained model encounters a stimulus it cannot represent, the failure
is not invisible noise — it produces **measurable signals** in activations
(over certain channels, over certain depths). Different kinds of failure
produce distinguishable signals.

**Status**: Partially supported by mechanistic interpretability work
(sparse autoencoders, activation patching), but mostly extrapolated from
hearing people talk about it. No controlled measurement at the scale we'd
run. This is the **foundational assumption**. If it's false, nothing else
follows.

**Refined stimulus selection**: Do NOT use completely alien stimuli.
Use stimuli that sit at the **edge** of what the model can already do —
they share representational structure with the training distribution but
demand more from the same channels. The key case: **partial degradation**,
where the model produces answers that are *almost* correct. Fully correct
answers = no capacity pressure. Fully random/wrong answers = model gave
up entirely and stopped trying. Partially correct = it *tried*, used the
right channels, and hit a wall. **That** is the signal.

**Minimal test**: Train a model on task A until converged. Present it
stimuli slightly harder than A (e.g., one more variable in logic-niiah,
or one extra hop in a chain query) — distributed so roughly half get it
wrong and half get it right. Measure per-channel activation statistics
on the failed/semi-failed samples and compare against the happy-domain
(A) distribution. Can you flag *which* channels saturated under partial
success but were well-behaved on easy examples? If yes, A1 holds.

**Risk**: You might measure *symptoms*, not causes. Channel X saturates
while channel Y carries the actual information. Fixing X does nothing.
(See §Danger Signs in `progressive-expansion.infer.md`.)

---

## A2. Bottleneck location encodes information flow topology

A channel group that simultaneously saturates/collapses under new inputs is
not randomly placed — it sits at a **junction point** where the network routes
information. It's already wired to upstream preconditions and downstream
postconditions. Inserting computation *at* that junction reuses existing
wiring; inserting it elsewhere would require rewiring.

**Status**: Pure hypothesis. This is the piece most likely to come from
other people's writing about modular architectures or circuit discovery. It
could be true (functional specialization creates real topological features)
or it could be an artifact of how simple networks self-organize (any wide
layer will show *some* channels saturating when pushed hard — that doesn't
mean they're junction points).

**Refinement — stable types, unstable locations**:
Channel numbers will *not* reproduce across training runs with different
seeds. The dataset programmes the capacity allocation but the exact channel
assignment is arbitrary — it's a symmetry of the optimization landscape.
What *will* be stable is the **functional role** of the bottleneck channels:
channels that encode "entity tracking" will appear in different positions
across runs but correlate with the same dataset-defined semantic signals.

Two implications:
1. A2 must be tested **within a single model**, not across runs.
   Insert a layer at the detected bottleneck and verify this *specific*
   instance of capacity pressure is relieved — don't demand cross-run
   reproducibility of channel numbers.
2. To verify the dataset-coding hypothesis: train the *same model architecture*
   on two different datasets. The bottleneck pattern should change
   *structurally*, not randomly. A shift from entity-tracking bottlenecks
   to relation-tracking bottlenecks under a different task distribution
   would show that the dataset programmes capacity allocation.

---

## A3. Bottleneck-driven expansion is faster than global retraining

An expanded model learns the new capability faster than you could achieve by
finetuning the entire original model (same total compute budget, same
training data). The gain comes from concentrated gradient flow — only the
new layer and perhaps a small neighborhood update.

**Status**: Plausible but untested. Could be false if: (a) old weights
interfere with new learning despite being frozen, (b) the bottleneck isn't
well-localized so gradient flow still spreads everywhere, or (c) the new
layer's parameters are dwarfed by the frozen weight matrix it passes
through.

**Control**: Finetune full model on new task, matched parameter count,
matched steps. Compare convergence speed.

---

## A4. Expansion preserves old capabilities (non-obstruction)

Inserting layers and training new parameters does not degrade performance
on the original task, provided the new layers are initialized as identity
and old weights are frozen during the new-task phase.

**Status**: Probably true in practice for shallow insertions (single layer
between existing layers), but becomes harder as the model grows. With
identity initialization and frozen weights, the network computes exactly
what it computed before — except now the backward pass sees extra params and
may push neighbors away from their optimum. This is **catastrophic forgetting
by gradient contamination**, not by representation erasure.

**Test**: Measure old-task accuracy before expansion, immediately after
training on new task (with frozen old weights), and after unfreezing all
weights for a few steps. If step 2 > step 3, freezing protects the old
capability. If step 2 < step 3 even without unfreezing, the act of adding
parameters itself distorts the landscape.

---

## A5. Detection must precede treatment (decoupling)

You cannot tune the bottleneck-detection procedure on the same samples you
later use for expansion, because that smuggles supervision into the
detection step. Detection must operate on a held-out or structurally
independent stimulus set.

**Status**: Tautological correctness — standard ML hygiene. The danger zone
is subtle: if your saturation threshold, sample count N, or statistical test
is chosen based on observing the target stimulus, you've partially overfit
to it. Detection and expansion are separate optimization objectives; the
former should not implicitly depend on the latter.

**Mitigation**: Set detection hyperparameters using old-task validation
performance only. Or use a completely different easy task whose distribution
you understand well enough to predict what "normal" looks like.

---

## A6. One signal type first

Channel saturation, gradient-norm cliffs, and cross-covariance degeneracy
are three *different* measures of capacity pressure. They may agree or
disagree on which channel is the bottleneck. Running all three simultaneously
creates ambiguity about which mechanism is responsible.

**Status**: Design constraint, not a claim about reality. Must enforce
this rule. Pick **one** signal metric for the first experiment.

---

## What to prove first, and why

| Priority | Assumption | Reason |
|----------|-----------|--------|
| 0 | A1 | Foundational — if capacity pressure has no observables, abort everything |
| 1 | A2 | Determines whether the whole premise of *targeted* expansion has any point |
| 2 | A4 | Determines whether the approach is viable at all (obstructive = useless) |
| 3 | A3 | Determines whether the approach is *useful* (worth doing vs just fine-tuning) |
| Etc. | A5, A6 | Meta-rules that apply regardless |

Start at A1. If A1 fails, there's no progressive expansion — just noise.
If A1 succeeds, proceed to A2. Don't reach ahead.
