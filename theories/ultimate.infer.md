# ultimate.infer.md — interpretation of theories/ultimate.md

This file fills in what `ultimate.md` leaves implicit.

---

## The claim

`ultimate.md` is a **methodology claim**, not an architecture claim. It
asserts that progress in ML can be made by:

1. **Small systems** — orders of magnitude smaller than the field's
   defaults.
2. **Small experiments** — one hypothesis, one run, one proof.
3. **No emergence** — every claimed property must be independently
   provable, not the side-effect of scale.

The combination is:

> *A novel result can be produced by isolating one mechanism, proving it at
> the smallest scale where the proof is meaningful, and repeating.*

This is the opposite of "throw more compute at it and see what sticks."
It is closer to physics-style experimental design than to deep-learning
flavor-of-the-month research.

## What "smaller" means (unstated)

The claim is not "use fewer parameters arbitrarily." It is "use the smallest
system that still sustains the proof."

- **Smallest viable system**: one where the property being tested is *the
  only thing the system has to do*. If the property requires breadth
  (general language, general reasoning), shrinkage is impossible and the
  claim has no home. If the property is narrow (associative recall, world
  state tracking, loss on predicted-wrong-tokens), shrinkage is unlimited.
- **Small as a forcing function**: every additional parameter, layer, or
  head must be justified by a specific failure or capability gap. If it
  can't be tied to a measurable effect on the target property, it's
  bloat.
- **Compute, not just params**: a small system is also short to train. CPU,
  minutes, not GPUs and days. This is what makes the iteration loop fast
  enough to actually repeat the *prove-one-thing* cycle.

## What "1 thing at a time" means (unstated)

A single experimental unit tests a **single causal claim**. Concretely:

- **One independent variable** between matched runs (architecture, loss,
  data distribution, decoding strategy — *exactly one*).
- **All other variables held constant** (params, data scale, training
  budget, eval).
- **The dependent variable is a single measurable property**, not a
  hand-wavy "the model is better."
- **The result must be reproducible from the run's git hash**.

A multi-thing experiment ("we scaled X, changed Y, and rebalanced Z")
produces a number that no one — including the author — can attribute to a
cause. It is the equivalent of a confounded study in biology.

The hidden rule: **if a run improves over baseline, you must be able to say
*which* change made it improve.** If you can't, the run is throwaway
regardless of the final number.

## What "not waiting for emergent properties" means (unstated)

Emergence is the field's escape hatch: "we threw more data and parameters
at it, and now the model does X, and we're not totally sure which part
of which layer is responsible."

The user's claim rules this out. Every demonstrated capability must come
with:

- **A mechanism** — a story for *which part* of the system produces it.
- **A minimum scale** — the smallest configuration where the capability
  still appears.
- **A failure mode at one remove** — a controlled experiment that
  disables the proposed mechanism and shows the capability degrades in
  the predicted direction.

If a property doesn't survive all three, it's emergent, not proven.

## What "novel" means in this frame (unstated)

Novel has a specific meaning here. Not "new architecture," not "new
benchmark," but:

> *A new, narrowly stated, falsifiable claim about what computation is
> sufficient and necessary for a named capability.*

Sufficiency and necessity are both required:
- **Sufficiency**: this small system achieves the capability.
- **Necessity**: removing the named mechanism fails the capability at the
  same scale.

A result that only proves sufficiency is a demo. A result that proves both
is a contribution.

## How this connects to the project's previous work

The existing `exp001` matches this template exactly:

- One change: latent-state "think once" vs re-encode-per-query baseline.
- Same params (~7.6k vs ~7.5k), same data, same training budget.
- One number: overall accuracy, broken out by query type (WHERE / AT /
  SAME).
- Result attributable to a single cause: the recurrent latent state.

It is the smallest possible proof-of-method for the thinking/speaking
distinction. It does not yet address **necessity** — there is no ablation
that shows *which part* of the latent state is doing the work. That is the
natural next prove-one-thing experiment in this frame.

## What this implies for the file structure

Every future theories file should answer one question:

- *What is the single thing this proves, at what minimum scale, by what
  controlled comparison?*

If it doesn't answer that, it's not a theory under this methodology —
it's a speculation or a survey.

## What we are explicitly *not* doing

- **Chasing SOTA**: no leaderboards, no "matches GPT-4 on X," no
  surrogate wins.
- **Scale as a substitute for understanding**: a 1B-parameter run that
  doesn't isolate a mechanism is *worse* than a 10k-parameter run that
  does.
- **Multi-thing A/B tests**: every ablation is single-variable with the
  baseline locked.
- **Retrofitting explanations**: a result must come with its mechanism
  proposed *before* (or at worst parallel to) the run, not invented
  afterwards to fit the curve.

## What "novel" looks like in practice

A successful contribution from this project would look like:

> *We prove that [property P] is sufficient and necessary for [capability C]
> at scale [S], demonstrated by [experiment E], and rejected the
> alternatives [A1, A2, A3] at the same scale.*

Small. Specific. Reproducible. Citation-form-ready.

That is the bar.
