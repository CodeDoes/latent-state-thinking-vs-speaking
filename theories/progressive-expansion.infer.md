# progressive-expansion — infer

This document interprets the intuitive leaps and unstated assumptions
behind [`progressive-expansion.md`](progressive-expansion.md). It never
repeats content already in the source `.md`.

## What you're *really* doing vs what you said

You described it as "monitor activations for a bottleneck → expand there."
Beneath that phrasing sits a sharper hypothesis: **capacity pressure manifests
as observable signatures**. Not vaguely "something interesting happens" —
specific, measurable signals that tell you *which channel*, *which layer*,
and *what kind* of representational gap appeared. The expansion is not blind
growth; it's targeted surgery.

### What that means concretely

When a trained model sees a stimulus it can't handle, the failure propagates
differently depending on *where* the information gets stuck:

- **Channel saturation** — a neuron cluster drives to hard sigmoid/relu
  limits for all inputs in the new-task distribution but was active-range
  under the original task. Capacity is exhausted, not absent. Solution:
  widen or bypass that channel group.
- **Gradient cliff** — the backward pass shows norms collapsing past a certain
  depth, meaning earlier layers have nothing useful to send downstream.
  Signal degradation, not representation gap. Solution: insert a learnable
  shortcut between layers.
- **Cross-talk interference** — activations look fine individually but the
  *joint* covariance structure is degenerate when both old-task and new-task
  samples are mixed. Two representations are fighting for the same basis
  vectors. Solution: orthogonal projection head before merging back.

These aren't three different hypotheses. They're the same underlying idea
seen through three different measurement instruments. Pick one instrument
for the first experiment.

## The unspoken constraint

You said the expansion "would not be obstructive." What you mean, implicitly:
**existing capabilities must hold during and after insertion**. This is the
hard part. Any layer inserted between two trained layers will change the
compositional mapping unless you either (a) initialize it as identity and
freeze it while training the new stuff, or (b) inject the new-task signal
only through a side-branch that merges *after* the critical path.

Method (a) is the clean ablation. Method (b) is the pragmatic hack. Start
with (a).

## The real question worth proving

Everything else is implementation detail. The one thing that matters:

> Does identifying a capacity-pressure signature via activation analysis
> produce a *better* expansion point than arbitrary insertion?

That is, if you insert a new layer at the detected bottleneck location
versus inserting it at the top of the stack versus inserting it randomly,
does the bottleneck-inserted model learn the new capability faster / more
completely / without degrading the old?

If the answer is "yes," you've proven a mechanism. If "no," the whole
activation-monitoring premise is noise-collecting theater.

## Why a bottleneck marker makes sense (mechanistic intuition)

A well-trained network develops **functional specialization**: certain
channels become tuned to carry certain kinds of information reliably. A
bottleneck — a set of channels that all saturate or collapse simultaneously
when new data arrives — is a **named interface** that happened to emerge
from optimization. It's already wired to the right preconditions and
postconditions because the rest of the network routed through it.

Inserting new computation *at* that named interface reuses the existing
wiring. You're not grafting onto random tissue; you're plugging into a
junction the network already treats as important. That's why convergence
should be faster: the gradient signal already flows that way, and the
upstream/downstream neighbors are already conditioned to expect traffic
through that bottleneck.

## Sensitivities to watch

| Variable | Expected effect | How to measure |
|----------|----------------|----------------|
| Bottleneck detection granularity (layer-level vs channel-level vs token-position) | Coarser detection → slower targetability, but more robust | Compare expansion speed at each granularity |
| Sample diversity used to detect bottlenecks | Too few samples → false positives (noise looks like saturation) | Track false-positive rate on held-out old-task samples |
| Expansion width (new layer d_model relative to parent) | Overwide → swamps old signal; underwide → still bottlenecks | Plot old-task accuracy vs new-layer width post-expansion |
| Freezing vs fine-tuning old weights during expansion | Freezing → clean attribution but slower total learning; fine-tuning → faster but conflated gradient signal | Two arms: freeze half, unfreeze half |

## Danger zones (anti-proofs)

- **"It works!"** without a control where identical expansion is done at a
  random point. No baseline = no evidence the bottleneck mattered.
- **Conflating correlation with causation.** Just because channel X saturates
  during new-task inference doesn't mean X is the *cause* of failure. It
  could be a victim. The ablation has to **remove** the suspected bottleneck
  artificially and see if performance drops regardless of input.
- **Overfitting the detection step.** If you tune bottleneck-detection
  hyperparameters on the same samples you later expand on, you've smuggled
  in supervision. Detection and expansion must be decoupled.

## Minimal experiment skeleton

Not yet formalized into the status file. Proposed shape (not approved):

1. Train a fixed-depth model on task A until convergence.
2. Run N samples of task B (simple, known-to-be-harder, e.g. one extra
   variable in logic-niiah). Collect activation magnitudes per-channel,
   per-layer.
3. Flag channels exceeding a saturation threshold OR showing maximal
   gradient-norm change vs task A. That's your bottleneck set.
4. **Arms:** (a) Insert new layer *at* bottleneck location, (b) insert at
   stack top, (c) no insertion (original architecture, harder task). All
   else matched.
5. Train only the new layer (arm a/b) on task B for T steps. Measure:
   - New-task accuracy improvement vs arm c (is the expanded model strictly better?)
   - Old-task accuracy retention (non-obstruction metric)
   - Training convergence speed (steps-to-target vs baseline)
6. Single outcome number: Δ(new-task accuracy) − Δ(old-task degradation),
   compared across arms.

To make this actionable, you'd pick the **one** bottleneck signal to
instrument first (channel saturation seems the cheapest to measure) and
hold everything else constant.
