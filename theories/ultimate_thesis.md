# Ultimate Thesis

What the project is for, in one paragraph per topic. No proofs in here — proofs live in `proofs.md`; per-theory proof status lives in `<theory>.md`.

## The thesis in one line

Small ML systems can be proven one claim at a time, faster than large systems can be proven as a whole.

## What the user kept saying

- **On scale and emergence.** "The ultimate goal is to produce something novel by doing small experiments. My ultimate theory is that machine learning can be done with a smaller system. And we can prove 1 thing at a time instead of waiting for emergent properties."
- **On the substrate.** "The core of the learning will be read-many-context-tokens answer with few but verifiable tokens." — the model eats lots of bytes, emits few tokens, and the tokens have a clear right-or-wrong check.
- **On speed of derivation.** "My ultimate goal is to make a latent that can induce many tokens. Instead of processing the whole context repeatedly and outputting 1 token ... I process the context and derive the future and after that rapidly decode the remaining tokens required." — latent state does the heavy work, the byte sequence is the captured future.
- **On the path.** "I prefer to start from the underlying assumptions." — assumptions are spelt out, not inherited.
- **On failure.** "Either fix your error ... or if the experiment proves the theory is invalid ... you can create a different theory that might unlock a different advantage." — failed theories are pivoted, not buried.
- **On concrete phases.** "A small pure BLT model → working. A small BLT encoder and decoder with a small RWKV → working. A RWKV based encoder and decoder with a normal transformer → working. A pure RNN based BLT model → working. In that order. Would be better." — incremental complexity.
- **On self-direction.** "Continue with your self-directed research." — once a working rule ladder exists, the AI picks the next step.

## Threads the project has open

### Architecture thread — bytes ↔ latent ↔ bytes

Three scaffold variants around the same idea:

- **byte-state-byte.md** — byte encoder → patch-level state model → byte decoder. Two parallel attempts on disk (step-function and RWKV-based).
- **b3d-rwkv-nano.md**, **diffusion-grid-terminal.md** — 2D spatial and diffusion variants.
- **movable-grid-scratchboard.md**, **screen-viewport-zoom-pan.md** — pointer-based variants where the model emits metadata to move a viewport.

### Memory thread — trunk + branches

- **dendrite_memory.md** — frozen trunk + many small LoRA adapters per memory. The LoRA framing was the user's first attempt; introspection later changed it.
- **dendrite_growth.md** — frozen trunk + many *small architectural modules* per memory. Same shape, different mechanism. Currently the forward-facing version.
- **delta-mem.md**, **progressive-expansion.md**, **read-twice.md** — related but distinct interventions on the trunk: stateful attention side-channels, layer-by-layer growth, recurrent passes.
- **rwkv-state-carry.md** — long-memory mechanism.

### Capability thread — adaptive compute

- **adaptive-exit-entropy.md** — adaptive loop on confidence thresholds.
- **injection-frequency.md**, **dynamic-patch-vs-fixed.md**, **token-vs-byte-head.md** — derivative ablations from the byte-state-byte proof chain.

### Application thread — what the work is for

- **realtime_ai.md** — continually-learning AI reading device streams.

### Methodology threads — how the work is done

- **working_method.md** — the operating principles.
- **smoke_test_methodology.md** — the 1-minute proof rule.
- **generating-loss.md**, **adaptive-exit-entropy.md** — variations on the loss signal.

## What this project is not doing

- Scale-up of any large model without a hypothesis. (Forbidden.)
- Multi-cause experiments. (Forbidden by `working_method.md`.)
- Re-opening retired lines from `theories/archive/` until the active threads are stable.

## Read order

For an outside reader: `ultimate.md` → this file (`ultimate_thesis.md`) → the live `theories/<topic>.md` of the thread you care about → the experiment in `experiments/<id>/`. For proof backlog: `theories/proofs.md`. For dashboard: `experiments/` and the `status` view from `_status.py`.
