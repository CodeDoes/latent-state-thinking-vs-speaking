# Realtime AI

A side project with a different end from language modeling: AI that never stops learning, fed by streams of PC telemetry.

## What was asked for

- An AI you can use in realtime.
- It must keep learning — not a frozen model that you hope was trained well enough.
- Concrete target: combine all the device signals on a PC (Wifi, Bluetooth, anything else) into usable answers. Things like "is there a phone nearby," "is anyone using wifi," "what are my devices doing."

The information is "meta-information that is normally hidden" — signals that are present but not surfaced in any current UI.

## Why not a normal ML model

A normal model trains on a dataset, then inference is a forward pass. There is no slot for new observations during use, and no notion of "check the answer" — when the model spits out "yes, there's a phone nearby," there's no oracle to confirm or deny that. The user explicitly named this as a requirement: a way to check its answer.

The workspace matters here because:
- The training has to be fast enough that it converges on CPU with consumer hardware (which it does for small RWKV-nano models).
- Patience for an unproven model is low — the user was clear "i do not have the patience to train a unproven model yet" and wants to validate at small scale first.
- Realtime + constantly-learns = continual learning regime, which is genuinely unsolved for non-trivial models.

## Possible architecture shape

A small piece of this is implementable today, with caveats:

1. **Ingestion**: each device stream becomes a byte channel (or feature channel if features are easier). Concatenated or interleaved for the encoder.
2. **Encoder**: byte-level RWKV with the surprise/trigger mechanism already in `byte-state-byte.md` work. Inserts a patch token whenever the entropy signal fires.
3. **State**: a persistent state buffer carries across calls. Read-many-context, write-few-tokens. The state *is* the memory.
4. **Same trunk, multiple branches**: routes signal into multiple "question heads" (phone-presence, wifi-usage, etc.) — same pattern as `dendrite_growth.md`. New question = add new branch.
5. **Verification loop**: when the model emits a prediction, a downstream check is needed. It can be:
   - A second model with different inductive bias (shadowing).
   - A periodic re-fed probe.
   - A user-supervised correction event ("that answer was wrong, here's what happened").
   - Self-consistency over time (state moves systematically).

## Hypotheses to test

- **R1**: A byte-level recurrent model ingests multiple parallel device streams and learns co-occurrence patterns.
- **R2**: Trained on synthetic streams, the model adapts to real device data with branch addition only — no full retraining.
- **R3**: End-to-end latency fits a 10–50 ms decision budget on CPU for small RWKV-nano + branches.
- **R4**: Continual branch-addition does not catastrophically forget past knowledge.
- **R5**: A prediction-self-check mechanism is feasible without ground-truth labels (a research question, may fail).

## What's pending before any test runs

- A synthetic device-stream generator that emits plausible wifi-association / bluetooth-advertisement / pings / packet-burst patterns. This is the largest unknown.
- A verification protocol — the answer to "who's around" can be wrong silently. Without a check mechanism, the experiment is unmeasurable.
- Capture pipeline when going to real device data.

## Status

- Theory established; see [`dendrite_growth.md`](dendrite_growth.md) for the trunk-and-branch pattern this design assumes.
- Implementation: not started.
- Real hardware capture: not started.
- Verification mechanism: not designed.
