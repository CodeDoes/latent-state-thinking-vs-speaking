# Smoke Test Methodology

Small-proof experiments done in one minute of CPU time on synthetic data that has a learnable shape.

## Where it came from

After observing that several training runs produced no useful signal, the user diagnosed the cause: the synthetic data had no learnable pattern, and the experiments were costlier than necessary. From there the constraint emerged as a hard rule: synth data has to have *structure*, and the runtime has to be small enough to run many of them.

## The two rules

### 1. Synth data must have a learnable pattern

A generator has a learnable pattern when:
- There is a fixed transformation rule (some `f(x) -> y` that does not vary across examples).
- Examples vary in input but not in target rule.
- A model with sufficient capacity can *memorize* the rule by gradient descent on enough examples.
- A *memorized* model can *generalize* to unseen inputs that follow the same rule.

A trivial example: pick a random number between 1..10 and emit ("you're thinking of 7", "yes/no" based on whether the input equals 7). That has no learnable pattern — each example is one possible target out of ten. A model fitting this is memorizing, not learning. Same answer is right as often as not regardless of what it learns.

A non-trivial example: take a digit sequence, the target is "sum >= 50 ? yes : no". Now there is a *rule* the model can recover. Different inputs have different targets but the rule is shared.

### 2. The 60-second CPU cap

The honest goal of a smoke test is not to get the best possible accuracy, but to get *signal*. Is the model class capable of fitting the rule at all? If the answer is no in 60 seconds, a longer run will not fix it. If the answer is yes in 60 seconds — the model gets to ~80% on a rule task, say — then we now know where to spend more compute.

60-second cap = can run dozens per hour = cheap iteration.

## Experiments that survived this discipline

- `byte_loop_001`: encoder-decoder loop, 117K params. 500-step run, loss 5.78 → 2.00. Decoder trigger rate ~0.42.
- `adaptive_loop_001`: encoder-patcher-decoder, 228K params. 1000-step run, loss 5.74 → 0.47. Encoder loops 1→3.
- `dendrite_rwkv_001`: LoRA-on-RWKV for synthetic rule storage. Smoke-test ran but hit a collation bug before producing a result.

## A failure that this discipline would have prevented

- `shared_state_unrolled_010`: per-step Shared-encoder-decoder with separate weights, trained on TinyStories byte-level. Loss 11.4 → 0.19, then samples blank. Memory cost may be a more useful signal at this point than running it longer.

## Template for a smoke test

```python
# 1. Generator
generator = make_generator(seed=42)  # rule shape fixed; examples vary

# 2. Tiny model
model = build_model(...)

# 3. Train with 60-second cap
run_to_completion(model, generator, max_seconds=60, log_every=10)

# 4. Inspect
plot_loss_curve()
sample_predictions(model, generator)
```

Rule of thumb:
- If loss is monotonically decreasing at any point in the cap: model class is fit-able. Continue.
- If loss is flat from step 1: data has no rule, or model is broken.
- If loss explodes: bug in code (LR, loss shape, masking).
- If loss goes to 0 instantly: broken evaluation mask (vacuously-easy targets).

## Why this is a theory, not just a guideline

Because every accepted experiment in the project has to pass it, and several theories hinge on results that are *only* honest at this scale. A 1-hour run that gets 0% on a logic task is far less informative than a 1-minute run that gets 0% — the minute result rules out "needs more time," the hour result raises more questions than it answers.

If you are tempted to skip the smoke test, you are admitting the experiment design is uncertain. Run the smoke test.

## Status

- Hard discipline established.
- Template above is informal; a `theories_lint.py` and `smoke_runner.py` would encode it. Not yet built.
- Several experiments carry forward into longer runs after smoke success; tracked in `proofs.md`.
