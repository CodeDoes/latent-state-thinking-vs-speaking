# Smoke Test Methodology — Inferred Interpretation

> **Source**: `theories/smoke_test_methodology.md` (verbatim)  
> **Date**: 2025-07-17

---

## The Methodology (Your Words → My Rules)

### 1. **Observable feedback during training**
*"i liked the previous experiment but i feel it gave no feed back. i couldn't see the result over time."*

- Every training run must log per-step loss (or every N steps), not just end-of-run.
- Per-epoch metrics to stdout (and ideally a file).
- If you can't see the curve, you can't diagnose failure.

### 2. **Goal of small scale experiments**
*"can we even learn things" / "can you do anything useful at this small scale?"*

- Small scale is a **probe**, not a deployment.
- Question being answered: *"does the architecture/code/infrastructure actually work and learn anything?"*
- Not: *"does it match SOTA?"*

### 3. **Use cheap training time deliberately**
*"can't we rather consider things we might use the training time for. is there not something we can deliberately overfit it on."*

- Don't waste compute on long runs while infrastructure is broken.
- **Deliberate overfitting** is a valid first goal: prove the model has capacity, even if regularization is wrong.
- Pick tasks where overfitting is *easy* (personal-data patterns, small synthetic rules) — proves basic mechanism.

### 4. **The 1-minute rule**
*"your synth data might just be bad. it needs to learn the underlying shape or pattern. not the question -> answer. your data has no pattern. thus its not very useful evaluation. from now on do the same but aim for 1 minute training time each"*

- **Time budget per experiment: ≤ 1 minute of wall-clock training time.**
- Synth data must have a **learnable pattern** (shape/structure), not just questions.
- Question → answer pairs without underlying pattern = memorization, not learning.
- Most important: if you can't design data with the right shape, the experiment is meaningless.

### 5. **Layered minimal-proof strategy** (verified by extension into the byte-level architecture)

"a small pure BLT model -> working
a small BLT encoder and decoder with a small RWKV -> working
a RWKV based encoder and decoder with a normal transformer -> working
a pure RNN based BLT model -> working

in that order. would be better"

- Always go from **simplest → more complex** in matched steps.
- Each stage must work end-to-end before adding complexity.
- Each stage respects the 1-minute budget.

---

## The Experiment Pattern (My Template)

```python
def smoke_experiment():
    # 1. Synth data with a clear, learnable shape
    generator = build_generator(seed=42)  # rule: fixed transformation
    
    # 2. Tiny model (≤ 300K params, CPU-friendly)
    model = build_model(...)  # RWKV-nano or similar
    
    # 3. 1-minute training budget
    train(model, generator, max_seconds=60, log_every=10)
    
    # 4. Observe: loss curve, accuracy curve, sample generations
    # 5. Verdict: model overfit? Underfit? Mode-collapsed? Diverged?
    
    return result, sample_outputs
```

---

## What "Learnable Pattern" Means

A synth data generator has a learnable pattern if:
1. There is a **fixed transformation rule** (e.g., "sum of digits ≥ 50").
2. Examples can have **variable inputs** that probe the rule.
3. The model can memorize the rule by gradient descent on enough examples.
4. A "memorized" model can *generalize* to held-out examples using the same rule.

Tasks that meet this:
- `sum_threshold`: digits → ✓/✗ based on sum
- `vowel_majority`: letters → ✓/✗ based on vowels
- `endpoint_match`: letters → ✓/✗ if first==last
- `count_trigger`: letters → ✓/✗ if count(x)>3

Tasks that DON'T:
- Random number → random answer (no rule)
- Each example gives a unique, non-recurring answer (memorize-only)

---

## Common Failure Modes (You Name Them)

| Failure | What you said | Diagnostic |
|---|---|---|
| No signal direction | *"i feel it gave no feedback. i couldn't see the result over time"* | log loss every step |
| Data has no pattern | *"your synth data might just be bad. it has no pattern"* | test memorization on one example first |
| Loss is 0 from step 1 | (Jul 17 session) | check eval — vacuous true if mask is all zeros |
| Mode collapse to `"eee..."`* | byte-level pitfalls in `shared_state_unrolled_shared_010` | entropy mask, separation of weights |
| Decoder stall | byte-loop `shared_state_unrolled_010` | adaptive exit / surprise router |

---

## Latent Assumptions (Infer Notes)

- "1 minute" assumes **CPU**, single-thread. GPU might be merely "fast enough" not "instant."
- "Observable" implies **you, watching**. Per-step stdout logs or a tensorboard.
- "Deliberate overfit" implies you accept test-acc = 1.0 as a positive result for capacity proof. Generalization is a separate stage.
- The "shape/pattern" rule is doing real work — without it, the experiment is uninformative. Most AI-generated synthetic data is **anti-pattern**: it's designed to be procedurally diverse but lacks a *stable rule*.
- The 1-minute rule may conflict with proof depth. If a property needs more, document why and increase budget explicitly.

---

## Open Follow-Ups

1. `smoke_001`: Implement a 1-minute smoke test that *always* runs (every model under test gets one)
2. `smoke_002`: Build a `SynthEvaluator` that flags "no learnable pattern" generators in code
3. `smoke_003`: Add `log_every_step` option to training scripts so curves are visible
4. `smoke_004`: Build a per-experiment dashboard in `experiments/<id>/samples.json` with loss/accuracy curves
