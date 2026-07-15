# rwkv.infer.md — interpretation of theories/rwkv.md

This file fills in what the `.md` leaves implicit.

---

## The unstated hypothesis

The `.md` is a directive ("make a small rwkv model and find out"), not a
hypothesis. The actual claim being tested is:

> **A nano-scale RWKV can learn a program's input/output behavior by reading
> many context tokens and producing few verifiable answer tokens.**

The `.md` now refines the core learning format:

> read-many-context-tokens, answer with few but verifiable tokens

This reframes everything. The task is **compression + query**: the recurrent
state must compress a long context into a representation that can decode into
a short, exact answer. The output is tiny (a few tokens) but every token must
be correct — no approximate answers.

This maps cleanly to program simulation:
- **Context tokens**: serialised program + input (many tokens)
- **Answer tokens**: program output (few tokens)
- **Verification**: exact string match against the real program's output

For the thinking/speaking question:
- The long context is **thinking fuel** — the model reads, and its recurrent
  state builds an internal representation.
- The few answer tokens are **speaking** — the model externalises the result.
- The gap between context and answer is where the "thinking" happens, purely
  in latent state, with no intermediate output.

## What "nano" means (unstated)

"RWKV nano" implies the smallest viable RWKV — likely well under 1M parameters.

- **~100K–1M parameters** — just an embedding layer, a couple of RWKV blocks,
  and an output head. Think of it as the RWKV equivalent of a TinyStories-
  scale model.
- The architecture is aggressively narrow (embedding dimension 64–128) and
  shallow (2–4 layers).
- The output head depends on the target program's output format. If the
  program produces structured output (JSON, a single token, a number), the
  head can be task-specific rather than a full LM head.

## Training speed (unstated)

RWKV trains faster than a transformer of equivalent size for two reasons:

1. **Linear attention** — WKV is O(n d²) instead of O(n² d). Longer sequences
   widen the gap.
2. **No KV cache** — the recurrent state is a fixed-size vector. No memory
   pressure.

The user's ".md" doesn't say this, but the "how quickly it can train" question
presupposes this advantage. The real comparison is *throughput-to-accuracy*,
not just final accuracy.

## The program vs logic needle-in-a-haystack

The `.md` gives two related but distinct training ideas:

1. **Simulate a real program** — learn a fixed input/output mapping from
   execution traces.
2. **Logic needle-in-a-haystack** ("logic niiah") — a templated reasoning
   format where the model must track and transform values through noise.

### Logic niiah format (new)

The user described the template precisely:

> instruction + noise + needle + noise + needle-action-transformation +
> noise + repeat-X-times + ask-questions-about-needle-transformations

Concretely:
- **Instruction**: "Track the values. At the end, answer the questions."
- **Noise**: irrelevant sentences ("The sky is blue. Cats are mammals.")
- **Needle**: a variable assignment ("Let A = 5")
- **Needle-action-transformation**: an operation on that variable
  ("Add 3 to A" → A becomes 8)
- **Repeat X times**: multiple needles interleaved, each with multiple
  transformations
- **Ask questions**: "What is the final value of A?"

This is essentially a **register machine** task — the model must maintain a
set of variable bindings in its recurrent state, update them through noise,
and recall them at query time.

### Why this is a better logic test than program simulation

| Aspect | Real program | Logic niiah |
|--------|-------------|-------------|
| State tracking | Implicit (the program's state) | Explicit (named variables) |
| Noise | None (all input is signal) | Deliberate, parametrisable |
| Difficulty control | Change the program | Change X, #variables, noise ratio |
| Interpretability | Hard (what does the state represent?) | Easy (does the state track A?) |
| Failure diagnosis | Model is wrong — why? | Model lost variable A at step N |

The logic niiah format is **more controllable** and **more diagnostic**. You
can precisely vary the number of needles, the number of transformations, and
the noise level. If the model fails, you know exactly what it failed on
(e.g., it can track 2 variables but not 3).

### Relationship to "read many, answer few"

The format is a perfect instantiation:
- **Many context tokens**: instruction + noise + needles + transformations + noise
- **Few answer tokens**: the variable values at the end
- **Verifiable**: exact match on each variable's value

### Generator design for logic niiah

The synthdata-generator for this format would be parametrised by:
- `num_needles` (how many variables to track, e.g., 1–5)
- `num_transforms` (how many operations per needle, e.g., 1–10)
- `noise_ratio` (how much noise between signals, e.g., 1–5 sentences)
- `value_range` (numeric range for variables, e.g., 0–100)
- `operation_types` (add, subtract, multiply, divide, set, etc.)

Each call generates a fresh random instance with a unique combination of
parameters. The generator runs a simple interpreter to compute the ground-
truth answers, then formats everything as text.

## Data strategy: generator + solver (key insight)

The `.md` now names two components:

**synthdata-generator** — a program (Python) that:
1. Takes a random seed / random input.
2. Runs the **real program** on that input.
3. Emits (input_representation, output_representation) as text.
4. Can be parametrised to control difficulty (input size, nesting depth, etc.).

**synthdata-solver** — the RWKV model that learns to map input → output.

This is the insight from earlier, now with a concrete mechanism:

> *"if we train quickly enough we can fix the synthdata generator until
> overfitting is no longer possible."*

Applied to program simulation:
1. Generate (input, output) pairs by running the real program.
2. Train RWKV to predict output from input.
3. If the model overfits (memorises specific inputs), increase the input
   space (more random seeds, larger array sizes, deeper nesting).
4. The generator is never exhausted — it produces new examples every step.
5. The only limit is whether RWKV's architecture can *represent* the
   program's logic, not whether it can memorise the training set.

**Procedurally generated content** means the inputs are generated by a
procedure (random sampling), not sampled from a fixed corpus. This is what
makes the data infinite.

## How to train: loss masking

The core format is now clear. Each example is:

```
<context tokens...> <answer tokens...>
```

The loss is **masked**: zero gradient for context tokens, cross-entropy only
on answer tokens. This is critical — the model is never asked to reconstruct
the context, only to use it to produce the correct answer.

In PyTorch-like pseudocode:
```python
logits = model(sequence)          # shape (B, T, vocab)
loss = cross_entropy(logits, targets, reduction='none')
loss = loss * answer_mask         # zero out context positions
loss = loss.sum() / mask.sum()
```

**Why this works**: the gradient flows backward through the answer tokens,
through the recurrent state at the answer position, and back through *all*
the context time-steps via the WKV recurrence. Every context token
influences the state that produces the answer, so the model learns to build
better state representations even though there's no direct loss on context.

**Intermediate traces** are optional but could help: if you include the
program's state after each step as context (e.g., "step 1: state = ..."),
the model gets a richer signal. But the `.md` doesn't specify this, and the
"few answer tokens" constraint suggests the user prefers a lean signal.

## Why this makes validation trivial

Few answer tokens + verifiable = exact match accuracy. No fuzzy matching, no
metric engineering. The accuracy is simply:

```python
correct = (predicted_tokens == target_tokens).all(dim=-1)
acc = correct.float().mean()
```

This is the "easy to validate" from the very first `.md` line, now
formalised.

## The implicit constraint

We are training **from scratch**, not fine-tuning a pretrained RWKV checkpoint.
For program simulation this is actually *essential* — we want the model to
learn *only* the target program, not general language.

Overfitting is **solved architecturally**: the generator produces fresh
(random) inputs every step. The model can never see the same example twice.
The validation set is generated with a fixed seed so it's stable for
comparison, but training data is infinite and ever-changing.

## Why RWKV specifically (unstated)

The user chose RWKV over other sub-quadratic architectures (Mamba, Mamba-2,
Based, GLA, H3, etc.). For program simulation, RWKV has a unique advantage:

- **Stateful recurrence** — a program is a state machine. RWKV's WKV vector
  is a fixed-size state that updates at every step. This maps directly to a
  program's program counter + memory.
- **State is interpretable** — you can inspect the WKV vector at each step
  and ask: does it encode the program's internal state?
- **Canalization** — RWKV-7/Eagle gating suppresses irrelevant state
  dimensions. For program simulation, this means the model can learn to
  zero out dimensions that track irrelevant parts of the program state.
- **CPU-friendly** — nano-scale RWKV trains in seconds on a laptop. This is
  critical for the "observable, resumable, constantly improvable" workflow.

## What we're not testing (important gap)

- **Generalisation beyond the program** — we're explicitly not testing this.
  The model should be useless on other programs. That's the point.
- **Comparison to a baseline** — no transformer control is mentioned. If we
  don't run one, we can only say "RWKV can learn this program", not "RWKV
  learns it better than X".
- **Scaling laws** — we're searching over one tiny size, not exploring how
  performance changes with parameters.
- **Length generalisation** — if the context length varies, can the model
  handle longer contexts than it saw during training? The `.md` doesn't say.
  RWKV's recurrence should handle this better than transformers (no position
  embedding cutoff), but it's unstated.
- **Intermediate supervision** — we're not training on execution traces, just
  end-to-end I/O. The model must discover the program's internal logic
  entirely through the gradient signal from the few answer tokens.

## Experiment design that's implied

1. **Pick a target program** — small enough to run fast in Python, complex
   enough that learning it is non-trivial. See table above.
2. **Write synthdata-generator** — a Python function that:
   - Generates a random valid input for the program.
   - Runs the program to get the output.
   - Serialises both as text.
   - Optionally includes intermediate states (execution traces).
   - Is parametrised (input size, nesting, seed range) so you can dial
     difficulty.
3. **Write synthdata-solver** — an RWKV nano model (train loop, checkpointing,
   logging). The model reads the serialised input and must predict the output
   tokens.
4. **Train loop must be resumable** — save optimizer state, step count, and
   generator RNG state in each checkpoint. On resume, restore everything and
   continue generating fresh data.
5. **Observability** — log loss, accuracy, tokens/sec, and sample generations
   to a file and stdout every N steps. Plot curves live if possible.
6. **Git-bound experiments** — each run records `git rev-parse HEAD` in its
   config. If you change code, the next run has a different hash. You can
   always trace which code produced which model.
7. **Easy to clear** — all model weights, logs, and config go under
   `experiments/<exp_id>/`. Delete that directory = full cleanup.
8. **Iterate on the generator** — if the model overfits or plateaus, expand
   the generator's input space / add more program behaviors. Don't touch the
   model architecture. Resume from checkpoint with the improved generator.
9. **Optional: compare to a tiny transformer** at the same parameter count.
   Without this, results are descriptive, not comparative.
