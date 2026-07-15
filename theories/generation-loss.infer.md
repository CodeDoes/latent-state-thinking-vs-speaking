# generation-loss.infer.md — interpretation of theories/generation-loss.md

This file fills in what `generation-loss.md` leaves implicit.

---

## The proposal, restated as a mechanism

After the model has read its context and produced a sequence of generated
tokens (one per step), record the **logits at each generated step**. Apply
cross-entropy loss, position-by-position, against the **correct** (target)
token. But apply gradient only at positions where the **generated token ≠
the correct token**.

The proposal is a *training-signal change*. It is **not** an architecture
change. It is **not** a decoding change at inference. It only modifies what
gets supervised during training.

Concretely, for an answer span of length `L`:

```python
# at each answer position i in 0..L-1
logits_i  = model.forward(context + generated[:i])[-1]   # next-token logits
target_i  = ground_truth_answer[i]
taken_i   = generated[i]                                 # argmax(logits_i) under whatever decoding
mask_i    = (taken_i != target_i)                        # only punish wrong ones
loss_i    = F.cross_entropy(logits_i, target_i) * mask_i
loss      = loss_i.sum() / max(mask_i.sum(), 1)
```

Backprop flows backward through every step that produced a wrong token.

## What it is (single sentence)

It is **self-generated-token-by-token cross-entropy, masked to wrong
tokens only**, fed back as the training signal.

## What it looks like vs standard CE / teacher-forcing

Three supervision regimes on the same model + same data:

| Regime | Source of supervision | What gets a gradient | When it differs |
|---|---|---|---|
| **Standard TE (teacher-forcing)** | Teacher tokens (always correct) | Every answer position | Baseline. Doesn't see the model's own behaviour. |
| **Scheduled sampling** | Mix of teacher tokens and the model's own samples | Every answer position | Reduces exposure bias; doesn't mask the loss. |
| **Generation-loss (proposal)** | The model's own greedy generation | Answer positions where the model was wrong | Both: uses **own generation as input** AND **masks correct positions**. |

The proposal is *closer to*: "**do policy gradient / REINFORCE with the
cross-entropy of the correct token at positions where my behaviour was
wrong**." It is a *behaviour-aware* masked CE.

## Why it's plausible that this helps (the case for)

The user's intuition, made explicit:

1. **Exposure bias is real.** A model trained teacher-forced has never
   seen its own distribution of outputs. At inference it sees them and
   breaks. Sampling its own tokens during training makes the
   train/inference distribution match.
2. **Correct tokens carry no information.** If the model already produces
   the right token at position `i`, the cross-entropy there is already
   minimal in expectation. Punishing it more is a no-op. Masking it
   frees the gradient budget to spend effort *only on positions where
   error exists*.
3. **Wrong tokens are the ones the model needs to fix.** A model that
   gets `correct = "24"` but generates `"25"` would, under standard CE,
   get a non-zero loss on *all* five answer positions. Under this
   proposal, it gets a non-zero loss on **only the `5` position** plus
   earlier wrong positions. The gradient is concentrated where the
   error is.
4. **Carries over from autoregressive decoding feedback.** Once you
   accept that the model's behaviour matters, conditioning the loss on
   the behaviour produces a tighter gradient signal than conditioning it
   on the teacher.

## Why it might *not* help / places it can fail

The proposal is a single-variable change. It can be tested honestly only
if all of these are accounted for first.

1. **Compounding errors.** If the model is wrong at step 1 of the
   answer, the input to step 2 is now off-piste. Steps 2..L see a
   distribution that diverges from training. Standard CE has the same
   *exposure* problem; this proposal doesn't solve it. Worse: with
   masking on correct tokens only, an early mistake pollutes the input
   for all later steps, and the gradient on the later steps is
   conditioned on that polluted input.
2. **Gradient-vanishing on greedy-wrong positions.** Greedy decoding gets
   a token by argmax. The gradient on `cross_entropy(logits, target)`
   at the wrong position is `softmax(logits) - one_hot(target)`. If the
   logits already pin the wrong token very confidently, the gradient is
   small at saturation. A *learning* model rarely saturates, but a
   *pre-trained* or partially-trained one can.
3. **Distribution shift between training-time sampling and test-time
   decoding.** If during training we sample stochastically (temperature,
   top-k) and during test we decode greedily, we optimise the wrong
   mode. To be a clean ablation, training and inference decoding must
   match.
4. **Constant baseline lie.** Generation-loss is *implicitly* compared
   to a teacher-forced baseline that does **not** see the model's own
   behaviour. If the proposal wins, we don't know whether it won
   because (a) the masking helped, (b) the self-conditioning helped,
   or (c) both. The single-variable frame requires *one* change at a
   time — this proposal twiddles **two** knobs simultaneously.
5. **Output length dependency.** The proposal implicitly assumes an
   answer-length CE: if the model doesn't know `L` in advance and
   generates until an EOS, the "wrong token" mask depends on whether
   we're aligned with the target length. Alignment needs to be handled
   explicitly (or experimental setup must fix the answer length).

## What this proposal is **not** — important to disambiguate

- It is **not** REINFORCE / policy gradient. There's no reward, no
  baseline, no log-prob-of-action weighting. It's plain CE, just at
  different positions.
- It is **not** DAgger (Dataset Aggregation). DAgger queries an expert
  on the model's own states and adds those to the training set. This
  proposal *uses the model's own outputs as both input AND the target
  selector*, but the targets are still ground-truth — no expert roll
  is required.
- It is **not** an inference-time change. Greedy or sampled decoding
  doesn't move.
- It is **not** changing the model. No new heads, no new losses, no
  architecture.

It is a *single change to the supervision signal* that is, by design,
isolated and testable.

## How to test it under theories/ultimate.md

Per `theories/ultimate.md`, the test must:

1. Be **single-variable** between runs.
2. Hold **params, data, training budget** constant.
3. Pick **one measurable property** as the dependent variable.
4. Be **reproducible** from the run's git hash.

The **cleanest experiment design** for this idea:

> Compare three supervision regimes on the same model + data + budget:
>
> 1. **T** (baseline): teacher-forcing CE on every answer position.
> 2. **M** (masked only): own-generation CE, gradient only on wrong
>    positions (the proposal).
> 3. **U** (unmasked): own-generation CE on every position (a control).
>    U separates the two effects — own-generation input vs masking-out
>    correct positions. If M > T, fine; if M ≈ U, then the gain comes
>    from self-conditioning, not the mask; if M < U, the mask is doing
>    the work.

Three runs, one variable each:

- **T → M**: changes both self-conditioning AND masking. Two variables.
- **T → U**: changes self-conditioning only. One variable.
- **U → M**: changes masking only. One variable.

The pair (U → M) isolates the user's specific contribution. If that pair
moves the needle, the user's proposal has a clean single-variable proof.

### Recommended minimum scale for the first run

Per `ultimate.md`, smallest viable:
- **Task**: single-digit answer (`L=1`). Removes length-alignment
  confounders (point 5 above).
- **Data**: logic-niiah with 1 needle, 1 transform, no noise. The model
  reads a context containing `A = X; add Y to A; What is A?` and must
  produce `Z = (X+Y) mod 10` or similar. Ground truth is exact.
- **Model**: a small GRU (the RWKV-nano line is stalled on a different
  bug). GRU trains fast on CPU and proves the principle without
  entangling it with the RWKV collapse.
- **Three runs**: T, M, U. Matched seeds where possible. Track:
  - Final exact-match accuracy on a held-out generator seed.
  - Loss curve + gradient norm per step.
  - Whether the model converges, plateaus, or diverges.

This is exactly the `exp003` proposal from `theories/status.md`,
restated under the rigorous single-variable frame the methodology
demands.

## What the proposal is unlikely to teach us (honesty check)

Even if it works perfectly, it does **not** prove:

- That thinking/speaking separation is necessary or sufficient. (That's
  the think-once line, governed by `rwkv.md` and `exp001`.)
- That small models beat larger ones for this capability. (That's a
  scale ablation, separate experiment.)
- That the masking is the *specific* mechanism in deep models at
  scale. (This proposal is a toy-scale proof of concept.)

What it *would* prove, if U → M shows a clean win:

> *The model's own generation, used as input during training, plus a
> gradient mask restricted to positions where the model was wrong,
> improves exact-match accuracy on a narrow answer task at matched
> compute.*

That is a publishable-shape single statement. It is also the kind of
narrow claim that survives being challenged.

## Implementation note (sketch only — not in `src/` until the experiment
is approved)

- The model must be able to **take its own previous outputs as input
  during training**. For a recurrent model this is one forward pass per
  answer step; for a transformer it's a KV-cached forward.
- The `taken_i != target_i` mask must be computed **after** the full
  generation, then backpropped. PyTorch handles this with
  `retain_graph=False, create_graph=False` between positions; for a
  model that doesn't backprop through itself, this is trivial.
- There is **no need** to backprop through the sampling decision. The
  generation is done *without* grad (or with grad-detached), then
  CE-with-mask is applied. This is what makes the implementation
  cheap.

## Status
Proposed 2026-07-15. Not yet implemented. Awaiting approval to run
under the test design above, ideally as `experiments/exp003/`.
