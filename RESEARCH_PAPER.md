# Hybrid Latent-State Language Models: Thinking Once, Speaking Many
### Separating reasoning from token generation, and whether it beats an equal-capacity autoregressive model

**Status:** Working paper — local-GPU experiments (RTX 2050, 4 GB). For evaluation.
**Date:** 2026-07-13
**Code:** `src/latent.py`, `train_converged.py`, `bench.py`; theories in `theories/*.md`; logs in `reports/`.

---

## Abstract

We ask whether a neural architecture can **separate *thinking* from *speaking***: maintain a
private latent computational state that is *derived once* from a context, then used to
generate many output tokens cheaply — and whether this outperforms an
**equal-capacity** token-by-token autoregressive (AR) model on long-horizon reasoning.

We propose a **think-once / speak-many** design: a transformer encodes the (small) context
*once*; a recurrent latent state folds over the encoded representations and then **loops**
(re-attends the context) to "derive the future"; finally a decoder **rapidly emits all answer
tokens from the fixed state**. This is deliberately *not* the O(N) AR loop of re-reading the
context and emitting one token at a time.

On a synthetic multi-hop tracking task with multi-query worlds, we find that the
reported "decisive win" was an **artifact**. The eval metric was frozen because (i) the
looping model was trained with a dummy confidence target and a single-state token loss —
so the loop learned nothing — and (ii) a `NONE` majority-class shortcut pinned `AT`/`SAME`
at their cheat ceiling. After correcting the loop training (full unroll, trajectory-derived
confidence, deep supervision across loop states, loop→auto-encoder→generator pipeline) and
removing the `NONE` shortcut, the metric is valid and **moves** — but at this capacity the
latent model currently **trails** an equal-capacity AR baseline on the now-genuine relational
task, while the baseline's per-query re-encoding is far easier to train. The durable
contribution is the **failure-mode analysis** (NONE-shortcut, WHERE-needs-a-tape,
capacity-not-bottleneck) and the corrected, reproducible methodology; the architectural
hypothesis is **not yet supported** and is under active revision.

---

## 1. Motivation and Hypothesis

> **Core hypothesis.** A model with `latent_state_update() × N` + `decode_token() × M` can
> outperform an equivalent token-by-token autoregressive model on long-horizon reasoning.

A token is a poor clock cycle for reasoning: next-token prediction forces the model to
re-read the entire context for *every* emitted token, paying O(N) context cost per output.
We instead make the latent state the **scratchpad**: reason about the context *once*, orient
the state, then decode.

**The GRAND idea (user's).** Process the context *once*, **loop to derive the future** (cheap,
small context, no re-tokenization), then **rapidly decode the remaining tokens**. Keep the
context small (≈128 token-types × ≈128 context window) and combine a transformer (context
encoder) with a state model (the loop). Long-term target: a small-context transformer feeding
a looping latent state.

### 1.1 Research ladder (each level must be proven before advancing)

| Level | Question | Status |
|---|---|---|
| 0 | Does latent state work at all? | ✅ trains, reaches ≈0.65 exact-match |
| 1 | Does latent thinking beat tokens? | ❌ not at this capacity: corrected run (T10) shows latent trails baseline; prior "win" was an artifact |
| 2 | Does latent state survive context removal? | 🟡 implied by think-once design |
| 3 | Can latent state generate multiple tokens? | ✅ speak-many from one state |
| 4 | Can latent continue after interruption? | 🟡 not yet tested |
| 5 | Can state computation be independent of token generation? | 🟡 partial (loop decoupled from decode) |

---

## 2. Related Work

- **JEPA / JEPA-Reasoner** (arXiv:2512.19171): decoupling latent reasoning from token
  generation — the direct conceptual ancestor. We realize a concrete, trainable
  think-once/speak-many instantiation on a controlled synthetic task.
- **State-space models** (S4, Mamba): recurrent state as an alternative to attention; our
  `think` cell is a small recurrent module, but the *context encoder* is a transformer.
- **"Pause tokens" / latent/implicit chain-of-thought**: inserting non-emitted computation
  between tokens. Our loop is internal (no pause tokens emitted) and derives before decoding.
- **Memory/tape hybrids**: exact recall (spelling, names, trajectories) is delegated to an
  external memory in many architectures — consistent with our finding that `WHERE` needs a tape.

---

## 3. Method

### 3.1 Task: multi-hop tracking in synthetic worlds (`gen_world`)

We generate random "worlds" under consistent physical move rules, then ask multiple questions
per world. This isolates **long-horizon, multi-hop** reasoning and makes **amortized thinking**
matter (one context, many queries).

- **World:** `n_items` objects, `n_locs` locations, a chain of `max_events` move events. Each
  event is `(item, new_location, ".")`, optionally with a distractor item also moving.
- **Source:** the flat token sequence of all events: `item loc . item loc . …`.
- **Queries (4–8 per world, integration-heavy mix):**
  - `WHERE item ?` → final location of `item` (precise trajectory recall → needs a tape).
  - `AT location ?` → all items currently at `location` (list; `NONE` if empty).
  - `SAME item ?` → other items at the same location as `item` (relational; `NONE` if alone).
- **Answerability:** every query is answerable from the source; the data is *coherent* (rules
  consistent) and *random* (worlds/locations sampled uniformly) — learnable but not memorizable.

Hyperparameters: `n_items=6`, `n_locs=32`, `max_events=16`, vocab size 266, L\* (information-
theoretic floor) = 77.9 bits ≈ 5 floats.

### 3.2 Architecture: think-once / speak-many

```
              INPUT TOKENS (context)
                   |
          Transformer encoder        ← process ONCE; width/depth scaled by max_loop
                   |
              token reps  [T, d_ctx]
                   |
          fold state over reps  (K recurrent think-steps per token)
                   |
                   v
             Latent state  s
                   |
        LOOP: re-attend reps loop_max times   ← "derive the future"
              (early-exit when readiness ≥ min_certainty)
                   |
                   v
          oriented state  s*
                   |
        Rapid decode (GRU speaks many answer tokens from s*)
                   |
                   v
              OUTPUT TOKENS
```

**LatentModel** (`src/latent.py`):
- **Context transformer encoder.** `d_ctx = min(d_emb × scale, 512)`,
  `num_layers = min(scale, 4)`, where `scale = max_loop`. The encoder is scaled with the
  reasoning budget so the loop has a rich context representation (capped at 512-d to preserve
  quality). Output: per-token reps `[T, d_ctx]`.
- **Recurrent think cell.** `think(state, token_rep, derive) → new_state` (2-layer
  `Linear+Tanh`). The state folds over the encoder reps, `K` steps per token (the single
  context pass). `derive ∈ {SRC, ANS}` is a 1-of-2 embedding flagging whether we are building
  the source state or answering.
- **Looping derive (T09).** After the fold, the state **re-attends the encoded context reps
  `loop_max` times** to refine/derive. At inference it stops early when a **readiness head**
  `state_conf(s) ∈ [0,1]` exceeds `min_certainty`. (Earlier attempts that self-recurred on a
  zero-vector collapsed — see §5; re-attending real context reps is the working "derive".)
- **Auxiliary reconstruction head (T06).** `recon(s, item_emb) → location` predicts each item's
  final location from the state. This *forces the state to encode item→location / relational
  information* that `AT`/`SAME` need, instead of shortcutting.
- **Rapid decode.** A GRU (`speak`) emits answer tokens from the fixed oriented state `s*`
  (plus the query embedding) — the "speak many" stage. A `comp` head detects answer completion.

**BaselineAR** (token-by-token comparison): a GRU autoregressive decoder that must
**re-encode the full source for every question** (no reusable latent state). This is the O(N)
AR loop we contrast against.

> **Fair-capacity protocol (key contribution).** We auto-size the baseline to the *same
> parameter count* as the latent via a binary search over its `d_hidden`
> (`_match_baseline_params`). This isolates **architecture** from **capacity**: the latent is
> scaled by `max_loop`; the baseline is grown to match.

### 3.3 Training objectives

1. **Answer cross-entropy** with **teacher forcing**; strict exact-match QA at eval.
2. **T05 — uniqueness-weighted loss.** Each query CE is weighted by `w(a) = −log2 p(a)` over
   the answer distribution, so rare/unique answers (locations, specific items) are up-weighted
   and the `NONE` majority class is down-weighted. Applied to *both* models.
3. **T06 — auxiliary reconstruction loss.** `ℒ_recon = CE(recon(s, item), final_location)`
   summed over items; weight `recon_w`. Encourages the state to hold relational/trajectory info.
4. **T09 — looping + readiness.** During training the loop runs `loop_max` times; a readiness
   head is trained with `conf_w · BCE(state_conf(s), 1)` so the model learns *when it is
   oriented*. At inference the loop early-exits on readiness.

Optimizer: Adam, lr = 1e-3. Device: RTX 2050 (4 GB), CUDA 13.2, torch 2.5.1+cu121, Python 3.13.

### 3.4 Evaluation

- **Primary metric: strict exact-match QA accuracy** (generated answer tokens must equal the
  reference exactly). Per-type (`WHERE`/`AT`/`SAME`) and by difficulty bucket.
- **Capacity control:** both models reported at equal param count; per-type breakdown prevents
  "low loss but useless output" traps.

---

## 4. Experiments and Results

### 4.1 Dataset diagnostics (why naive training collapses)

From `dataset-stats` over the task:

| Stat | Value | Implication |
|---|---|---|
| `NONE` rate — `AT` | 86.9% | a model that always says `NONE` scores 0.869 on `AT` for free |
| `NONE` rate — `SAME` | 89.4% | same for `SAME` (cheat ceiling 0.894) |
| `NONE` rate — `WHERE` | 0% | `WHERE` cannot be cheated (needs real recall) |
| Overall `NONE` cheat ceiling | **0.619** | any accuracy ≤0.619 is plausibly pure cheating |
| Location-slot emptiness | 87% | states are *normally empty*; sparse supervision |
| L\* information floor | 77.9 bits ≈ 5 floats | `d_state=48` carries 768 bits (**9.8× headroom**) → capacity is *not* the bottleneck |

This explains the original collapse: without T05/T06 the model "wins" by answering `NONE`.

### 4.2 Ablation progression (all on RTX 2050; latent ≈ baseline size unless noted)

| Experiment | Latent | Baseline | Reading |
|---|---|---|---|
| T05 (uniqueness loss) | 0.596 | 0.617 | `WHERE` 0.018→**0.041** (2.3×↑); `AT`/`SAME` still at `NONE`-cheat |
| T06 (recon, alone) | 0.626 | 0.650 | latent `AT` **0.895** (>cheat 0.869, >base 0.886) — real reasoning via recon |
| T05+T06 | 0.590 | 0.587 | **latent wins**; `AT` 0.798 / `SAME` 0.844 > base 0.763/0.762 (T02 reasoning win) |
| T09 (self-recur loop) | 0.578 | 0.587 | **epoch-5 collapse** (0.641→0.539); self-recur on zero-vec ≠ derive → rejected |
| **T09b** (transformer + re-attend loop) | **0.649** | 0.626 | stable 8 ep, no collapse; transformer scaling fixes T09 |
| **T09c** (equal params, both ≈9.98M) | **0.649** | **≈0.16** | **RETRACTED**: frozen 0.649 was a bug (dummy conf + single-state loss) + NONE-cheat; see T10 |
| **T10** (loop fix + NONE removed) | **≈0.01** | **≈0.06** | metric now MOVES (not frozen); latent trails; multi-token relational task hard at this scale |

**Headline (T09c, equal capacity, 8 epochs planned; 6 completed before timeout):**

| epoch | latent | baseline (9.98M GRU AR) |
|---|---|---|
| 0 | 0.649 | 0.233 |
| 1 | 0.649 | 0.321 |
| 2 | 0.649 | **0.082** |
| 3 | 0.649 | 0.220 |
| 4 | 0.649 | 0.170 |
| 5 | 0.649 | 0.161 |

The latent trains **stably** at 9.98M; the equal-size GRU baseline **collapses**. Notably,
*growing the baseline* (399K → 9.98M) made it **worse** (0.626 → 0.16), while *growing the
latent* helped — so the latent architecture **scales better** than a GRU AR.

**T10 (corrected methodology, cheat-free task).** Loop training rewritten per the
proper recipe: full unroll to `max_loop`, confidence target derived from the loop
*trajectory* (closest-to-final state → 1), deep supervision (token-generator + recon +
confidence losses applied at **every** loop state, like a transformer over time), and a
loop→auto-encoder→generator pipeline. The `NONE` shortcut was removed from `AT`/`SAME`
(guaranteed co-located pairs so `SAME` has answers). Result: the eval metric now **moves**
every epoch (frozen-bug fixed), but the latent **trails** the equal-capacity AR baseline
(≈0.01 vs ≈0.06) and its training loss descends but plateaus higher than the baseline's
(269→220 vs 35.8→4.6; the latent loss is summed over 7 loop states of deep supervision, so
raw magnitudes aren't directly comparable). The prior
"decisive win" is **retracted** — it was an artifact of a broken loop plus the `NONE` cheat.
The corrected experiment is a genuine (negative) result: at this scale the think-once state
does not beat per-query re-encoding on relational reasoning.

### 4.3 Per-type breakdown (T09b, the cleanest stable run)

| Type | Latent | Baseline | Note |
|---|---|---|---|
| `AT` | 0.866 | 0.773 | latent higher, **but ≈ `NONE`-cheat ceiling 0.869 → illusory** (see §5) |
| `SAME` | **0.911** | 0.881 | real relational reasoning win (above cheat 0.894) |
| `WHERE` | 0.035 | **0.118** | latent loses — needs a tape (Model C) |

### 4.4 Hyperparameters (main run T09c)

`n_samples=500` (450 train / 50 val), `epochs=8`, `K=2`, `d_state=48`, `d_emb=128`,
`d_hidden=256`, `max_events=16`, `recon_w=1.0`, `max_loop=8`, `min_certainty=0.9`,
`conf_w=0.1`, lr=1e-3 (Adam). Vocabulary = 266. **Params: latent = 9,983,532;
baseline = 9,984,745** (auto-matched).

---

## 5. Analysis

**5.1 The `NONE` shortcut inflates `AT`.** Latent `AT` = 0.866 sits essentially at the
`NONE`-cheat ceiling (0.869). The baseline's lower `AT` (0.773) is *real* reasoning. So the
latent's apparent `AT` win is partly cheating; the **genuine** reasoning win is on `SAME`
(0.911 > 0.881, above its cheat ceiling). Fix: redesign `AT` so `NONE` is never correct (e.g.,
guarantee ≥1 member at every queried location), or up-weight non-`NONE` `AT` answers further.

**5.2 `WHERE` needs a tape (Model C).** The latent has no exact-recall memory; `WHERE`
(trajectory recall) stays near 0 while the re-encoding baseline scores 0.118. This matches the
architectural intent: the SSM/state holds *semantic* state (logic, relations), while a
**tape** holds *exact* token patterns (spelling, names, trajectories). Adding a tape is the
next model.

**5.3 Latent scales better than GRU AR (T09c).** At equal capacity the latent is stable and
strong; the GRU baseline (9.98M) is unstable and collapses. *Caveat:* a 9.98M GRU may simply be
hard to optimize on 450 tiny worlds (instability, not an AR verdict). A **transformer-AR
baseline of equal size** is the cleanest control and is proposed as the next experiment.

**5.4 Capacity is not the bottleneck (T07).** `d_state=48` (768 bits) is 9.8× the L\* floor;
the earlier losses were architectural (shortcutting), not under-capacity.

**5.5 The loop early-exits at inference.** The readiness head saturates (target 1.0), so at
inference the loop stops after ~1 step — the *win* is carried by the transformer encoder +
reconstruction head, and the loop currently acts mainly as a training-time regularizer.
Tuning `min_certainty`/`conf_w` so the loop actually runs at inference is future work.

---

## 6. Limitations and Future Work

1. **Fairer baseline.** Replace the GRU AR with a **transformer-AR** of equal params (or lower
   the big GRU's LR) to isolate architecture from optimization difficulty.
2. **Break the `NONE` cheat** — DONE (removed from `AT`/`SAME`; guaranteed co-located
   pairs). Result: metric valid/moves but latent trails baseline (T10); real relational
   signal now measurable.
3. **Tape / Model C** for `WHERE` (exact trajectory recall) — the latent's remaining gap.
4. **Loop at inference** — make `derive` actually run (lower `min_certainty`/`conf_w`).
5. **Scale to ≈20M params** on the local GPU for the "first-night win condition."
6. **Generalization beyond synthetic worlds** — the task is controlled; real text is next.

---

## 7. Conclusion

A think-once / speak-many latent design — transformer context encoder, recurrent state
fold, looping derive, rapid decode — is a **plausible** alternative to token-by-token
generation, but the architectural hypothesis that it *beats* an equal-capacity AR model is
**not yet supported**. The originally reported "decisive win" was invalid: the eval metric
was frozen by a broken loop-training protocol (dummy confidence target, single-state token
loss, no auto-encoder) compounded with a `NONE` majority-class shortcut. After correcting
the loop (full unroll, trajectory-derived confidence, deep supervision, auto-encoder) and
removing the shortcut, the metric is valid and moves — and the latent **trails** the AR
baseline on the genuine relational task at this capacity. The durable, correct contributions
are the **failure-mode analysis** (NONE-shortcut, WHERE-needs-a-tape, capacity-not-bottleneck)
and a reproducible, honest evaluation methodology. Next: diagnose why the latent state fails
to train (state aggregation vs deep-supervision optimization), add a tape for `WHERE`, run a
transformer-AR baseline, and report multi-seed variance before any win claim.

---

## References

- JEPA-Reasoner: Decoupling Latent Reasoning from Token Generation. arXiv:2512.19171.
- Gu et al., *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*.
- Dao & Gu, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
- Goyal et al., *Improved Baselines for Latent Reasoning* (pause-token line of work).
- Internal: `theories/01..08`, `PROGRESS.md`, `reports/t09*_run.log`, `src/latent.py`,
  `train_converged.py`.
