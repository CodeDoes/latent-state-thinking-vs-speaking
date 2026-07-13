# Future Research: Past Hand-Holding, Toward True Learning and World Modeling

> **Thesis.** The structured world-table model (`src/world_state.py`) is *training wheels*:
> it proves a latent state *can* hold and serve structured facts, but only because we
> **hand it the structure** — inverted targets from `reverse_templates`, fixed pools, fixed
> head shapes, and clean single-answer queries. That hand-holding caps the model at the
> ontology we hard-code. True world modeling means the structure **emerges** from a
> self-supervised objective that gives the latent state a *causal, predictive role in the
> world*, read through an *open, generative* path. This document is the theory for how to
> remove the training wheels one rung at a time without the model collapsing.

---

## 1. What hand-holding bought us (and where it stops)

The current Model C succeeds *because* of three scaffolds we supplied:

1. **The target structure is given.** `reverse_templates` inverts the generator, so every
   training sample carries its exact `(entity→location/inventory, item→holder)` table. The
   model is never asked to *discover* what an entity or an item is — only to reproduce a
   table we already wrote.
2. **The vocabulary is closed and fixed.** `loc_head` is a 10-way classifier over `LOC_POOL`;
   `holder_head` a 10-way over `NAME_POOL`. The "world" is 10 names, 10 places, 10 things.
3. **The read is a lookup, not an utterance.** Answers are retrieved fields, not generated
   language. The model never has to *verbalize* a fact — only classify it.

These scaffolds are why we finally beat random (location 0.27, inventory 0.37, transfer 0.21):
the task was made *exactly representable* by the architecture. But they are also why the model
is not *learning* in any interesting sense — it is **memorizing a lookup table we pre-built**.
A retrieval cache would score the same. The win is architectural existence, not cognition.

The ceiling is concrete: the moment we (a) add an 11th location the model has never seen, or
(b) ask "what did Emma do before she left the kitchen?" (composition over events), or
(c) drop `reverse_templates` and give only the narrative + query, accuracy falls to random.
We have built a *structured reader*, not a *world model*.

---

## 2. The theory: what "true learning" requires

My core claim is that **world modeling is a predictive, not a retrieval, task.** A world model
is a state from which *future observations are predictable and past observations are
consistent*. Retrieval (our current setup) is the degenerate case where the only "future" we
ask about is a query we already wired an answer for. To escape it, four properties must hold:

**P1. The ontology is discovered, not enumerated.** Entities, items, and locations should be
*induced* from raw text as a variable-size set of slots (object-centric / slot-attention /
learned inductive points), not assigned fixed indices from `NAME_POOL`. The model learns "there
are some objects here, each with a state" the way a reader does — without a roster.

**P2. The read path is generative, not classificatory.** Replace `loc_head`/`holder_head`
(closed classifiers) with one decoder that *generates* the answer token-by-token conditioned on
the queried slot. This forces the latent slot to contain the fact in a form a language model
can *verbalize* — the real "speaking" test, and the bridge from structured-lookup to language.
(Recall/PASSWORD is the prototype; extend it to all fields.)

**P3. The write path is trained by prediction, not by inverted labels.** Stop feeding the model
the `reverse_templates` table. Instead train the world state with a **JEPA-style objective**:
- *Consistency:* the latent state after reading the prefix should predict the state after
  reading the whole narrative (the state converges to the same world regardless of partial
  observation).
- *Future prediction:* the state should predict the *next event* (who moves where next). This
  is the causal role that turns a cache into a model.
This removes the exact-target scaffold while keeping the model honest (no "low loss, random
accuracy" because there is no query shortcut to collapse into).

**P4. Thinking is iterative and *runs*.** Our write is one encoder pass + last-mention pooling
— a single step. True thinking is **N latent update steps** that re-derive consequences (the
original think-once/speak-many loop, which earlier collapsed). Train the loop with P3's
predictive signal: after K internal steps the state predicts the future *better* than after 1.
And train it to actually *run* — readiness/early-exit on a real prediction-gain signal, not a
saturated head (the earlier T09 failure). The original hypothesis — *more latent computation
improves reasoning* — is finally testable on a model that represents structure.

### The unifying mechanism

All four properties fall out of one mechanism: **make the latent state a predictive model of
the observation stream, addressed by discovered object slots, read by a generative decoder.**
Hand-holding is what you do when you can't yet train that objective (you substitute exact
targets). The research program below is the *curriculum that retires each substitute*.

---

## 3. The de-handholding curriculum

Each rung removes exactly one scaffold and is promoted only when the previous rung still learns.
This is deliberately incremental — the failure modes we already hit (silent-zero supervision,
collapsing heads) were caused by removing too much at once.

**Rung 0 — Structured reader (DONE).** Exact inverted targets, closed vocab, classification
reads. *Proves the representation exists.* (Current Model C.)

**Rung 1 — Generative reads, exact targets.** Keep `reverse_templates` targets, but replace
`loc_head`/`holder_head` with the shared answer decoder: predict the location/holder *tokens*
from the slot. *Test:* does the slot contain the fact well enough to verbalize it? If
classification worked (0.27/0.37/0.21) but generation fails, the slot is a retrieval artifact,
not a representation — the single most informative diagnostic we lack today.

**Rung 2 — Weak targets.** Stop giving *every* query's answer. Supervise only a random subset
of fields per sample, or add label noise. *Test:* can the state infer the un-supervised field
from the supervised ones (e.g., infer inventory from observed holder events)? This is the first
step from memorization toward inference.

**Rung 3 — Open vocabulary.** Remove `LOC_POOL`/`ITEM_POOL` from the model. Locations/holders
are generated as free tokens drawn from the narrative's own vocabulary. *Test:* zero-shot on a
location/name the model saw in *other* contexts but never as an answer. Systematic
generalization, not retrieval.

**Rung 4 — Consistency, no query labels.** Drop query answers entirely. Train only P3's
*consistency* loss: prefix-state predicts full-state. The model must build the *same* world
from partial and full observations. *Test:* freeze the writer, probe with Rung-1 generative
reads (now unsupervised) — does it still answer? If yes, we have a self-supervised world model.

**Rung 5 — Future prediction / dynamics.** Add P3's *next-event* loss. The state now predicts
what happens next, not just reconciles observations. *Test:* roll the state forward and ask
"where will the apple be after two more moves?" — a query type never in the training queries.

**Rung 6 — Discovered ontology.** Replace fixed `N_NAMES`/`N_ITEMS` slots with a learned,
variable-size slot set (slot attention over token reps, or learned queries). *Test:* present a
narrative with *k* entities where *k* is unseen at training; does the model allocate ~*k*
slots and track them? This retires scaffold #1.

**Rung 7 — Iterative latent reasoning that runs.** Replace last-mention pooling with the P4
*N-step* loop, trained by prediction gain. *Test:* does accuracy on multi-hop queries *increase
with K* (the original hypothesis)? Measure compute-vs-accuracy explicitly.

**Rung 8 — Continual / interactive worlds.** Static narrative + queries → an agent that acts
and observes; the world state *persists and updates* across steps (the tape/SSM role). *Test:*
after 100 interleaved actions, is the state still correct and cheap to update? This is where
"think once, update incrementally" becomes the computational win, not a lookup trick.

---

## 4. How we'll know we've arrived (and the risks)

**Arrival criteria (none of which hand-holding satisfies):**
- **Systematic generalization:** train on entity/relation pairs A, B, R, S; test on (A, S) and
  on a *novel* entity C. A retrieval cache scores ~0; a world model generalizes.
- **Open-set:** answer about a location/name never labeled as an answer during training.
- **Self-supervised utility:** after Rung 4, the unsupervised state still serves Rung-1 reads.
- **Compute scaling:** in Rung 7, more latent steps ⇒ better multi-hop accuracy.
- **Persistence:** in Rung 8, state stays correct across a long interactive trajectory.

**Risks (and how the curriculum contains them):**
- *Trivial-prediction collapse* (state predicts the mean world). Mitigate with the
  info-theoretic floor we already computed (L\* ≈ 5 floats) and contrastive terms; watch
  exact-match, never just loss.
- *Unaligned emergent slots* (the model discovers objects that aren't our entities). Expected
  and fine — probe with generative reads; if it can't answer, the induction is wrong, not the
  idea.
- *Generation gap* (classification works, verbalization fails at Rung 1). This is the key
  negative result to *expect and respect* — it would mean our slots are lookup artifacts. If it
  happens, the fix is a stronger decoder + more capacity, not re-adding hand-holding.
- *Optimization instability* at Rungs 4–7 (the earlier loop collapse). Contained by promoting a
  rung only after the prior one learns, and by keeping a frozen-writer probe so we can see
  exactly which scaffold's removal broke things.

---

## 5. Reconnect to the original hypothesis

The project began with: *a model with `latent_state_update() × N` + `decode_token() × M` can
outperform an equal-capacity autoregressive model on long-horizon reasoning.* Earlier attempts
"failed" only because the latent state held *nothing retrievable* (the `NONE` cheat, the
frozen loop, the `WHERE`-needs-a-tape gap). Model C fixes the *representation*: a structured,
addressable, read-head-served state. That was the missing primitive.

The remaining gap is the *learning*: the state is filled by us, not earned. This document's
theory is that the same structured state, once trained by **prediction + consistency +
discovered ontology + a running latent loop**, becomes a genuine world model — and *then* the
original win condition is finally measurable: amortized think-once should beat per-query
re-encoding on exactly the long-horizon, multi-query, open-vocab tasks that hand-holding hides.

**Research ladder, revised:**
| Level | Question | Status |
|---|---|---|
| 0 | Does latent state work at all? | ✅ (Model C: 0.21–0.37 above random) |
| 1 | Does it *learn* structure, or just retrieve it? | 🟡 Rung 0 only; Rungs 1–3 pending |
| 2 | Can it verbalize a fact (generative read)? | 🟡 recall path unwired; Rung 1 pending |
| 3 | Can it infer un-supervised fields? | 🟡 Rung 2 pending |
| 4 | Can it build the world with *no* query labels? | 🟡 Rung 4 pending |
| 5 | Can it predict dynamics / the future? | 🟡 Rung 5 pending |
| 6 | Does the ontology emerge? | 🟡 Rung 6 pending |
| 7 | Does *more latent thinking* help? | 🟡 Rung 7 pending (original hypothesis) |
| 8 | Does state survive continual interaction? | 🟡 Rung 8 pending |

The training wheels are off the *representation*. Everything below is the theory for taking
them off the *learning*.
