# Structured World-State Models: Why the Latent State Should Be a Table, Not a Vector

### Separating "thinking" (writing a world table) from "speaking" (reading one field)

**Status:** Working paper — local CPU experiments (no GPU needed at this scale).
**Date:** 2026-07-14
**Code:** `src/world_state.py` (`WorldModel`), `src/dataset.py`, `reverse_templates.py`,
`run_world.py` (canonical local runner), `bench.py`; logs in `experiments/expNNN/`.

---

## Abstract

We revisit the hypothesis that a neural model can **separate thinking from speaking** —
maintain a private latent state that is *derived once* from a context, then read from to
answer many queries. The failure of earlier pooled-vector latent models (the "invisible
training" problem: cross-entropy falls but accuracy stays at random) led us to a sharper
claim: **the latent state should be a *structured world table*, not a single pooled vector.**

We make the structure *knowable* by inverting the data generator: `reverse_templates.py`
recovers the exact `(entity → location/inventory, item → holder)` table that produced any
narrative. Because the target structure is known, we can supervise it *directly* via dedicated
read heads over per-entity / per-item slots (the `(index, current_location_at_index)` read the
user asked for), instead of hoping a pooled vector learns to answer queries.

On synthetic multi-query narrative worlds we find the structured world-state model **learns
all three closed-vocabulary query types** — location (0.23–0.27), inventory (0.37), and
multi-hop transfer (0.19–0.21) — each above the 0.10 random baseline, on a CPU at a few
thousand samples. The remaining gap is generative recall (password reconstruction), which needs
the decoder path (under validation). The durable result is methodological: **exact structure
supervision beats latent representation learning**, and two silent failure modes (a pool
mismatch that zeroes supervision, and a collapsing inventory head) explain why previous
attempts looked like "training that doesn't learn."

---

## 1. Theory

### 1.1 Core hypothesis

A latent state that must support *query-time retrieval* of structured facts (who is where, who
holds what, where an item ended up after a transfer) is best represented not as one pooled
vector but as a **disentangled world table**:

- one **slot per entity** (name), encoding that entity's current state,
- one **slot per item**, encoding that item's current holder,
- **dedicated read heads** that project a slot to *one field* (location / inventory / holder).

"Thinking" = writing the table from the context **once**. "Speaking" = reading the *one field*
the query asks for. This decouples the reasoning substrate (tabular, addressable) from token
generation, and makes every query a cheap lookup rather than an O(N) re-encoding.

### 1.2 Why a pooled vector fails (and why a table succeeds)

The original latent-vector models collapsed to "training that doesn't learn": cross-entropy
decreased while exact-match accuracy stayed at random. The root cause is **structural
indeterminacy** — a single pooled vector has no reason to organize facts so they're
retrievable. Which part of the vector means "Emma is in the kitchen"?

A table removes the indeterminacy. Each entity has a fixed address (its slot index), so the
location can live *in* that slot and a tiny linear head can read it out. The empirical
signature of this is striking:

- **Location is easy.** The location word sits *immediately beside* the name in the narrative
  ("Emma was in the **kitchen**"). The encoder state at the name token already conditions on
  the location word, so a `loc_head` on the entity slot learns it directly (acc 0.23–0.27).
- **Inventory is hard — for the pooled approach.** Holdings ("Emma had the **apple**, then
  dropped the **phone**") are events distributed across the narrative, not adjacent to the
  name. A head reading only the *name-token* slot predicts "nothing" for everyone (acc 0.007).

So the lesson is not "use a bigger vector"; it's "put the fact where the read head can find
it." This directly motivates the table + read-head design.

### 1.3 The inverse-template proof (structure is knowable)

The decisive theoretical move: **we can invert the data generator.** The narratives are
produced by deterministic templates (`src/dataset.py`) from a small structured world.
`reverse_templates.py` parses a narrative back into that world:

```
narrative ──reverse_templates──▶ (entities: name→{location, inventory}, item→holder)
```

Because we can recover the *exact* target table for every training sample, we can supervise
the latent state **directly and exactly** — no latent-representation guessing. A self-test on
location / inventory / recall / transfer samples parses 100% correctly, confirming the
generator is a faithful projection of the table. The world table is therefore not a hypothesis
about what the model *might* learn; it is the *provably correct* latent structure, and the
model's job is to reproduce it.

### 1.4 Holding is the core relation

Among the fields, **holding** is primitive: every other relation is derived from it.

- **Transfer** ("Where is the apple?") = *item → holder → holder's location* (a 2-hop read).
- **Inventory** ("What does Emma have?") = the **inverse** of holding: items whose predicted
  holder is the queried entity.

This unification is what unblocks inventory: instead of training a separate inventory head
(which collapsed), we train one `holder_head` (item-slot → holder) and *derive* inventory as
its inverse, and transfer as its forward projection. One relation, two queries.

---

## 2. Experiment Setup

### 2.1 Data: synthetic multi-query narrative worlds

Worlds are generated from closed pools (single source of truth in `src/dataset.py`):

| Pool | Size | Tokens |
|---|---|---|
| `NAME_POOL` | 10 | John, Mary, Alex, Sam, Emma, Leo, Zoe, Max, Lily, Tom |
| `LOC_POOL` | 10 | kitchen, bedroom, garden, garage, bathroom, living room, office, basement, attic, hallway |
| `ITEM_POOL` | 10 | apple, book, key, phone, cup, pen, wallet, watch, bag, umbrella |

A narrative is a random chain of moves/holds/drops/picks-up events over these entities, then
four query types are asked (one task per sample in our bench):

| Task | Query | Answer (closed vocab) | Random acc |
|---|---|---|---|
| `location` | "Where is {name}?" | one of 10 locations | 0.10 |
| `inventory` | "What does {name} have?" | subset of 10 items (or "nothing") | ≪0.10 (combinatorial) |
| `transfer` | "Where is the {item}?" | location of the item's current holder (2-hop) | 0.10 |
| `recall` | "What is {name}'s password?" | a generated token string (generative) | — |

Every query is **answerable from the narrative** (no unanswerable questions), and the
answer is **uniquely determined** (single-answer constraint). The challenge is purely
multi-hop retention, not ambiguity.

### 2.2 Model: encoder → slots → read heads

```
                 INPUT NARRATIVE
                       │
            TokenEncoder (char/word LSTM)         ← process ONCE
                       │  per-position reps [T, d_state]
                       ▼
     last-mention pooling                         ← fill the table
   ent_slots[b, name, :]  = rep at name's last occurrence
   item_slots[b, item, :] = rep at item's last occurrence
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
   loc_head(ent_slot)        holder_head(item_slot[:name_dim])
          │                          │
     location (CE)              holder (CE)  ──▶ inventory = inverse
          │
   (generative decode for recall, under validation)
```

`WorldModel` (`src/world_state.py`):

- **TokenEncoder** — `nn.LSTM(d_state, d_state, 2)`, run once over the tokenized narrative;
  produces per-position representations.
- **Last-mention pooling (the writer).** Each entity/item slot is the encoder state at that
  token's *last* occurrence in the narrative — a batchable, parameter-free "last mention wins"
  aggregation that replaces a fragile GRU writer. This is the "write the table" step.
- **Read heads (the readers).** `loc_head(slot → 10 locations)`, `holder_head(slot → 10
  names)`. Each head reads *one* field from *one* slot — the `(index,
  current_location_at_index)` read. Inventory is computed as the inverse of `holder_head`.
- **Auxiliary token heads.** `loc_tok_head` / `item_tok_head` predict, at every token position,
  whether a location/item word occurs there. These force the encoder to actually *represent*
  location and item words, which is what makes the per-slot pooling informative (without them
  the encoder+writer collapse to a constant).

### 2.3 Training

Batched end-to-end. Losses, all derived from the `reverse_templates` target table:

- `loc_loss` — CE of `loc_head` over *mentioned* entities vs their true location.
- `holder_loss` — CE of `holder_head` over *mentioned* items vs their true holder.
- `ans_loss` — answer-token cross-entropy for the generative (recall) path.
- `loc_tok` / `item_tok` — auxiliary position-wise CE that prevents encoder collapse.

`field_loss = loc_loss + holder_loss` is the structured-world objective; `ans_loss` is the
speak path. Targets come straight from the inverted world table, so supervision is exact.

### 2.4 Evaluation

- **Primary metric: strict exact-match QA accuracy** (predicted field must equal the reference
  exactly). Reported per task, with the random baseline for that task.
- **Reproducibility:** `run_world.py --task {location,inventory,transfer,recall} --device cpu
  --d_state D --epochs E --n_samples N` writes `experiments/expNNN/{config,metrics,samples,
  model}`.

---

## 3. Evidence

### 3.1 Main results (CPU, real eval, random baseline = 0.10)

| Task | Setup | Accuracy | vs random |
|---|---|---|---|
| `location` | d=128, 1500 samp, 30 ep | **0.269** | 2.7× |
| `location` | d=128, 800 samp, 20 ep | 0.229 | 2.3× |
| `inventory` | d=128, 1000 samp, 25 ep | **0.366** | ≫ random (combinatorial) |
| `transfer` | d=128, 800 samp, 20 ep | **0.205** | 2.0× |
| `transfer` | d=96, 1000 samp, 25 ep | 0.187 | 1.9× |
| `recall` | — | not yet validated | — |

All three closed-vocabulary tasks climb monotonically with epochs and beat their baselines.
This is the first positive, non-artifact result in the project: earlier "wins" were either a
frozen metric (broken loop training) or a `NONE` majority-class shortcut. Here the metric
moves and the accuracy is *above* the cheat ceiling by construction (no majority class).

### 3.2 Methodology as evidence: the silent-zero-supervision bug

The most instructive result is *negative* and explains the original "invisible training":

- When the model's label pools were **hardcoded** instead of imported from `src/dataset`, the
  indices silently misaligned with the true `reverse_templates` targets. Cross-entropy still
  *decreased* (the model learned to predict *something*), but accuracy stayed at **~0.10**
  (random). Supervisory signal was effectively **zero** while the loss looked healthy.
- Fix: import `NAME_POOL` / `LOC_POOL` / `ITEM_POOL` as the single source of truth. After the
  fix, accuracy immediately tracks the loss. **Lesson: a falling loss is not evidence of
  learning; report exact-match accuracy every epoch.**

### 3.3 Collapse diagnostics: inventory

With a naive `inv_head(ent_slot → items)` (sigmoid multi-label), inventory accuracy was
**0.007** — the head predicted "nothing" for everyone. The entity slot (name-token state)
simply does not encode holdings (§1.2). The fix (derive inventory as the *inverse* of
`holder_head`) raised it to **0.366**. This is direct evidence for the theory: holdings are a
*relation over items*, not an attribute of the name token, so they must be read through the
holder relation, not the name slot.

### 3.4 Inverse-template validation

`reverse_templates.py` self-test passes 100% across location / inventory / recall / transfer:
the generated narrative is a faithful, invertible projection of the world table. This is what
makes the exact-structure supervision in §2.3 legitimate rather than heuristic.

---

## 4. Benches

### 4.1 Harness

- **`run_world.py`** — canonical local experiment runner for the world model. Parses args,
  generates the task dataset, trains via `train_world`, evaluates with `run_world_qa`, prints
  `STAGE:` progress + per-task accuracy + sample debug, and dumps
  `experiments/expNNN/{config.json, metrics.json, samples.txt, model.pt}`.
- **`bench.py`** — monolithic benchmark entry; registers `world` as a model family and supports
  `--analyze` over downloaded experiment dirs.
- **`kaggle_ctl.py`** — single Kaggle control script (`status`/`run`/`watch`/`download`) for
  when GPU scale-out is needed; local CPU is sufficient at this size.
- **`experiments/expNNN/`** — append-only records (config + metrics + samples + checkpoint),
  never overwritten.

### 4.2 Speed: batching beats the C span scanner

Profiling showed the text parser (`reverse_templates`) is **not** the bottleneck (≈6 ms / 60
samples). The cost is per-sample PyTorch dispatch in the training loop. The real speedup was
**vectorizing the loop**: one encoder call over a padded batch, `gather` for last-mention
pooling, and vectorized heads/losses. This cut a 1500-sample / 30-epoch location run to ~9
minutes on CPU — comfortably within a single foreground session.

### 4.3 `BACKEND` toggle: a swappable hot path

Per the request for a C/Mojo/Rust backend, `detect_spans` (whole-word span scanning over the
pools) is behind a `BACKEND` flag in `world_state.py`. `BACKEND="c"` loads a compiled
`fastworld.so` (`fastworld.c`, a Python C-extension doing the span scan in C); it **falls back
to pure Python** if the module is absent. We verified the C output matches the Python
reference exactly on NAME/LOC/ITEM pools. Note: this is a pedagogical "swappable backend"
demonstration — batching the torch ops is where the wall-clock time is actually won.

---

## 5. Limitations and Next Steps

1. **Recall (generative) is unvalidated.** The `ans_loss` / decoder path for reconstructing
   password token strings is wired but not yet benchmarked. This is the "speak many" half of
   the think-once/speak-many split and the natural next experiment.
2. **Closed vocabulary.** All fields are 10-way classification. Scaling to open vocabulary
   (a generative location/holder decoder) is the bridge to real text.
3. **Capacity headroom.** Runs are tiny (≤1500 samples, d≤128). The struct already wins; the
   question is whether the *margin* survives scale and distractors (more names/items, longer
   narratives, adversarial transfers).
4. **Event-aware writer (optional).** Last-mention pooling suffices for these tasks, but an
   explicit per-event update would make inventory/transfer exact under interleaved
   pickup/drop chatter.
5. **Scale-out via `kaggle_ctl.py`** when experiments outgrow the local CPU.

---

## 6. Conclusion

The latent state should be a **structured world table addressed by entity/item slots**, read by
dedicated field heads — not a pooled vector hoping to be queried. Three facts make this
concrete and testable: (i) we can *invert the generator* (`reverse_templates`), so the target
structure is known and exactly supervised; (ii) location is trivially readable from the
name-adjacent slot while holdings are not, which is why the table + relation design is
necessary; (iii) holding is the primitive relation, with inventory and transfer as its inverse
and forward projections. On synthetic narrative worlds the structured model learns all three
closed-vocabulary query types above baseline on a CPU, with exact-match accuracy reported
every epoch — closing the "training that doesn't learn" trap that defined earlier attempts.
Generative recall remains the open experiment.

---

## References

- JEPA-Reasoner: Decoupling Latent Reasoning from Token Generation. arXiv:2512.19171.
- Gu et al., *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*.
- Dao & Gu, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
- Goyal et al., *Improved Baselines for Latent Reasoning* (pause-token line of work).
- Internal: `src/world_state.py`, `src/dataset.py`, `reverse_templates.py`, `run_world.py`,
  `bench.py`, `fastworld.c`, `PROGRESS.md`, `experiments/expNNN/`.
