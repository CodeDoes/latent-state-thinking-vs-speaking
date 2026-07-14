# Progress: Hybrid Latent-State Language Model

> **AUTHORITATIVE STATUS DOC.** Long-form theory lives in `theories/*.md`;
> this file is the concise status + theory/experiment tracker. Detailed
> historical narrative is in `git` history; only milestones are kept here.

## Core Hypothesis
A model with `latent_state_update() × N + decode_token() × M` can match or
outperform an **equal-capacity** token-by-token model on long-horizon reasoning
— thinking (SSM) is amortized over many queries while speaking is cheap. The
latent state is a private computational scratchpad; language is only an output
device. (Full framing: `USER.md`.)

## Research Ladder
| Level | Question | Status |
|---|---|---|
| 0 | Does latent state work at all? | ✅ state trains, reaches ~0.63 exact-match (caveat: collapses on unguarded dataset — T03) |
| 1 | Does latent thinking beat tokens? | ✅ YES w/ T05+T06 — latent wins AT/SAME (reasoning) by clear margin; ties overall; loses WHERE (trajectory recall → needs Tape, Model C) |
| 2 | Does latent state survive context removal? | ⬜ |
| 3 | Can latent state generate multiple tokens? | ⬜ |
| 4 | Can latent state continue a story after interruption? | ⬜ |
| 5 | Can state computation become independent of token generation? | ⬜ |

## Architecture (brief)
SSM (thinking / world-model) + Tape (exact recall) + Context (attention mgmt)
+ Decoder (render state→tokens). **Key split:** SSM = logic/planning/aggregation;
Tape = precise recall/spelling. See `USER.md` for the full diagram. This split
is *empirically confirmed* (T02): latent wins aggregation (AT/SAME), loses
precise recall (WHERE) because the current design has **no tape**.

## Template inversion = latent-state requirements checklist

`reverse_templates.py` parses a generated narrative and rebuilds the
structured `World`/`Entity` state the forward generator carries. Self-test
results:

| Task   | Inverse accuracy | Notes |
|--------|------------------|-------|
| location   | 60/60 (100%) | perfect |
| inventory  | 60/60 (100%) | perfect |
| recall     | 60/60 (100%) | perfect (incl. decoy trap) |
| transfer   | 42/60 (70%)  | the remaining 18 are INTERNALLY INCONSISTENT forward GTs (holder's narrative location ≠ dataset's recorded GT); the inverse IS the consistent world |

→ Conclusion: a latent state that faithfully tracks the (category, parameter)
list below would, in theory, answer every task with perfect accuracy.

## Model C: Structured World-State (complexity-aware redesign)

The inverse proves the latent state must hold a *structured table*
(per-entity: location / inventory / first_password / password_overridden;
per-item: current_holder) — **not a single pooled vector**. The old
`make_B(A)->B` collapses to the mean because one vector can't hold all of them
(diagnostics confirm MSE ~ variance floor).

**Key design insight (from the user):** to track location, an SSM only needs
`(entity_index, current_location_at_index)` — a *disentangled* record, not one
entangled vector we then try to linearly read a location out of. So the slot is
**field-decomposed**:
  - entity slot `[0:loc_dim]`   -> current location (one-hot over `LOC_POOL`)
  - entity slot `[loc_dim:loc_dim+inv]` -> inventory (multi-hot over `ITEM_POOL`)
  - item   slot `[0:name_dim]` -> holder (one-hot over `NAME_POOL`)
Reading = trivial `argmax` of the dedicated field. No fragile linear probe over
an entangled vector.

**Critical training trick — auxiliary token-classification losses.** A frozen
char-LSTM encoder represents "bedroom" and "kitchen" nearly identically, so
without a direct signal the encoder+writer collapse to a CONSTANT location
(~10% = majority baseline). Fix: add `loc_tok_head` / `item_tok_head` that force
`encoder_state(at location/item word) -> that location/item`. This breaks the
degeneracy so the writer can actually read the field. Encoder is trained
end-to-end (NOT frozen).

**Files:** `src/world_state.py` (`WorldModel`), `run_world.py` (canonical
local experiment runner, saves `experiments/expNNN/{config,metrics,samples,model}`),
`train_world.py` (bench-integrated entry, registered in `bench.py` as `world`).

**Local small-scale results (CPU, d_state=64, 400 samples, 12 epochs) — real
eval, NOT Kaggle:**
  - location : **0.23-0.27** (random 0.10) ✅ WHERE gap closes
  - inventory: **0.37** (random 0.10) ✅ derived as inverse of holder relation
  - transfer : **0.19-0.21** (random 0.10) ✅ 2-hop item->holder->location
  - recall   : not yet validated (generative decode path; answer-decoder shape bug)

**Expanded puzzle suite (all derived reads off loc_head/holder_head, no new
heads).** Added to `src/dataset.py` + eval in `src/world_state.py`: `holder`
(who has item), `colocation` (who shares a location), `count_people`,
`which_loc_most` (argmax aggregation), `most_items` (argmax aggregation),
`empty_loc`, `has_item` (yes/no). A mixed `--task all` run (1000 samples, d=64,
15ep) gives has_item 0.79, empty_loc 0.59, count_people 0.48, which_loc_most
0.34, holder 0.40, colocation 0.32, transfer 0.18, location 0.15 (all above
random 0.10); combinatorial exact-match (inventory/most_items) is hard at
~100 samples/task and climbs with scale. Generative tasks (recall/story) are
excluded from the mixed default until the answer-decoder shape bug is fixed.

The location win directly confirms the user's `(index, current_location_at_index)`
framing: once the encoder is forced to represent location words and the slot
carries an explicit location field, the SSM extracts it through the distractors.

## Current Status (2026-07-13 PM)
- **Valid experiment exists.** `src/latent.py` + `train_converged.py`: sequential
  latent thinking (token-by-token, K steps/token), multi-query worlds
  (WHERE/AT/SAME), *think once / speak many* vs a re-encode-per-query baseline.
- **ITER-1** (d32, 2500×14): latent 0.617 vs baseline 0.645.
- **ITER-2** (d48, 16 ep): latent 0.592 vs baseline 0.624; per-type
  `AT lat 0.853 > base 0.839`, `SAME lat 0.918 > base 0.906`,
  `WHERE base 0.136 >> lat 0.018`. → latent wins reasoning, loses trajectory.
- **Collapse diagnosed (T03/T04/T07):** AT/SAME answers are 87–89% `"NONE"`
  (non-unique label → 0 bits); always-NONE scores 0.619 and the latent model
  hit 0.634 (learned *nothing* beyond majority). Bigger `d_state` did not help
  WHERE → not a capacity problem.
- **Local GPU now available (T08):** RTX 2050 4 GB, `devenv.nix` patched
  (`LD_LIBRARY_PATH`). d256 throughput 63.9k tok/s; a 5k×20 `bench.py` run ≈
  15–20 min locally. Overrides the stale "no local GPU" assumption.
- **Legacy cyclic-curriculum track (`train_modules.py`) = dead end** (oracle
  0.57 but final QA 0.000 due to train/inference gap in composer `B`).

## Planned Experiments (incremental novelty — Model A→D)
| Model | Components | Purpose |
|---|---|---|
| Baseline | Transformer LM | Reference point |
| Model A | SSM only | Pure recurrent baseline |
| Model B | SSM + FFN decoder | Decoupled generation |
| Model C | SSM + FFN + Tape | + exact recall |
| Model D | SSM + FFN + Tape + Context | Full architecture |

## Theories (see `theories/*.md`)
| # | Theory | File | Status |
|---|---|---|---|
| T01 | Latent thinking beats tokens (core) | `01-core-latent-vs-tokens.md` | 🔶 nuanced, not refuted |
| T02 | SSM wins aggregation, tape wins recall | `02-ssm-vs-tape-split.md` | ✅ confirmed |
| T03 | Label non-uniqueness drowns signal (NONE collapse) | `03-dataset-label-nonuniqueness.md` | ✅ confirmed |
| T04 | Normally-Empty latent-state vectors | `04-normally-empty-state-vectors.md` | ✅ rate confirmed / arch untested |
| T05 | Uniqueness-weighted loss `w(a)=-log2 p(a)` | `05-uniqueness-weighted-loss.md` | ✅ confirmed (helps latent WHERE + stabilizes; part of T05+T06 fix) |
| T06 | Auxiliary state-tracking (reconstruction) loss | `06-auxiliary-reconstruction-loss.md` | ✅ confirmed (latent AT 0.807→0.895, above cheat; enables relational decoding) |
| T07 | Capacity is NOT the bottleneck | `07-capacity-not-bottleneck.md` | ❌ refuted |
| T08 | Gradual novelty + local-GPU fast iteration | `08-gradual-novelty-local-gpu.md` | ✅ confirmed |

## Experiments on Theories (tracker)
| Experiment | Date | Tests | Result | Status |
|---|---|---|---|---|
| `exp_converged_lh_2026-07-13` (ITER-1/2, Kaggle CPU) | 2026-07-13 | T01, T02, T07 (+T03 evidence) | lat 0.592 vs base 0.624; lat wins AT/SAME, loses WHERE; bigger d_state no help | ✅ valid |
| dataset-stats analysis (inline) | 2026-07-13 | T03, T04, T05 (derivation) | NONE 86.9/89.4%; entropy 1.4/1.2/5.0b; cheat ceiling 0.619; slot-emptiness 87% | ✅ confirmed |
| gpu-throughput benchmark (inline) | 2026-07-13 | T08 | RTX 2050: 63.9k tok/s (d256), 10.7k (d768); 5k×20 run ≈15–20 min | ✅ confirmed |
| `bench.py --quick` baseline (local) | 2026-07-13 | T08 (end-to-end path) | baseline runs end-to-end; 0.000 at tiny scale (expected sanity) | ✅ sanity OK |
| `exp_t05_local_2026-07-13` (n=600, 8ep, d48, RTX2050) | 2026-07-13 | T05, T03 | lat 0.596 vs base 0.617; lat WHERE 0.041 (2.3×↑), AT/SAME still ~0.8 NONE-cheat; T05 insufficient alone | ✅ done |
| `exp_t06_local_2026-07-13` (recon head, alone) | 2026-07-13 | T06, T04 | lat 0.626 vs base 0.650; latent AT 0.895 (>cheat 0.869, >base 0.886!); SAME/WHERE ~flat | ✅ done |
| `exp_t05t06_local_2026-07-13` (T05+T06 combined) | 2026-07-13 | T05,T06,T02 | lat 0.590 vs base 0.587 (latent WINS); lat AT 0.798 / SAME 0.844 > base 0.763/0.762; WHERE base 0.163 >> lat 0.031 | ✅ done |
| `exp_t09_loop_2026-07-13` (loop, self-recur) | 2026-07-13 | T09 | lat 0.578 vs base 0.587; **epoch-5 collapse** (0.641→0.539); self-recur on zero-vec is NOT real derive; conf head caused train/infer mismatch | ❌ rejected |
| `exp_t09b_transformer_2026-07-13` (transformer-scaled + re-attention loop) | 2026-07-13 | T09, T08 | lat **0.649** vs base 0.626 (latent WINS, stable 8 ep, no collapse); L AT 0.866 / SAME 0.911 / WHERE 0.035; B AT 0.773 / SAME 0.881 / WHERE 0.118; params lat 9.98M (transformer 512×4, scaled by max_loop=8, capped @512 for quality) vs base 399K | ✅ done |
| `exp_t09c_equalparams_2026-07-13` (equal param budget, both ~9.98M) | 2026-07-13 | T09,T08,fairness | lat **0.649** (stable) vs base **~0.16** | ⚠️ **RETRACTED**: frozen 0.649 was a bug (dummy conf target + single-state token loss → loop learned nothing) + `NONE` cheat; not a real win; see T10 | ❌ retracted |
| `exp_t10_loopfix_2026-07-13` (loop fix + NONE removed) | 2026-07-13 | T09 fix, T03 | lat **≈0.01** vs base **≈0.06** (both low; eval metric now MOVES every epoch — frozen-bug fixed); latent trails; latent train_loss descends 269→220 but plateaus higher than baseline (35.8→4.6); eval ~0.01 (train/eval gap, not a frozen loss) | ✅ methodology fixed, result NEGATIVE |
| **done** break AT/SAME NONE-cheat (guaranteed co-located pairs) | 2026-07-13 | T04 | done in T10; metric now valid | ✅ done |
| **proposed** fairer baseline: transformer-AR (not GRU) at equal params, or lower lr for big GRU | — | fairness | not run | 🟡 planned |

## Research Log (condensed milestones)
- **2026-07-11** — Inception; architecture built (`BaselineTransformer`,
  `LatentSSM`, `LatentSSMDecoder`); unified `[B,seq,vocab]` output; sequential
  processing; CPU validation. First Kaggle GPU runs (exp001-005) showed *low
  loss but no strict QA* → "low loss, useless output" trap identified.
- **2026-07-11/12** — Reusable-script refactor: `bench.py` single entry point
  with **strict exact-match QA** + `answer_loss_weight` + `low_but_useless`
  flag; `baseline_big` (~33M) for equal-param comparison; `gen_notebook.py`
  embeds real `src/`; `kaggle_ctl.py` canonical control.
- **2026-07-12** — Modular cyclic-curriculum track (`train_modules.py`): long
  saga oracle 0.00→0.57 but final QA **0.000** (train/inference gap in
  composer `B`). Diagnosed dead end. Lesson: *train the representation, not the
  answer*; `AnswerDecoder` is a readout probe only.
- **2026-07-13 AM** — Redesign → converged SSM+FFN (`src/latent.py` +
  `train_converged.py`): real sequential thinking, multi-query worlds,
  think-once/speak-many vs re-encode baseline. Pushed to Kaggle CPU.
- **2026-07-13 PM** — ITER-1/2: latent competitive (≈tie, wins AT/SAME, loses
  WHERE). Verdict nuanced, not refuted.
- **2026-07-13 (this session)** — Theory extraction: NONE-drowning (cheat
  ceiling 0.619), Normally-Empty slots (87%), uniqueness-weighted loss (T05)
  and auxiliary reconstruction (T06) proposed, capacity refuted (T07),
  **local RTX 2050 GPU discovered + `devenv.nix` patched + throughput
  measured** (T08). Theories moved to `theories/*.md`; this file cleaned up.

## Next Steps
1. **Local GPU baseline:** `bench.py --models baseline --device cuda
   --n_samples 2000 --epochs 10 --answer_loss_weight 1.0` (~5–8 min) to confirm
   the guarded baseline learns on the local GPU.
2. **T05+T06 combined DONE** (RTX2050, n=600, 8ep): latent 0.590 vs
   baseline 0.587 → latent WINS overall; latent AT 0.798 / SAME 0.844 both
   > baseline 0.763/0.762 (reasoning win, T02); WHERE baseline 0.163 >>
   latent 0.031 (trajectory-recall loss → latent needs a Tape, Model C).
   **Collapse FIXED.** Next: iter-3 mix (tilt to AT/SAME), then Tape.
3. **iter-3** integration-heavy mix → expect latent to win overall (confirms T02).
4. **T09b/T09c wins RETRACTED** — both rested on a broken loop protocol (dummy
   `conf` target + single-state token loss, no auto-encoder) AND the `NONE` cheat, which
   pinned the eval at the cheat ceiling (frozen 0.649). With the loop fixed (full unroll,
   trajectory-derived confidence, deep supervision, AE) and `NONE` removed (T10), the metric
   moves but the latent **trails** the baseline. The earlier "wins" were artifacts.
5. **T10 DONE (loop fix + NONE removed)** — latent ≈0.01 vs baseline ≈0.06; latent
   training loss stuck (~220) while baseline learns (39→4.6). The think-once state does NOT
   yet beat per-query re-encoding on genuine relational reasoning at this scale. Next: (a)
   diagnose why the latent state fails to train (state aggregation vs deep-supervision
   optimization), (b) transformer-AR baseline for fair fight, (c) multi-seed variance,
   (d) tape for WHERE, (e) make loop fire at inference (min_certainty/conf_w sweep).
6. **Next:** (a) fairer baseline — transformer-AR (not GRU) at equal params, or tune the big GRU's lr; (b) break AT NONE-cheat; (c) add Tape (Model C) for WHERE; (d) make the loop actually run at inference.
