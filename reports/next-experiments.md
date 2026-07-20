# Next-Experiments Survey

Sorted smallest-first. Each matches the smoke-test rule. The hottest items link to arXiv papers via `python src/arxiv_query.py "<query>"`.

## 1. Adaptive-exit-entropy sweep — smoke test runnable today

[Theory](../threads/adaptive_compute/adaptive-exit-entropy.md) — 60-second runnable, CPU only.

**Sweep**: `entropy_weight ∈ {0.0, 0.001, 0.01, 0.05, 0.1}` at matched dims and 100 steps each (5 minutes total). Predicted:

- w=0.0 → loops collapse to 1 (no exploration incentive). Lambda_mean stays near 0.5 (no penalty for committing early).
- w=0.01 → balanced, encoder adapts loops 1→3 (per B5 result).
- w=0.1 → loops stay high, slower convergence.

**How to run**:
```bash
python -m src.train_adaptive_entropy --sweep --exp_id_prefix ent_sweep --steps 100
```

**Status**: smoke-tested at 100 steps with w=0.0 — `enc=1, dec=1`, loss 5.59→2.67. Confirms collapse. The full 5-arm sweep with longer steps is the next move.

**Cost**: ~6 min × 5 arms = 30 min total.

---

## 2. Dendrite Growth G1 — min viable proof

[Theory](../threads/memory_growth/dendrite_growth.md) — needs an implementation but is otherwise cheap.

**Hypothesis**: A frozen trunk + one trainable branch can learn a new task without modifying the trunk.

**What to implement** (≈ 2–3 h of code):
1. `src/rwkv_growing.py` — frozen `RWKVNano` + `GrowBranch` class (cross-attn + head).
2. Train one branch with trunk frozen on a smoke task ("sum ≥ 50").
3. Evaluate trunk outputs before and after on a probe set; check delta < ε.

**How to run**:
```bash
python src/train_grow_branch.py --task sum_threshold --exp_id grow_G1a
```

**Cost**: implementation budget + ~5 min training.

---

## 3. Token-VS-byte head — already-existence ablation

[Theory](../threads/adaptive_compute/token-vs-byte-head.md) — needs ablation script.

**Hypothesis**: byte head (vocab=258) vs token head (vocab=50k) yields same accuracy at matched params but byte head uses dense supervision.

**What exists**: model code (`src/token_byte_head_model.py`).

**What to do**: run both heads at matched model capacity, log loss + sample quality side by side.

**Cost**: ~30 min to assemble, ~10 min to run.

---

## 4. Realtime AI synthetic stream — R1

[Theory](../threads/realtime_learning/realtime_ai.md) — needs synth generator.

**Hypothesis**: byte-level RWKV with surprise-router ingests multi-channel streams and learns co-occurrence (e.g. "wif[i]-assoc, bt[ooth]-probe co-occur").

**Risk**: synthetic stream generator is the unknown. If channels are purely random, no learning. Need a *learnable* rule.

**What to build first**: `src/stream_generator.py` with two channels and a temporal coupling rule.

**Cost**: generator + smoke test ≈ 2 h before any conclusion.

---

## 5. Read-twice vs progressive-expansion matched compute

[Theory](../threads/memory_growth/read-twice.md) — needs both implementations.

**Hypothesis**: read-twice (same trunk twice) ≥ progressive-expansion (one extra layer) at matched FLOPs.

**Blocker**: progressive-expansion layer insertion is not yet implemented end-to-end.

**Cost**: ≥ 4 h to implement progressive-expansion. Skip until other proofs have completed.

---

## What's on disk that we should *not* redo

- `byte-state-byte` experiments: B5 proven (`adaptive_loop_001`), B6 partial (`byte_loop_001`). Don't re-run.
- `dendrite_rwkv_001`: blocked on data collation bug. Fix the bug *or* move on — don't keep retrying.
- `dendritic_compare`, `niiah_wc_*`: from a different thread (dendritic RWKV vs standard). Tangential to current claims. Archived.

---

## How to pick

If you have **30 minutes**: do the entropy-weight sweep. (`adaptive-exit-entropy` theory will get a clean evidence line.)

If you have **2–3 hours**: implement Dendrite Growth G1. (Highest-reward open claim.)

If you have **half a day**: both, and start designing the realtime-AI generator.

---

## Arxiv search winners worth a read

```bash
python src/arxiv_query.py "all:mixture+of+experts+adapters+isolation"
python src/arxiv_query.py "all:RWKV+byte"
python src/arxiv_query.py "all:state+space+model"
```

The Decoupled MoE paper (pulled earlier as a reference for the Dendritron correction) lives at `arXiv:2606.14243`. It splits experts from base, which is the same architectural story as Dendrite Growth — and already validated at scale.

---

*Pinned 2025-07-17. Run `python src/arxiv_query.py "<topic>"` to refresh the references above.*
