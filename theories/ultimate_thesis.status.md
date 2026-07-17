# ultimate_thesis.status

Status of the core thesis (small + observable + state + branching).

## Claims

- **U1** — *A small system can match large models for narrow tasks.*
  Status: **partially proven** — RWKV-nano at 100K–1M params trained cleanly on multiple tasks.

- **U2** — *Byte-level RWKV avoids tokenizer artifacts.*
  Status: **in progress** — `byte-state-byte.md` experiments ongoing.

- **U3** — *State, not weights, is the right memory substrate.*
  Status: **not proven** — both `dendrite_memory` (LoRA) and `dendrite_growth` (extensions) untested.

- **U4** — *Adaptive looping (until confident) > fixed-length.*
  Status: **partially proven** — `adaptive-exit-entropy.md`, `encoder-patcher-decoder.py`.

- **U5** — *Multi-rate (bytes fast, patches slow) is optimal.*
  Status: **in progress** — adaptive_loop_001 proved encoder puncture works.

- **U6** — *Branches/growth avoid catastrophic forgetting.*
  Status: **not tested** — `dendrite_growth` theory defined, no implementation.

- **U7** — *Realtime + branch-addition = constantly learns without retraining.*
  Status: **untested**.

## Mechanism Gaps

- State-based memory: no benchmark for "uses state as memory correctly"
- Branch growth: no `src/rwkv_growing.py` yet
- Realtime: no synthetic device-stream generator

## Follow-Ups (Cheapest → Expensive)

1. Fix known bugs in `dendrite_rwkv_001` and `logic_niiah_*` exp families
2. Implement `src/rwkv_growing.py`
3. Run `dendrite_growth_001` (one branch, isolation test)
4. Build synthetic device-stream generator (`realtime_001`)
5. Measure end-to-end CPU latency for byte-state-byte forward pass (U7 budget)
