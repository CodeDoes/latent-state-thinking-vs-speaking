# dendrite_memory.status (ORIGINAL THEORY — SUPERSEDED)

> **Note**: This status tracks the *original* LoRA-based theory from `theories/dendrite_memory.md`. The corrected theory (architectural growth / state-based dendrites) is in `theories/dendrite_growth.md` (to be written) and will have its own status file.

---

## Original Claims (from `dendrite_memory.md`)

| Claim | Description | Status |
|---|---|---|
| **D1** | Frozen RWKV + independent LoRA + routing + lifecycle gates → isolated memories | **Superseded** — theory corrected to state-based |
| **D1a** | Independent LoRA prevents interference | **Superseded** — isolation = state orthogonality |
| **D1b** | Address + PPCA on hidden states beats always-first | **Superseded** — routing on state vectors |
| **D1c** | Hash-gated install/verify catches corruption | **Partially valid** — applies to state registry too |
| **D1d** | LoRA registry ports to RWKV | **Superseded** — registry is on state, not LoRA |

---

## Experiment Status

| Experiment | Theory | Status | Blockers |
|---|---|---|---|
| `dendrite_rwkv_001` | Original (LoRA) | **Blocked** — data collation bug (answer spans > max_len) | Fix generator config or max_len |
| `dendrite_rwkv_002` | Original | Not started | — |
| `dendrite_rwkv_003` | Original | Not started | — |
| `dendrite_rwkv_004` | Original | Not started | — |

**Recommendation**: Do not fix `dendrite_rwkv_001`. The original theory is wrong (LoRA ≠ dendrite). Start fresh with corrected theory.

---

## Corrected Theory Status (see `theories/dendrite_growth.status.md` when created)

| Claim | Description | Status |
|---|---|---|
| **G1** | Frozen RWKV trunk + state-vector dendrites + routing + lifecycle → isolated functional memories | Not started |
| **G1a** | State vectors prevent interference (vs LoRA) | Needs `dendrite_state_001` |
| **G1b** | Routing on state vectors works | Needs `dendrite_state_001` |
| **G1c** | Hash-gated state install/verify works | Needs `dendrite_state_001` |
| **G1d** | State vectors ≥ LoRA adapters on all metrics | Needs `dendrite_state_001` |

---

## Mechanism Gaps (Corrected Theory)

- How to distill rule → steady state vector reliably?
- Do state vectors for different rules naturally occupy orthogonal subspaces?
- Can multiple state vectors be composed (`α`-merge) without interference?
- Does RWKV-8-ROSA's suffix automaton state compose with dendrite state?
- How to verify state → behavior equivalence deterministically?

---

## Next Actions

1. **User writes** `theories/dendrite_growth.md` (verbatim corrected theory)
2. **Assistant creates** `theories/dendrite_growth.infer.md` (this interpretation)
3. **Assistant creates** `theories/dendrite_growth.status.md` (empty, awaiting experiments)
4. **Implement** `src/rwkv_growing.py` (trunk + branch architecture)
5. **Run** `experiments/dendrite_state_001` (first proof: G1a)