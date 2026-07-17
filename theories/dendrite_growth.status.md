# dendrite_growth.status

Proof chain for the "RWKV that grows" theory.

## Claims

- **G1** — *A frozen RWKV trunk + new architectural branch (cross-attn/layers/head) trained on its task does not modify the trunk and does not affect other branches' tasks.*
  Status: **not proven** — awaiting implementation `src/rwkv_growing.py` and experiment `dendrite_growth_001`.

- **G1a** — *Branch isolation: trunk logits on probe set within ε of pre-install after branch training.*
  Status: open.

- **G1b** — *Branch independence: training branch A then B gives B the same loss as training B in isolation.*
  Status: open.

- **G1c** — *Branch attachment depth matters (mid-trunk vs post-trunk).*
  Status: open.

- **G1d** — *Two branches active simultaneously compose multiplicatively.*
  Status: open.

## Mechanism Gaps (What We Know We Don't Know)

- Whether trunk hidden states give a useful read-interface for branches
- Whether multiple branches can be trained in parallel without interference
- Whether branch registry (hash-gated install/verify) operates correctly on full modules (not just weights)
- What is the minimal viable branch type that proves the concept

## Follow-Ups (Cheap → Expensive)

1. `dendrite_growth_001`: One cross-attention branch on frozen trunk, isolation (G1a)
2. `dendrite_growth_002`: Attachment depth sweep — layer L/2, L-1 (G1c)
3. `dendrite_growth_003`: Sequential vs joint branch training (G1b)
4. `dendrite_growth_004`: Two branches, router picks one (G1d)
5. `dendrite_growth_005`: Hash-gated branch registry (parity with `dendrite_memory.md` gates)

## Implementation Status

| Component | Status |
|---|---|
| Branch base class | not started |
| Cross-attention branch | not started |
| Residual expansion branch | not started |
| Output-head branch | not started |
| Growth registry (install/verify/delete) | not started |
