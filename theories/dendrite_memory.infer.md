# Dendrite Memory — Inferred Interpretation

> **Source**: `theories/dendrite_memory.md` (verbatim theory)  
> **Interpreter**: Assistant  
> **Date**: 2025-07-17  
> **Status**: **Superseded by architectural growth theory** — see `theories/dendrite_growth.md` (to be written). This infer reflects the *original* LoRA-based theory; the corrected theory uses **RWKV recurrent state as the memory substrate**, not LoRA adapters.

---

## Original Theory Summary (from `dendrite_memory.md`)

| Component | Original Proposal |
|---|---|
| **Soma** | Frozen RWKV backbone |
| **Dendrites** | Independent LoRA adapters (one per functional rule) |
| **Routing** | Logistic regression address head + PPCA verifier on hidden states |
| **Lifecycle** | Hash-gated install/verify/delete/reinstall of LoRA weights |
| **Isolation** | LoRA weights don't touch backbone or each other |

**Core claim (D1)**: Frozen backbone + independent LoRA + routing + lifecycle gates = causally isolated, auditable functional memories.

---

## Critical Gaps in Original Theory (My Interpretation

the Original Theory (Latent Assumptions Exposed)

1. **LoRA ≠ dendrite**. The original theory treats LoRA as the "dendritic branch." But a dendrite in biology is a *conductive pathway* that receives and integrates signals — it's part of the neuron's *computational structure*, not a detachable weight patch. LoRA is a *training artifact*, not a runtime structure.

2. **Memory ≠ weights**. The original theory equates "functional memory" with "LoRA adapter weights." But RWKV *is* a recurrent state machine. Its memory is the **recurrent state** `(num, den, xx, xx2)` carried across timesteps. A rule like "sum ≥ threshold" is encoded in the *trajectory through state space*, not in static adapter weights.

3. **Routing on hidden states is indirect**. The original theory routes on pooled hidden states at ~35% depth. But the *actual* memory content is in the recurrent state. Routing should operate on state vectors directly.

4. **Lifecycle on weights is wrong granularity**. Install/verify/delete on LoRA safetensors misses the point. The lifecycle should be on **state vectors**: install a state trajectory, verify it produces the rule behavior, delete the state vector.

5. **Isolation mechanism is wrong**. LoRA isolation = weight disjointness. True isolation = **state orthogonality** — dendrites occupy disjoint subspaces of the recurrent state space so their dynamics don't interfere.

---

## Corrected Theory (Architectural Growth)

| Original | Corrected |
|---|---|
| **Dendrite = LoRA adapter** | **Dendrite = state vector trajectory** (a region in recurrent state space) |
| **Install LoRA** | **Distill rule → steady state vector → store state** |
| **Route on hidden states** | **Route on state vectors** (address head + PPCA on `num/den/xx`) |
| **Verify logit equivalence** | **Verify state → behavior equivalence** (replay from state) |
| **Isolation = weight disjointness** | **Isolation = state subspace orthogonality** (Gram-Schmidt on `num/den`) |
| **Lifecycle on files** | **Lifecycle on state registry** (state vectors + metadata) |

**Key insight**: RWKV's recurrent state *is* the memory substrate. A "dendrite" is a **distilled state trajectory** that the backbone naturally evolves into when processing inputs matching the rule. The backbone is the *soma* (fixed dynamics); dendrites are *attractor basins* in state space.

---

## Minimal Proof Scale (Corrected)

| Component | Spec |
|---|---|
| **Backbone** | RWKV-nano, dim=128, 3 layers (~300K frozen) |
| **Dendrites** | 4 rules × 1 state vector each (4 × 4 × 128 = 2KB) |
| **Distillation** | Train tiny LoRA (rank=4) on rule → run probe → capture steady state → discard LoRA |
| **Routing** | Address head (logistic) + PPCA verifier on state vectors |
| **Data** | 4 synthetic rules × 2000 train / 500 test |
| **Compute** | CPU, <2 min per rule distillation |

---

## Revised Experiment Design (D1a–D1d)

| Exp | Variable | Baseline | Test | Measure |
|---|---|---|---|---|
| **D1a** | Memory substrate | LoRA adapters | **State vectors** | Interference (Δacc on other rules) |
| **D1b** | Routing | Always first | Address + PPCA on **state** | Wrong-dendrite activation |
| **D1c** | Lifecycle | Load without check | Hash-gated **state** install/verify | Silent corruption |
| **D1d** | Growth vs LoRA | LoRA adapter per rule | **State vector** per rule | Parity of all metrics |

---

## Follow-Ups (Corrected)

1. **State orthogonalization** — Gram-Schmidt on `num/den` subspaces so dendrites don't collide
2. **Dynamic composition** — merge state vectors via `α`-weighted combination (AND/OR gating)
3. **ROSA integration** — RWKV-8-ROSA's suffix automaton state + dendrite state = dual memory
4. **Real rules** — distill from few-shot prompts instead of synthetic generators
5. **Cross-architecture** — same state-registry interface, Transformer backbone with KV-cache as "state"

---

## What This Infer Adds Beyond the Original Theory

| Original Theory Leaves Out | Infer Fills In |
|---|---|
| Why LoRA? | It was a scaffold for distillation; the *result* is state |
| What is "memory"? | Recurrent state trajectory, not static weights |
| How does routing work? | On state vectors, not hidden activations |
| What is isolation? | State subspace disjointness, not weight file separation |
| How to verify? | Replay from state → check behavior, not logit diff |
| Why RWKV specifically? | Because its state *is* its memory; no KV-cache dualism |

---

## Next Step

**User needs to write `theories/dendrite_growth.md`** — the verbatim corrected theory. This infer file documents the *gap* between original and corrected. The experiment folder `experiments/dendrite_rwkv_001` tests the *original* (LoRA) theory and is currently blocked on a data collation bug. A new experiment `experiments/dendrite_state_001` will test the corrected theory.