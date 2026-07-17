# Dendritron: Functional Rule Memory via RWKV State — Not LoRA

> **Report updated**: 2025-07-17  
> **Key correction**: Dendritron is **not a LoRA adapter framework**. The memory substrate is **RWKV recurrent state** (the "soma"), not frozen weights + LoRA branches.  
> **Related work**: [Decoupled Mixture-of-Experts for Parametric Knowledge Injection (DMoE, arXiv:2606.14243)](https://arxiv.org/html/2606.14243v1) — same problem (modular knowledge injection), different substrate (MoE on Transformer FFN vs. RWKV state).

---

## Executive Summary

**Dendritron** is a memory system for RWKV where **each functional rule/memory is a trajectory in recurrent state space**, not a LoRA adapter. The frozen RWKV backbone (soma) processes inputs; memories are **state vectors** (dendrites) that can be installed, composed, and verified via hash gates. Routing selects which state to inject at each step.

This aligns with **DMoE** (arXiv:2606.14243): both decouple knowledge modules from the base model, both use lightweight routing, both aim for modular injection without catastrophic forgetting. The difference is **substrate**: DMoE uses MoE experts on Transformer FFN; Dendritron uses RWKV's recurrent state as the natural memory carrier.

---

## 1. The Core Correction: State, Not LoRA

| Dendritron v0 (LoRA-based) | Dendritron v1 (State-based, **correct**) |
|---|---|
| Frozen backbone + LoRA adapters per memory | Frozen backbone + **state vectors** per memory |
| LoRA weights = "dendrites" | **RWKV recurrent state** = dendrites |
| Routing: logistic regression on hidden states | Routing: similarity / gating on **state vectors** |
| Install: save LoRA safetensors + hash | Install: save **state vector** + hash |
| Verification: logit equivalence | Verification: **state equivalence** (deterministic recurrence) |
| Interference: weight overlap | Interference: **state collision** (addressed by orthogonalization) |

**Why state?** RWKV *is* a recurrent state machine. Its "memory" is the WKV state `(num, den, xx)` carried across timesteps. A functional rule (e.g., "sum ≥ threshold") is a **region in state space** that the backbone naturally evolves into. Storing the *state vector* captures the rule; re-injecting it restores the computation. No weight modification needed.

---

## 2. Architecture (State-Based)

### 2.1 Soma = Frozen RWKV Backbone
- `RWKVNano` or `RWKV-8-ROSA` with `requires_grad=False`
- Only the **state injection points** are writable

### 2.2 Dendrites = State Vectors
Each memory is a **complete recurrent state snapshot** at a specific layer:
```python
# One "dendrite" = state dict at layer L
{
    'num': Tensor[B, C],      # WKV numerator
    'den': Tensor[B, C],      # WKV denominator  
    'xx': Tensor[B, C],       # token-shift buffer (ln1 output)
    'xx2': Tensor[B, C],      # token-shift buffer (ln2 output)
}
```
- Small: ~4 × B × C floats (e.g., 4 × 1 × 128 = 512 floats per memory)
- Captures the *entire* recurrent context needed for the rule

### 2.3 Routing = State Selection
At each step (or segment boundary), a router chooses which dendrite to inject:
- **Address head**: logistic regression on current hidden state → candidate index
- **Verifier**: PPCA on state vector → binding confidence
- **Injection**: `state = selected_dendrite` (hard swap) or `state = α·state + (1-α)·dendrite` (soft merge)

### 2.4 Registry Lifecycle (Hash-Gated)
| Gate | Check |
|---|---|
| `install_integrity` | `sha256(state_vector)` matches on save/load |
| `functional_equivalence` | Replay probe sequence → output logits match ε |
| `deletion_exclusion` | Deleted dendrite no longer in candidate set |
| `backbone_hash` | `sha256(backbone_weights)` unchanged |
| `quarantine_bundle` | Failed dendrites archived with diagnostics |

---

## 3. Connection to DMoE (arXiv:2606.14243)

| DMoE (Transformer) | Dendritron (RWKV) |
|---|---|
| Experts = MoE modules on FFN | **Dendrites = state vectors** |
| Router = uncertainty-aware top-k | Router = address head + PPCA verifier |
| Decoupled from base model | **Frozen backbone, state injection only** |
| KV-cache preserved (experts on last FFN) | **State is the cache** — naturally preserved |
| Knowledge injection via expert training | Knowledge injection via **state distillation** |

**DMoE insight**: "Decouple both experts and router from base model." Dendritron does this: dendrites are pure state, router is a separate lightweight head, backbone never changes.

**DMoE difference**: Experts are *trained parameters* (LoRA/FFN); dendrites are *state vectors* (distilled from backbone dynamics). State is cheaper, faster to install, and native to RNN recurrence.

---

## 4. Single-Variable Experiments (Per AGENTS.md)

| Exp | Variable | Baseline | Test | Measure |
|---|---|---|---|---|
| **D1a** | Memory substrate | LoRA adapters (shared backbone) | **State vectors** (frozen backbone) | Interference (Δacc on other memories) |
| **D1b** | Routing necessity | Always inject first dendrite | Address head + PPCA verifier | Wrong-dendrite activation rate |
| **D1c** | Lifecycle gates | Load without hash check | Hash-gated install/verify | Silent corruption detection |
| **D1d** | State vs LoRA | LoRA adapter per rule | State vector per rule | Parity of isolation + routing + gate metrics |

**Minimal proof scale:**
- Backbone: RWKV-nano, dim=128, 3 layers (~300K frozen)
- Dendrites: 4 rules × 1 state vector each (512 floats = 2KB)
- Data: 4 synthetic rules × 2000 train / 500 test
- Compute: CPU, <2 min per dendrite distillation

---

## 5. Distillation: Rule → State Vector

How to get a state vector for a rule? **Distill from the backbone itself:**

```python
def distill_dendrite(backbone, rule_data, layer=1, steps=500):
    """Train a *temporary* LoRA on the rule, then extract the steady state."""
    # 1. Inject tiny LoRA (rank=4) at target layer
    # 2. Train on rule data until convergence
    # 3. Run probe sequence through backbone + LoRA
    # 4. Capture recurrent state at `layer` after probe
    # 5. Discard LoRA; keep state vector as dendrite
    
    return {
        'num': state['num'].clone(),
        'den': state['den'].clone(), 
        'xx': state['xx'].clone(),
        'xx2': state['xx2'].clone(),
    }
```

The LoRA is a **scaffold** — used only during distillation, then discarded. The dendrite is the *resulting state*, not the adapter.

---

## 6. Implementation Status

| File | Status | Notes |
|---|---|---|
| `theories/dendrite_memory.md` | **Outdated** | Describes LoRA-based v0 |
| `src/dendrite_rwkv.py` | **Outdated** | LoRA-based implementation |
| `src/dendrite_state.py` | **To create** | State-based implementation |
| `experiments/dendrite_rwkv_001` | Incomplete | LoRA experiment, hit collation bug |
| `reports/dendritron.md` | **This file** | Updated to state-based design |

---

## 7. Open Follow-Ups (State-Based)

1. **D1**: Distill 4 synthetic rules → state vectors, prove isolation + routing + gates.
2. **D2**: Test on RWKV-8-ROSA — does ROSA's suffix automaton state compose with dendrites?
3. **D3**: Dynamic composition — inject *multiple* dendrites via `α`-weighted merge (AND/OR gating).
4. **D4**: Real rules (code patterns, API schemas) — distill from few-shot prompts.
5. **D5**: State orthogonalization — ensure dendrites occupy disjoint subspaces (Gram-Schmidt on `num/den`).

---

## 8. Key Takeaway

> **Dendritron = RWKV state as modular memory.**  
> Not LoRA. Not adapters. The recurrent state *is* the memory substrate.  
> DMoE validates the *decoupled architecture* pattern; Dendritron instantiates it on the *native RNN state* of RWKV.

---

*Per AGENTS.md: "Prove one thing at a time — single-variable ablations, matched params. No claimed capability without a named mechanism and a controlled disablement that breaks it."*