> **Archived 2026-07-20:** superseded by [`../memory/dendrite_growth.md`](../memory/dendrite_growth.md).

# Dendrite Memory Registry for RWKV

**Theory: A frozen RWKV backbone with independent, auditable LoRA "dendrite" branches that can be installed, deleted, and verified without corrupting the core model.**

---

## Core Claim

**D1** — *A frozen RWKV backbone + independent LoRA adapter branches (one per functional memory) + autonomous routing + physical install/delete + hash-gated verification yields a system where each adapter is a causally isolated "memory" that can be audited, quarantined, and reinstalled without affecting the backbone or other adapters.*

**Mechanism**: The dendrite metaphor maps directly:
- **Soma** = frozen RWKV backbone (weights never change after pretraining)
- **Dendritic branches** = independent LoRA adapters, each trained on a single rule/task
- **Synaptic routing** = logistic regression address heads + PPCA verifiers on hidden states
- **Physical lifecycle** = install → evaluate → quarantine/retain → delete/reinstall with hash proof

**Necessity proof**: Removing the adapter isolation (shared backbone fine-tune) causes catastrophic interference between memories. Removing the routing/verification causes wrong-adapter activation. Removing the physical lifecycle gates allows silent corruption on reload.

---

## Architecture

### 1. Backbone (Frozen)
- RWKV-nano (this repo's `rwkv_nano.py`) or RWKV-8-ROSA
- `requires_grad=False` for all backbone params
- Only LoRA injection points are trainable

### 2. Adapter Branches (Dendrites)
- One `LoRAConfig` per functional memory (e.g., `sum_threshold`, `vowel_majority`, `endpoint_match`, `count_trigger`)
- Targets: `key`, `value`, `receptance`, `output`, `fc_key`, `fc_value`, `fc_receptance` — the RWKV projection names
- Rank `r=8`, alpha=16 (configurable)
- Saved as physical files: `adapter_config.json` + `adapter_model.safetensors`

### 3. Routing (Address Proposal + Verification)
- **Address head**: Logistic regression on pooled hidden states → proposes candidate adapter
- **Verifier**: PPCA (label-conditional Gaussian) on same pooled states → binds evidence to binary decision
- Both trained on backbone hidden states (tap at ~35% depth)

### 4. Registry Lifecycle Gates (Ported from Dendritron v0.4.2)
| Gate | Check |
|------|-------|
| `install_integrity` | `sha256(adapter_dir)` before/after load matches |
| `functional_equivalence` | Reload adapter → logits on probe set match within ε |
| `deletion_exclusion` | Deleted adapter no longer appears in candidate set |
| `backbone_hash` | `sha256(backbone_weights)` unchanged after any adapter op |
| `quarantine_bundle` | Failed adapters zipped with `registration_diagnostics.json` |

---

## Single-Variable Experiments (Per Ultimate.md)

| Exp | Variable | Baseline | Test | Measure |
|-----|----------|----------|------|---------|
| **D1a** | Adapter isolation | Shared backbone fine-tune (all memories) | Independent LoRA per memory | Interference (Δacc on other memories) |
| **D1b** | Routing necessity | Always use first adapter | Address head + verifier | Wrong-adapter activation rate |
| **D1c** | Lifecycle gates | Load adapter without hash check | With `install_integrity` + `functional_equivalence` | Silent corruption detection |
| **D1d** | RWKV vs Transformer backbone | Same adapter logic on SmolLM2 | Same on RWKV-nano | Parity of isolation/routing metrics |

---

## Minimal Proof Scale

- **Backbone**: RWKV-nano, dim=128, 3 layers (~300K params frozen)
- **Adapters**: 4 adapters × 8k params = 32k trainable
- **Data**: 4 synthetic rules × 2000 train / 500 test each (TinyStories byte-level or logic-NIIAH)
- **Compute**: CPU, <5 min per adapter train
- **Total params**: ~332k (backbone frozen, only adapters train)

This is the smallest system that sustains the proof — every component is load-bearing.

---

## What This Is Not

- Not a "memory-augmented LLM" with external storage
- Not RWKV-ROSA's suffix automaton (that's context retrieval; this is functional rule memory)
- Not a PEFT survey — this is a registry engineering framework with specific gates

---

## Open Follow-Ups

1. **Dynamic adapter composition** — multiple adapters active simultaneously (AND/OR gating)
2. **Adapter versioning** — semantic diff between adapter versions, not just hash
3. **Cross-architecture registry** — same adapter interface, backbone swapped (RWKV ↔ Transformer)
4. **Real-world rules** — replace synthetic tasks with code patterns, API schemas, user preferences

---

**Research links:** [`research/modular_dendritic_networks.md`](../research/modular_dendritic_networks.md) — LoRA, PathNet, MoE, Progressive Networks, Net2Net. See also [`research/memory_associative.md`](../research/memory_associative.md) for complementary learning systems and DREAMSTATE (RWKV state editing).