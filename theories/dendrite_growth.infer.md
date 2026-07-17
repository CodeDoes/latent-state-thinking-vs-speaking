# Dendrite Growth — RWKV That Grows

> Inferred corrections/additions to `dendrite_memory.md`. Original theory used LoRA. This file is the corrected framing: **architectural extensions** to a frozen core.

---

## The Correction (Your Words)

> *"no i wanted extensions to the network. imagine in the ideal. growing from a core and adding parts as you go. but keeping RWKV as is."*

> *"its an RWKV that grows"*

---

## What "Grows" Means (Verbatim → Mechanism)

| Your Words | Mechanism |
|---|---|
| "extensions to the network" | New architectural modules attached at well-defined interfaces |
| "growing from a core" | Frozen RWKV trunk (layers 0..L-1) + branches accreted outward |
| "adding parts as you go" | Branch-by-branch training, isolated, hash-verified |
| "keeping RWKV as is" | Backbone weights never change; recurrence state untouched |

---

## Why Not LoRA (Your Words + My Inference)

| LoRA-Version | Growth-Version |
|---|---|
| "just a lora" — your critique | Not weight patches within same layers |
| Modifies existing projections | Adds new modules, new layers, new heads |
| Same depth/width always | Depth/width grows with capabilities |
| Hidden inside the architecture | Visible as new computation graphs |

---

## Hypotheses (Provisional, Need Proof)

**G1** — *Adding a new architectural branch to a frozen RWKV trunk enables that branch's task without modifying the trunk or affecting other branches' tasks.*

**G1a** — *Branch isolation: a new branch's parameters do not drift trunk outputs (trunk logits on probe set within ε of pre-install).*

**G1b** — *Branch independence: training branch A first, then branch B, gives B the same loss curve as if only B was trained.*

**G1c** — *Branch attachment depth matters: mid-trunk attachment gives different capability coverage than post-trunk attachment.*

**G1d** — *Branches compose: two branches active simultaneously (AND/OR routing) yield multiplicatively richer outputs.*

---

## Optional Connection to Original Dendrite Memory (From `dendrite_memory.md`)

The original theory used **LoRA** as the branch mechanism. This corrected theory separates:
- The **what** (each capability = branch with own weights)
- The **how** (branches can be LoRA, FFN blocks, cross-attn, output heads, vocab expansions)

LoRA is one possible branch type. Cross-attention, residual blocks, output heads, vocab expansions are others.

---

## Initial Branch Types (Hypothesis for Open Investigation)

1. **Cross-attention branch** — reads trunk hidden states, processes with own layers, outputs
2. **Residual expansion branch** — new RWKV blocks *after* the trunk, own recurrence
3. **Output-head branch** — specialized vocab/projections on top of trunk logits
4. **Vocabulary expansion branch** — new embedding rows concatenated to trunk embed
5. **State-memory branch** — attaches to the trunk's WKV state (num, den, xx), augments it

---

## Open Follow-Ups (Cheap → Expensive)

1. `dendrite_growth_001` — One cross-attention branch on frozen trunk, no interference test (G1a)
2. `dendrite_growth_002` — Branch attachment depth sweep (G1c): layer L/2 vs L−1
3. `dendrite_growth_003` — Sequential vs joint branch training (G1b)
4. `dendrite_growth_004` — Two branches active, router picks one (G1d)
5. `dendrite_growth_005` — Branch registry: hash-gated install/verify/delete (parity with original theory's lifecycle gates)

---

## What Needs to Be Built First

- `src/rwkv_growing.py` — `RWKVTrunk` (frozen) + `Branch` (trainable) + `GrowingRWKV` (composition)
- 3 branch types minimum to test the hypothesis
- One experiment (`dendrite_growth_001`) with single-variable ablation matching AGENTS.md rules

---

## Latent Assumptions (Infer Notes)

- You might mean "grow monolithically" (single model grows forever) vs "compose dynamically" (pick branches at inference). Pick one before designing.
- "Extensions" could mean *parameters* (added weights) or *topology* (added pathways). Architecture-of-extensions = topology. Weight-only = LoRA.
- You said "keeping RWKV as is" — could mean either "RWKV is the trunk" or "RWKV is one of many possible backbones." The first is the simplification.
- The "the word of angle said" reference to DMoE (arXiv:2606.14243) is a parallel architecture (Transformer MoE). Worth noting for cross-arch comparison but not the focus.
