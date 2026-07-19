# Modular & Dendritic Networks: Research Overview

Papers covering modular neural networks, dendritic computation,
progressive network growth, and adapter-based architectures.

---

## Adapter-Based Modularity (LoRA and variants)

The project's [`dendrite_memory.md`](dendrite_memory.md) originally framed dendrites as LoRA adapters attached to a frozen RWKV backbone. The relevant published work:

### LoRA: Low-Rank Adaptation of Large Language Models
- **arXiv**: [2106.09685](https://arxiv.org/abs/2106.09685) (2021)

The original LoRA paper. Freezes pre-trained weights and injects trainable rank-decomposition matrices into specific layers (typically attention Q/K/V/O projections). A LoRA adapter for a d×d weight matrix uses two small matrices A (d×r) and B (r×d) where r ≪ d, so the update is ΔW = BA with rank r.

**Relevance**: This is the mechanism the project's first dendrite attempt used. Each "memory" was a LoRA adapter attached to RWKV projection layers. The project found LoRA adapters didn't quite match the desired behavior — leading to [`dendrite_growth.md`](dendrite_growth.md) which uses full architectural modules instead of low-rank updates.

### LoRA-Based Memory / Lifecycle Management
- **No specific paper found** for the hash-gated registry lifecycle (install/verify/quarantine) proposed in `dendrite_memory.md`. This appears to be a novel contribution of the project.

---

## Modular / Multi-Branch Architectures

### Pathway Networks (PathNets)
- **arXiv**: [1611.05433](https://arxiv.org/abs/1611.05433) (2016) — GAs for PathNet
- **arXiv**: [1701.08734](https://arxiv.org/abs/1701.08734) (2017) — PathNet evolution

Early work on modular networks where different "paths" through the network are selected for different tasks. Each path activates a subset of modules. Modules can be added incrementally.

**Relevance**: Conceptually similar to dendrites — the trunk has many possible routes (dendrite branches) and routing selects which to use. PathNet evolves the routing; the dendrite project uses NO-OP probing or learned gates.

### Mixture of Experts (MoE)
- **arXiv**: [1701.06538](https://arxiv.org/abs/1701.06538) — Sparsely-gated MoE
- **arXiv**: [2008.07414](https://arxiv.org/abs/2008.07414) — GShard
- **arXiv**: [2101.03961](https://arxiv.org/abs/2101.03961) — Switch Transformer

MoE uses a router to select which subset of "expert" sub-networks process each input. Each expert is typically a feed-forward block. The router is a learned gating function. Training uses load-balancing loss to prevent expert collapse.

**Relevance**: The project's dendrite architecture (trunk + branches + routing) is architecturally a form of Mixture of Experts where experts are attached to a frozen trunk. Key differences: (1) dendrites are physically isolated (install/delete), (2) routing can be NO-OP probing rather than learned gating, (3) each dendrite is trained independently, not jointly.

---

## Progressive Network Growth

### Progressive Neural Networks
- **arXiv**: [1606.04671](https://arxiv.org/abs/1606.04671) (2016) — Progressive Networks

Introduced columns (full networks) that are added when learning new tasks, with lateral connections to previous columns. Old columns are frozen. New columns can use features from old columns via learned adapters.

**Relevance**: Direct ancestor of [`dendrite_growth.md`](dendrite_growth.md). The progressive network adds a full new "column" (new set of layers) per task. The dendrite project adds smaller modules (branches) rather than full columns, but the principle is identical: freeze old, add new, connect laterally.

### Net2Net / Network Morphism
- **arXiv**: [1511.05641](https://arxiv.org/abs/1511.05641) — Net2Net (2015)
- Function-preserving transformations that expand network capacity without changing behavior.

Net2Net allows expanding a network's width or depth while preserving the function it computes. After expansion, training can specialize the new capacity.

**Relevance**: Related to [`progressive-expansion.md`](progressive-expansion.md) — the idea of expanding network capacity without breaking existing capabilities. Net2Net provides the mathematical framework; the project applies it to RWKV specifically.

---

## Dendritic Computation in Neural Networks

### Dendritic Computation in a Single Neuron
- **arXiv**: [2101.10281](https://arxiv.org/abs/2101.10281) — A literature review
- Biological neurons are not point-like; dendrites perform local computation, nonlinear integration, and input segregation.
- Models of dendritic computation show that a single neuron with dendritic structure can solve XOR-like problems.

**Relevance**: The project's dendrite metaphor draws from this biological grounding. The key insight from computational neuroscience: dendrites allow a neuron to segregate inputs and process them independently before integrating — exactly the design goal of the project's dendrite branches.

---

## Key Observations from the Literature

**What's novel about this project's approach:**

1. **Hash-gated lifecycle** — No published work (found) uses cryptographic integrity checks for adapter/module installation. The `install_integrity` and `functional_equivalence` gates appear to be a project innovation.
2. **NO-OP probing for routing** — Using NO-OP branches to probe state outcomes, then selecting the branch that moves state in the desired direction, is not a standard MoE or routing approach. It's conceptually novel.
3. **Frozen trunk + independently-trained branches** — This is similar to Progressive Networks but the branches are much smaller (not full columns), and the routing mechanism (NO-OP probing) is different.

**What the literature suggests:**

1. **Load balancing** — MoE literature shows that routers tend to collapse (always pick the same expert). The project's entropy-regularized gates in `adaptive-exit-entropy.md` address a similar collapse problem. Expect load-balancing loss to be needed if multiple dendrites are active simultaneously.
2. **Adapter interference** — Even with independent adapters, there can be interference at the trunk output (where branches read). The project's `D1a` experiment (adapter isolation) is exactly the right test.
3. **Routing necessity** — `D1b` (NO-OP probing vs. learned routing) tests whether complex routing is actually needed. The MoE literature suggests it is, for more than ~2 experts.

