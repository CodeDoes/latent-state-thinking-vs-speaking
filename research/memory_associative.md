# Memory & Associative Memory: Research Overview

Papers covering delta-rule associative memory, linear attention memory,
and complementary learning systems in neural networks.

---

## Delta Rule in Linear Transformers
- **arXiv**: [2406.06484](https://arxiv.org/abs/2406.06484)
- **Title**: Parallelizing Linear Transformers with the Delta Rule over Sequence Length
- **Authors**: Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim
- **Date**: 2024-06-10
- **Categories**: cs.LG, cs.CL

**Summary**: Traditional linear transformers (e.g., Linear Attention, Linformer) use an additive update rule for their recurrent state: `S_t = S_{t-1} + v_t ⊗ k_t`. This is simple but limited for associative recall. The delta rule replaces the additive update with: `S_t = S_{t-1} + v_t ⊗ (k_t - S_{t-1}^T u_t)` — essentially, it *erases* the old value associated with a similar key before *writing* the new one. This is exactly the DeltaNet architecture.

Key contributions:
- Shows the delta rule improves associative recall over additive linear transformers
- Provides a parallel formulation (chunk-wise computation) for training efficiency
- DeltaNet achieves state-of-the-art among linear-time models on synthetic recall tasks

**Relevance**: Directly relevant to [`delta-mem.md`](delta-mem.md) in this project, which implements a DeltaRuleStateMemory inside transformer attention. The project's implementation reads-before-writes and injects delta corrections to Q/K/V/O projections — closely related to DeltaNet but applied as a memory adapter rather than replacing the entire attention mechanism.

---

## HOLA: A Hippocampus for Linear Attention
- **arXiv**: [2607.02303](https://arxiv.org/abs/2607.02303)
- **Title**: A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets
- **Authors**: Wanyun Cui
- **Date**: 2026-07-02
- **Categories**: cs.AI

**Summary**: Addresses the fundamental limitation of linear attention: the recurrent state is lossy (when many key-value associations compete, earlier facts are overwritten). HOLA adds a hippocampal complement — an exact bounded KV cache that stores recent/precise memories, with the delta-rule state acting as a compressive memory.

Key contributions:
- Inspired by Complementary Learning Systems (hippocampus ↔ neocortex in the brain)
- Bounded exact KV cache (hippocampus) + delta-rule state (neocortex)
- Semi-parametric: queries can retrieve from either system
- Improves needle recall in long contexts while maintaining O(n) time

**Relevance**: Directly relevant to this project's memory thread:
- [`delta-mem.md`](delta-mem.md) — HOLA validates the delta-rule approach and extends it with an exact cache, which the project's RWKVStateMemory could similarly augment.
- [`dendrite_growth.md`](dendrite_growth.md) — The complementary learning system framing (separate fast and slow memory) maps directly to the trunk+branches metaphor.
- [`rwkv-state-carry.md`](rwkv-state-carry.md) — The issue of state loss over long contexts is exactly what the project's RWKVBlock fix addressed; HOLA suggests an alternative: keep a small exact cache alongside the recurrent state.

---

## Simple Linear Attention: Recall-Throughput Tradeoff
- **arXiv**: [2402.18668](https://arxiv.org/abs/2402.18668) (2024)

**Summary**: Systematically analyzes the recall-throughput tradeoff in linear attention models. Shows that simple linear attention (e.g., the additive rule) has good throughput but poor recall; the delta rule improves recall at modest throughput cost.

**Relevance**: Supports the project's [`delta-mem.md`](delta-mem.md) choice of delta-rule over simple additive memory. The tradeoff is real and measured.

---

## HRM-RWKV-Text (GitHub)
- **Repo**: [xiaol/HRM-RWKV-Text](https://github.com/xiaol/HRM-RWKV-Text)
- **Referenced in**: [`delta-mem.md`](delta-mem.md)

**Summary**: Introduces online stateful adapters (delta-rule associative memory) inside causal attention layers. Rather than fully replacing attention with recurrence (which can cause distribution shift), HRM adds small trainable recurrent side-channels. This is the direct inspiration for the project's delta-mem implementation.

The key insight: instead of swapping Transformer blocks for RNN blocks (which degrades reasoning), add *small* RNN adapters that operate alongside attention. The delta-rule adapter learns to update and read from an associative state without changing the core attention computation.

---

## DREAMSTATE (RWKV State Editing)
- **arXiv**: [2601.19221](https://arxiv.org/abs/2601.19221) — already covered in `rwkv_overview.md`
- Uses diffusion to edit RWKV's internal state

Shows that RWKV state can be treated as editable knowledge — the state is not just a computational convenience but a structured memory that can be read and modified. Relevant for all memory-related theories in this project.

---

## Dense Associative Memory / Modern Hopfield Networks
Related line of work (not directly queried but referenced in the project):
- **arXiv**: [2008.06969](https://arxiv.org/abs/2008.06969) — Dense Hopfield networks for attention
- **arXiv**: [2106.04882](https://arxiv.org/abs/2106.04882) — Learning with associative memories

These provide the theoretical grounding for why delta-rule updates (which resemble Hopfield energy minimization) improve memory capacity over simple additive updates.

---

## Key Takeaway for the Project

1. **Delta-rule is proven** (DeltaNet, HOLA) — the project's delta-mem choice is validated by published results.
2. **HOLA's exact-cache complement** is a natural extension — the project could add a small bounded KV cache alongside the delta-rule state to handle needle-in-haystack tasks.
3. **The complementary learning systems framing** (HOLA) provides a theoretical framework for the trunk+dendrite architecture: fast (exact, hippocampal) vs. slow (compressive, neocortical) memory.
4. **State editing** (DREAMSTATE) confirms RWKV state is tractable for explicit memory operations — supporting the dendrite registry and state-based routing ideas.

