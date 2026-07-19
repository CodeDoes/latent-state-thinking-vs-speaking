# RWKV: Research Overview

Papers collected from arXiv covering the RWKV architecture family and its derivatives.

---

## Core RWKV

### RWKV: Reinventing RNNs for the Transformer Era
- **arXiv**: [2305.13048](https://arxiv.org/abs/2305.13048)
- **Authors**: Bo Peng, Eric Alcaide, Quentin Anthony, et al.
- **Date**: 2023-05-22
- **Categories**: cs.CL, cs.AI

**Summary**: The original RWKV paper. Proposes a novel architecture combining the best of Transformers (parallelizable training) and RNNs (linear-time inference). RWKV uses a linear attention mechanism with a recurrent state that can be computed in parallel during training. The architecture has four core components: **R**eceptance, **W**eighted, **K**ey, and **V**alue. Time-mixing and channel-mixing blocks replace standard self-attention with a WKV (weighted key-value) operator that maintains a linear-time recurrence.

**Relevance**: This is the foundation of the project's core architecture (`src/rwkv_nano.py`, `core/rwkv.md`). The linear scaling property enables byte-level processing where sequence lengths are much longer than token-level models.

---

### RWKV-6 / Finch
- **arXiv** (via GoldFinch): [2407.12077](https://arxiv.org/abs/2407.12077)
- **Key improvement**: Enhanced state-tracking with improved WKV computation, decay scheduling, and head-specific state mixing.

RWKV-6 (Finch architecture) introduces a data-dependent decay mechanism and improved state passing, addressing earlier limitations in long-context recall.

---

### GoldFinch: RWKV/Transformer Hybrid with Linear Pre-Fill
- **arXiv**: [2407.12077](https://arxiv.org/abs/2407.12077)
- **Authors**: Daniel Goldstein, Fares Obeid, Eric Alcaide, et al.

**Summary**: Stacks a new "GOLD" transformer on top of an enhanced RWKV-6 (Finch) backbone. Achieves extreme KV-cache compression (linear time and space pre-fill) while retaining transformer-like quality. Demonstrates that hybrid linear-attention/transformer architectures can bridge the gap between pure RNN efficiency and transformer quality at 1.5B parameters.

**Relevance**: Hybrid approaches (RWKV + attention) are mentioned in this project's architecture ([`delta-mem.md`](delta-mem.md)) as a potential path. GoldFinch provides evidence that linear-attention + transformer hybrids work at scale.

---

### VisualRWKV
- **arXiv**: [2406.13362](https://arxiv.org/abs/2406.13362)
- **Authors**: Haowen Hou, Peigen Zeng, Fei Ma, et al.

**Summary**: First application of RWKV to multimodal (vision + language) tasks. Uses data-dependent recurrence and sandwich prompts. Shows RWKV's state can be adapted for modalities beyond text.

**Relevance**: Supports the project's assumption that RWKV state can carry multimodal information — relevant for [`byte-state-byte.md`](byte-state-byte.md) where bytes can represent any modality.

---

## RWKV Derivative Work

### Revenge of the Fallen? Recurrent Models Match Transformers at Human Language Comprehension
- **arXiv**: [2404.19178](https://arxiv.org/abs/2404.19178)
- **Authors**: James A. Michaelov, Catherine Arnett, Benjamin K. Bergen

**Summary**: Shows RWKV and Mamba match transformer performance on psycholinguistic benchmarks (N400 and frontal positivity effects). Important result: recurrent models capture human-like language processing despite lacking attention.

**Relevance**: Validates that recurrent architectures (without attention) can perform complex linguistic generalization — supports the project's goal of understanding *when* attention is actually necessary.

---

### DREAMSTATE: Diffusing States and Parameters for Recurrent LLMs
- **arXiv**: [2601.19221](https://arxiv.org/abs/2601.19221)
- **Authors**: Liu Xiao

**Summary**: Explores the RWKV state as an editable knowledge representation. Uses a conditional diffusion model to edit the internal state of RWKV models — treating the state as an explicit memory that can be modified.

**Relevance**: Directly relevant to this project's memory thread ([`dendrite_growth.md`](dendrite_growth.md), [`delta-mem.md`](delta-mem.md)). Shows that RWKV state can be treated as editable knowledge — a key assumption in the dendrite and delta-memory approaches.

---

### DeltaProduct: Improving State-Tracking in Linear RNNs
- **arXiv**: [2502.10297](https://arxiv.org/abs/2502.10297)
- **Authors**: Julien Siems, Timur Carstensen, Arber Zela, et al.

**Summary**: Addresses the expressivity-efficiency tradeoff in linear RNN state matrices. Diagonal matrices (used in Mamba, GLA, mLSTM) are efficient but limited. DeltaProduct uses Householder products for state-transition matrices, achieving better state-tracking while maintaining linear-time inference.

**Relevance**: The state-tracking expressivity problem is exactly what this project hits in [`byte-state-byte.md`](byte-state-byte.md) (decoder stall, mode collapse). DeltaProduct's approach to richer state transitions could inform better encoder/decoder designs.

---

### WuNeng: Hybrid State with Attention
- **arXiv**: [2504.19191](https://arxiv.org/abs/2504.19191)
- **Authors**: Liu Xiao, Li Zhiyuan, Lin Yueyu

**Summary**: Augments RWKV-7 with additional attention heads (from Hymba). Uses RWKV-7 state-driven heads alongside standard multi-head attention to enhance representation without sacrificing KV-cache efficiency.

**Relevance**: Another hybrid approach — directly relevant to [`delta-mem.md`](delta-mem.md) which explores adding RWKV-state memory as a side-channel inside attention models.

---

### RWKV-TS: RWKV for Time Series
- **arXiv**: [2401.09093](https://arxiv.org/abs/2401.09093)

Applies RWKV to time series forecasting. Shows the architecture's generality beyond text.

---

## Key Takeaway for the Project

1. **RWKV state is editable** (DREAMSTATE) — supports the memory/dendrite thread.
2. **RWKV + attention hybrids work** (GoldFinch, WuNeng) — validates the delta-mem approach.
3. **Recurrent state expressivity matters** (DeltaProduct) — the decoder stall problem in `byte-state-byte` may be a state-matrix expressivity issue.
4. **RWKV matches transformers on psycholinguistics** — the "attention is not always needed" thesis holds.

