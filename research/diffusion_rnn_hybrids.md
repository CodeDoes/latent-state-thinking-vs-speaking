# Diffusion + RNN Hybrids: Research Overview

Papers combining diffusion/denoising with recurrent neural network architectures.

---

## B3D-RWKV (Triplet-Block Diffusion RWKV)
- **arXiv**: [2605.25969](https://arxiv.org/abs/2605.25969)
- **HF**: [leonardklin/B3D-RWKV](https://huggingface.co/leonardklin/B3D-RWKV)
- **Already cited in**: [`threads/b3d/b3d-rwkv-nano.md`](b3d-rwkv-nano.md)
- **Scale**: 7.2B parameters

**Summary**: Proposes a triplet-block layout for diffusion RWKV. Each logical block of B tokens appears three times:
- b1: masked copy (e.g., 50% masks)
- b2: identical masked copy (loss computed here)
- b3: clean ground-truth copy (refreshes RWKV state for next block)

Because RWKV reads left→right, the hidden state arriving at any masked position of b2 has already seen every *unmasked* token of b1. So b2 gets **pseudo-bidirectional** access while staying strictly causal — no backbone change needed.

Inference is iterative per-block denoising: while masked positions remain, run RWKV, commit positions where top-1 prob > τ, unmask those in b1/b2, repeat. Reports 1.6× speedup vs causal AR at 7.2B scale.

**Relevance**: The project's [`b3d-rwkv-nano.md`](b3d-rwkv-nano.md) proposes a nano-scale replication (228K–1M params) to test whether the mechanism works at minimal scale or is an emergent property of scale. The research question: "does triplet-block diffusion work at 228K params, or only at 7.2B?"

---

## DREAMSTATE: Diffusing States and Parameters for Recurrent LLMs
- **arXiv**: [2601.19221](https://arxiv.org/abs/2601.19221) — also in `rwkv_overview.md`

Uses diffusion to *edit the internal state* of RWKV models, not to generate tokens. A conditional diffusion model predicts state edits that would produce desired outputs. This is a different use of diffusion — state editing rather than token generation.

**Relevance**: If adapted, this could be the routing mechanism for [`dendrite_growth.md`](dendrite_growth.md) — instead of training a separate routing network, a diffusion model on states could determine which dendrite branch to activate.

---

## Recurrent Diffusion for Motion Generation
- **arXiv**: [2406.07169](https://arxiv.org/abs/2406.07169) (2024)
- **Title**: RDM: Recurrent Diffusion Model for Human Motion Generation

Recurrent diffusion — the denoising process itself is recurrent (repeated refinement steps share weights). Related to the project's [diffusion-grid-terminal.md](diffusion-grid-terminal.md) idea of iterative grid denoising.

---

## Recurrent Diffusion for Parameter Generation
- **arXiv**: [2501.11587](https://arxiv.org/abs/2501.11587) (2025)

Uses a diffusion process to generate neural network parameters, with a recurrent structure that captures interdependencies between layers. Novel but not directly relevant.

---

## Recurrent Autoregressive Diffusion: Global Memory Meets Local Attention
- **arXiv**: [2511.12940](https://arxiv.org/abs/2511.12940) (2025)

Combines recurrent global memory with local attention in a diffusion framework. The global memory is a fixed-size recurrent state that captures long-range dependencies while local attention handles fine detail.

**Relevance**: Maps to the project's architecture thread — byte-level encoder produces a state (global memory), patch-level model processes state (attention/recurrence), decoder reconstructs (local detail). The concept of "global memory meets local attention" is almost exactly the design of `byte-state-byte.md`.

---

## Key Takeaway for the Project

1. **B3D-RWKV's mechanism is unproven at small scale** — the project's nano replication test (`b3d-rwkv-nano.md`) is the right approach to determine if the mechanism is architectural or scale-dependent.
2. **Diffusion for state editing** (DREAMSTATE) opens a third use of diffusion (beyond generation and denoising) — editing the RWKV state directly. This could be the key to making the dendrite routing trainable.
3. **Global memory + local diffusion** (Recurrent Autoregressive Diffusion) matches the byte-state-byte two-scale design. The research question is whether the "local" step should be attention (as in the paper) or RWKV recurrence (as in this project).

