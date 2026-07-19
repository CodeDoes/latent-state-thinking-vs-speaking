# State Space Models & Mamba: Research Overview

Papers covering the SSM family (Mamba, Mamba-2, variants) that compete with and complement RWKV.

---

## Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **arXiv**: [2312.00752](https://arxiv.org/abs/2312.00752)
- **Authors**: Albert Gu, Tri Dao
- **Date**: 2023-12-01
- **Categories**: cs.LG, cs.AI

**Summary**: The seminal Mamba paper. Introduces selective state space models (SSMs) that achieve transformer-quality performance with linear-time inference. Key innovations:
- **Selection mechanism**: Parameters (Δ, B, C) depend on the input, allowing the model to selectively propagate or forget information.
- **Hardware-aware parallel scan**: Efficient parallel computation during training.
- **Simplified architecture**: No attention, no MLP blocks — just SSM layers with gating.

**Relevance**: Mamba is the primary competitor to RWKV in the linear-time sequence model space. Both achieve O(n) inference, but Mamba uses a different state mechanism (continuous SSM discretization vs. RWKV's discrete WKV recurrence). This project uses RWKV, but the Mamba family provides a reference point for what linear-time models can achieve.

---

## Mamba-2
- **arXiv**: [2405.21060](https://arxiv.org/abs/2405.21060) (2024)
- Key change: Reformulates SSM as a "state space dual" to attention, unifying SSD (state space duality) with structured matrices. 2-8× faster than Mamba-1 while matching quality.

---

## MambaByte
- Already covered in `byte_level_models.md`. The intersection of Mamba and byte-level processing. Proves byte-level SSMs can compete with tokenized transformers.

---

## Selective State-Space Related Work

### Speech-Mamba
- **arXiv**: [2409.18654](https://arxiv.org/abs/2409.18654)
- Long-context speech recognition with selective state spaces. Shows SSMs work on long audio sequences.

### Bi-Mamba+
- **arXiv**: [2404.15772](https://arxiv.org/abs/2404.15772)
- Bidirectional Mamba for time series. Relevant for the project's exploration of bidirectional context in `b3d-rwkv-nano.md`.

### Differential Mamba
- **arXiv**: [2507.06204](https://arxiv.org/abs/2507.06204)
- Proposes differential state updates for Mamba, improving gradient flow and long-range dependency capture. Related to the project's concern about decoder stall in `byte-state-byte.md`.

---

## RWKV vs. Mamba: Key Differences

| Aspect | RWKV | Mamba |
|--------|------|-------|
| State mechanism | Discrete WKV recurrence (attention-like weighting) | Continuous SSM discretization (zero-order hold) |
| State size | Per-head key-value pairs (2 × d_head × n_heads) | Single hidden state per layer (d_model) |
| Training parallelization | Linear attention parallel scan | Selective scan (hardware-aware) |
| Time-mixing | WKV operator + receptance gating | Selective SSM + gating |
| Length extrapolation | Typically strong (recurrent state) | Typically strong (SSM state) |
| Bidirectional | Causal by default | Causal by default; modifications needed |

## Relevance to Project

1. **The project's `byte-state-byte` decoder stall** — MambaByte processes every byte directly and *doesn't* have decoder stall. This suggests the stall is specific to the encoder→patch→decoder topology, not to byte-level processing per se.
2. **Mamba's selection mechanism** — the input-dependent gating (Δ, B, C) is analogous to what the project's adaptive-loop model (`adaptive-exit-entropy.md`) tries to achieve. Mamba shows that selection is crucial for linear-time models to compete with attention.
3. **Bidirectional SSMs** (Bi-Mamba+) — relevant to the B3D-RWKV project which explores pseudo-bidirectional processing via triplet-block layout.

