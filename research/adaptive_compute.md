# Adaptive Compute & Early Exit: Research Overview

Papers covering dynamic compute allocation, early-exit architectures,
and input-adaptive processing in neural networks.

---

## DART: Input-Difficulty-Aware Adaptive Threshold for Early-Exit DNNs
- **arXiv**: [2603.12269](https://arxiv.org/abs/2603.12269) (2026)
- **Categories**: cs.AR (Hardware Architecture)

**Summary**: Proposes a method to dynamically set early-exit thresholds based on input difficulty, rather than using a fixed threshold. The threshold adapts per input, increasing confidence requirements for hard examples and decreasing for easy ones.

**Key ideas**:
- Input difficulty estimated from intermediate representations
- Threshold calibrated per-input using a small auxiliary network
- Reduces average compute while maintaining accuracy

**Relevance**: Directly relevant to [`adaptive-exit-entropy.md`](adaptive-exit-entropy.md) in this project. The project currently uses a fixed entropy weight (0.01) to control loop depth. DART suggests that *input-dependent* thresholds could yield better efficiency. The entropy weight sweep planned in `adaptive-exit-entropy.md` could be extended to include input-adaptive thresholds.

---

## RAViT: Resolution-Adaptive Vision Transformer
- **arXiv**: [2602.24159](https://arxiv.org/abs/2602.24159) (2026)
- **Categories**: cs.CV

**Summary**: A vision transformer that dynamically selects per-token resolution based on content. Low-resolution for background/simple regions, high-resolution for detailed regions. Uses a lightweight policy network.

**Relevance**: Although CV-focused, the concept maps to the project's byte-level processing: allocate more compute (loops/patches) to high-entropy byte regions and less to predictable regions. BLT's entropy-guided patching does this at the patch-boundary level; RViT does it at the per-token resolution level.

---

## Input Conditioned Layer Dropping in Speech Foundation Models
- **arXiv**: [2507.07954](https://arxiv.org/abs/2507.07954) (2025)
- **Categories**: cs.SD

**Summary**: Learns to drop layers during inference based on input characteristics. A gating network predicts which layers can be skipped for a given input without quality degradation.

**Relevance**: Similar to adaptive-exit, but at the layer level rather than the loop level. Could inform the [`dendrite_growth.md`](dendrite_growth.md) routing mechanism where branches are conditionally activated.

---

## The Project's Adaptive Loop Approach

The project's [`adaptive-exit-entropy.md`](adaptive-exit-entropy.md) uses:
- **AdaptiveExitGate**: A learned gating mechanism that controls how many encoder/decoder loops to use
- **Entropy regularization**: A weight (default 0.01) that encourages exploration of different loop depths
- **Empirical result**: encoder loops adapt 1→3, decoder stays at 1

**How this relates to published work**:

| Aspect | This project (adaptive-loop-001) | DART | Standard early-exit |
|--------|----------------------------------|------|---------------------|
| What adapts | Number of recurrent loops | Confidence threshold for exiting | Which layer to exit from |
| How it's learned | Entropy-regularized gating | Auxiliary network per input | Confidence from intermediate heads |
| Granularity | Per-token, per-step | Per-input | Per-token |
| Architecture | RWKV-based | Any DNN | Transformer heads |

---

## Key Gaps for the Project

1. **Entropy weight vs. input-dependent threshold**: The fixed entropy weight (0.01) is arbitrary. DART-style input-adaptive thresholds could replace it.
2. **Loop depth vs. layer skipping**: The project adapts loop depth; layer-skipping (like Input Conditioned Layer Dropping) is an alternative approach untested here.
3. **Compute allocation across patches**: BLT uses entropy to decide patch size; this project uses fixed-size patches. Adaptive patch sizes + adaptive loop depth could compound efficiency.

