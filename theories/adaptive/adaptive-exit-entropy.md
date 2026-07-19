# adaptive-exit-entropy

B5 adaptive_loop_001 uses AdaptiveExitGate with entropy regularization weight 0.01 to encourage exploration of loop depths. Metrics show encoder loops adapt 1→3, decoder stays 1.

Hypothesis: entropy weight controls loop utilization collapse. Too low → gate collapses to single depth (greedy, no exploration). Too high → prevents convergence (model keeps trying random depths).

Why 0.01 might be arbitrary: set in train_adaptive_loop.py without sweep. Need to prove its load-bearing.

Minimal test (single variable = entropy_weight):

- Same AdaptiveLoopModel dim=64, enc_max_loops=3, core_depth_loops=2, dec_max_loops=3, 228K params, 2k steps
- Sweep entropy_weight = 0.0, 0.001, 0.01, 0.05, 0.1
- Log: enc_loops, core_loops, dec_loops distribution (hist), final loss, gate mean lambda

Prediction:
- w=0.0: loops collapse to 1 (no incentive to explore), loss slightly higher because model doesn't use extra compute when needed
- w=0.01: balanced, enc 1→3 as in B5
- w=0.1: high entropy, loops stay high (3), slower convergence, loss higher initially but maybe better eventually

Proven if there exists w* where loop utilization non-trivial (at least 2 depths used >20% each) and loss minimal.

Single variable: entropy regularization weight.

This cheaply validates adaptive compute mechanism from B5.

---

**Research links:** [`research/adaptive_compute.md`](../research/adaptive_compute.md) — DART (input-adaptive thresholds), RAViT, Input-Conditioned Layer Dropping.

