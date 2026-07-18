# proofs

One line per proven result. Sorted newest first.

Format: `exp_id (commit): <claim under matched conditions>`.

The git_history of this file is itself the proof log — every
supersession, every refinement is a commit on this file.

Latest first. Older entries below; nothing is deleted, only superseded.

## Proven
- `patch_loop_001` (HEAD): patch-level encoder→sparse decoder with trigger cost at 133K params. trigger_cost=0.1: loss 5.76→2.40, rate~0.5; cost=2.0: rate~0.67. Trigger cost modulates sparsity. Proves B7.
- `shared_state_unrolled_feedback_001`: recurrent decoder feedback (feeding previous target embedding to decoder step i > 0) resolves the B3/B4 decoder stall and mode-collapse. Decoder loss drops rapidly to 0.035 in 500 steps, generating non-blank text.
- `rwkv_state_passing_001`: correct initial state (num/den) decay and shape-stable state saving in RWKVBlock ensures perfect recurrence memory across sequence segments and steps, resulting in successful step-by-step decoding.
- `byte_loop_001` (`a00fa4b`): byte-level adaptive encoder↔decoder at 117K params trains to loss 2.00 (from 5.78). Encoder always triggers (0.99), decoder does work (trigger 0.23). No decoder stall. Proves B6.
- `adaptive_loop_001` (`86e3c01`): adaptive-loop encoder→RWKV-core→decoder trains cleanly at 228K params (loss 5.74→0.47 in 2k steps). Encoder loops adapt from 1→3; decoder uses 1 loop. First byte-state-byte variant without decoder stall. See claim B5.
- `adaptive_loop_002` (`5d9a04c`): scale test ~1.2M params (dim=128, 3L each). Loss 5.80→0.33 in 1k steps. Encoder loops adapt 1→2, decoder stays 1. Coordination survives scaling. Reinforces B5.

## Partial
- `prog_exp_001` (`5161ad7`): A1 (capacity pressure is observable) partially proven — `effective_dimensionality` shows dynamic range; saturation/range metrics hit ceiling on this easy/hard gap. See [`progressive-expansion.md`](memory/progressive-expansion.md).

