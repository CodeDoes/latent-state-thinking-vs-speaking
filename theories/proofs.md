# proofs

One line per proven result. Sorted newest first.

Format: `exp_id (commit): <claim under matched conditions>`.

The git_history of this file is itself the proof log — every
supersession, every refinement is a commit on this file.

Latest first. Older entries below; nothing is deleted, only superseded.

## Proven
- `adaptive_loop_001` (`86e3c01`): adaptive-loop encoder→RWKV-core→decoder trains cleanly at 228K params (loss 5.74→0.47 in 2k steps). Encoder loops adapt from 1→3; decoder uses 1 loop. First byte-state-byte variant without decoder stall. See claim B5.
- `exp001` (`bb44c04`): recurrent latent state > per-query re-encode at matched params (~7.6k, WHOLE accuracy 0.336 vs 0.007).
- `encoder_state_ablation_001` (`c64f9cf`): encoder-state components load-bearing; static (not mutable) state, with patch+encoder composition, beats 6 other variants at matched ~6.5k params. Ranked full table in [`byte-state-byte.status.md`](byte-state-byte.status.md). See claim B1.
- `rnn_patch_002` (`faadfce`): phase-2 patch model *worsens* loss at matched ~6.5k params (2.83 → 3.33), so this step-function stack's hierarchical-hypothesis argument fails on this task. See claim B2.
- `shared_state_unrolled_010` (`4038849`): encoder half trains cleanly but decoder half stalls (loss 0.156). See claim B3.
- `shared_state_unrolled_shared_010` (`cc5ccea`): shared encoder-decoder weights mode-collapse to `eee...` (loss 2.49). See claim B4.

## Partial
- `prog_exp_001` (`5161ad7`): A1 (capacity pressure is observable) partially proven — `effective_dimensionality` shows dynamic range; saturation/range metrics hit ceiling on this easy/hard gap. See [`progressive-expansion.status.md`](progressive-expansion.status.md).

