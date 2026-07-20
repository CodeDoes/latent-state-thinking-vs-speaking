# proofs

One line per proven result. Sorted newest first.

Format: `exp_id (commit): <claim under matched conditions>`.

The git_history of this file is itself the proof log — every
supersession, every refinement is a commit on this file.

For published work behind the claims, see [`research/`](../research/) — each active theory links to its literature survey.

Latest first. Older entries below; nothing is deleted, only superseded.

## Proven
- `auto_tokenizer_001`: Per-token byte auto-tokenizer with smooth trigger ramp. Encoder reads fixed-width window (max 80 bytes), outputs trigger ramp 0→1 across token, latent at boundary. Decoder reconstructs bytes from latent. 241K params, 246 st/s. Perfect on single-byte tokens, >90% on 2-4 byte tokens after 1 epoch (471K tokens). See `src/train_auto_tokenizer.py`.
- `loopy_tokenizer_001` (cfca04b): Loopy tokenizer with minGRU accumulator, byte-position encoding, and 32-dim bottleneck head (6,970 params) learns to perfectly emulate TRIE tokenizer: 100% byte-to-token accuracy on 14 texts (93 tokens). Trigger fires at correct byte positions. Enables learned byte→world-token front-end for any frozen model. See `src/train_loopy_tokenizer.py`.
- `rwkv7_surgery_001` (5156f87): layer-aware surgery with real RWKV world tokenizer (65529 vocab). Split 3-layer model as 1 encoder + 1 core + 1 decoder. Byte encoder+decoder init from world-layer weights. 500 steps byte fine-tune: surgery loss 4.75→0.01 vs scratch 5.50→0.01. Modest transfer (Δ=0.75 nats initial) — BPE tokenizer creates bigger distribution shift from bytes than char tokenizer. Reinforces T1 with world tokenizer.
- `token_surgery_full` (f0e6434): token→byte interface surgery preserves 27× head start over from-scratch training. Pre-trained RWKV core (500 steps char-level) copied embed/head weights to byte positions, frozen core, trained only 33K interface params. At step 200: surgery loss 0.084 vs scratch 2.25. Proves T1.
- `patch_loop_001` (HEAD): patch-level encoder→sparse decoder with trigger cost at 133K params. trigger_cost=0.1: loss 5.76→2.40, rate~0.5; cost=2.0: rate~0.67. Trigger cost modulates sparsity. Proves B7.
- `shared_state_unrolled_feedback_001`: recurrent decoder feedback (feeding previous target embedding to decoder step i > 0) resolves the B3/B4 decoder stall and mode-collapse. Decoder loss drops rapidly to 0.035 in 500 steps, generating non-blank text.
- `rwkv_state_passing_001`: correct initial state (num/den) decay and shape-stable state saving in RWKVBlock ensures perfect recurrence memory across sequence segments and steps, resulting in successful step-by-step decoding.
- `byte_loop_001` (`a00fa4b`): byte-level adaptive encoder↔decoder at 117K params trains to loss 2.00 (from 5.78). Encoder always triggers (0.99), decoder does work (trigger 0.23). No decoder stall. Proves B6.
- `adaptive_loop_001` (`86e3c01`): adaptive-loop encoder→RWKV-core→decoder trains cleanly at 228K params (loss 5.74→0.47 in 2k steps). Encoder loops adapt from 1→3; decoder uses 1 loop. First byte-state-byte variant without decoder stall. See claim B5.
- `adaptive_loop_002` (`5d9a04c`): scale test ~1.2M params (dim=128, 3L each). Loss 5.80→0.33 in 1k steps. Encoder loops adapt 1→2, decoder stays 1. Coordination survives scaling. Reinforces B5.

## Partial
- `prog_exp_001` (`5161ad7`): A1 (capacity pressure is observable) partially proven — `effective_dimensionality` shows dynamic range; saturation/range metrics hit ceiling on this easy/hard gap. See [`progressive-expansion.md`](memory/progressive-expansion.md).

- `byte_loop_gpu_001` (1e1e87a): CLI framework works end-to-end on GPU. Loss 0.0172, 116806 params. Verdict: LEARNED.
