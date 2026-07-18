# rwkv-state-carry

RWKVBlock bug fix (rwkv_state_passing_001) restored long-term recurrence memory: num/den state now correctly decays and shape-stable across segments.

Hypothesis: **state carry across training segments matters for long-horizon tasks**, specifically WHERE queries in exp001 and logic_niiah long-context reasoning.

Current training: logic_niiah generator reshuffles each batch, state reset to zero each forward. For byte_ts_001 text stream, state is reset per batch (make_batches slices independent chunks without carry). This tests *intra-segment* memory only, not *inter-segment*.

Why carry should help: RWKV's linear recurrence can theoretically remember >10k tokens if num/den not reset. For NIAH-style task (needle in haystack with transforms), answer requires remembering transforms from 100s of steps ago. Zero init loses that.

Why carry might not help at nano scale: dim=64, num_layers=2 small, decay may be too fast (w_ close to 0) for long memory even with correct carry. Also training instability from BPTT across segments (gradient through state).

Minimal test (single variable = initial state):

- Arm A: zero init (existing), state=None each batch, standard
- Arm B: stateful, carry num/den/xx/xx2 from previous batch for same logical sequence, detach after carry (truncate BPTT) but preserve value. Like TBPTT.
- Both: RWKVNano dim=128, 3 layers, vocab logic_niiah (74 tokens), train 2k steps on generator with long context (max_len=256, noise_max=10)
- Metric: exact accuracy on hard WHERE-style final question where needle 200 tokens ago. Win if B accuracy > A +0.1.

Also second arm C: learned initial state (nn.Parameter) vs zero, to test if learnable start helps.

Single variable: initial state strategy.

This directly validates rwkv_state_passing_001 fix is not just correctness but useful.
