# byte-state-byte.status

Proof chain for the byte→state→byte architecture family. Linked
experiment proofs in [`theories/proofs.md`](proofs.md).

## Claims

- **B1** — *encoder-state components matter more than state
  mutability.* A controlled 7-arms ablation at ~6.5K params
  ranks static patch+encoder state as best (loss 3.10) and
  mutable-full-state as worst (loss 3.94). The order:
  `static_patch+encoder` (3.10) < `encoder` (3.12) ≈
  `static_patch+byte_level` (3.35) < `byte_level` (3.57) <
  2× tier at 3.84 / 3.94. Means: encoder-side state is load
  bearing; adding raw byte-level graph state or making state
  mutable during forward does not help, it hurts.
  Status: **proven** by `encoder_state_ablation_001` (`c64f9cf`).

- **B2** — *step-function phase-2 patch model does not help.*
  At matched ~6.5K params, after Phase 1 (encoder-decoder only)
  reaches loss 2.83, adding a patch-model in Phase 2 *with both
  halves frozen* moves loss to ~3.3 — the through-patch
  autoregression degrades rather than improving.
  Status: **proven negatively** by `rnn_patch_002` (`faadfce`).
  Mechanism for the divergence is not isolated — could be (a)
  the patch model is too small, (b) the frozen decoder was
  already converged on this scale, (c) something specific to
  the phase-2 path.

- **B3** — *unrolled encoder/decoder trains; decoder half
  stalls.* At 1.64M params with separate weights, encoder loss
  reaches 0.035 in 1k steps but decoder loss sticks at 0.156.
  Samples come out blank.
  Status: **proven** by `shared_state_unrolled_010` (`4038849`).
  The encoder/decoder asymmetry is the open question.

- **B4** — *shared encoder-decoder weights collapse.* Same
  arch as B3 with weight sharing. Loss plateau at 2.49 with
  sample mode-collapsed to `eee...`.
  Status: **proven negatively** by
  `shared_state_unrolled_shared_010` (`cc5ccea`).

- **B5** — *adaptive-loop encoder→RWKV-core→decoder avoids decoder stall.* At 228K params (encoder 2L+adaptive loops, RWKV-7 core 2L+2 depth loops, decoder 2L+adaptive loops), loss drops from 5.74 to 0.47 in 2k steps. Encoder loops adapt 1→3; decoder uses 1 loop. No mode collapse. First byte-state-byte variant without B3/B4 failure modes.
  Status: **proven** by `adaptive_loop_001` (`86e3c01`).

## Mechanism gap (what we know we don't know)
- Why decoder stalls when encoder doesn't (B3).
- Whether the step-function phase-2 negative (B2) is a property
  of the *patch model* or of training-the-frozen-decoder (i.e.,
  could a much bigger patch model undo the loss).
- Whether the entropy-surrogate from `surprise_patcher.py`
  ever matters in practice (none of the recorded runs use it
  as the patch signal).

## Open follow-ups (in cheap-to-expensive order)
1. Add decoder-only loss scaling or longer schedule to the
   `shared_state_unrolled` series to test whether B3's decoder
   stall is *real* or is just slow on the existing schedule.
   Cheap: rerun existing script for 5k steps.
2. Inside `byte-state-byte` shape (2), add the patch-from-encoder
   signal to the decoder at every step (not just initial state).
   Tests one variable: injection frequency.
3. Try `encoder_state_ablation_001`'s static-patch+encoder arm
   at larger scales (>=100K) to see if it survives scaling.
4. Compare unrolled vs *recurrent* encoder/decoder (current
   shared_state scripts are unrolled; never tried an actual
   recurrent WKV-encoder sequence).
5. Scale adaptive-loop model (B5) to >=1M params to test if
   encoder-decoder coordination survives scaling, and whether
   decoder starts using more loops.
6. Test dynamic patching (B5 used fixed patches) vs fixed
   patching at scale — does surprise-based variable patching
   add value when core has capacity to use it?
