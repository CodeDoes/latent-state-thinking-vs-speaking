# byte-state-byte

A family of architectures that map raw bytes to bytes via two scales:
a fast byte-level encoder/decoder and a slow patch-level middle model.
The "state" is the encoder-side hidden state and the patch-level model's
recurrent state; the "byte" latents are the bytes entering/exiting.

## Two parallel attempts

| Stack | Models | What kind of recurrence | Status |
|-------|--------|-------------------------|--------|
| step-function (GRU cells) | `src/rnn_patch_model.py`, `src/encoder_patcher_decoder.py`, `src/surprise_patcher.py`, `src/train_rnn_patch.py`, `src/train_encoder_patcher_decoder.py` | hand-rolled step functions; explicit `patch_lengths` from a learned entropy surrogate | phase-2 ablation in `rnn_patch_002` worsened loss; encoder_patcher_decoder variants tracked in commits `276b72c`, `6395f33`, `4abe185` |
| RWKV-based | `src/shared_state_model.py`, `src/shared_state_model_v2.py`, `src/shared_state_unrolled.py`, `src/encoder_state_ablation.py`, `src/train_shared_state*.py` | per-step RWKV blocks; separate encoder / patch / decoder weights (`unrolled`) or shared (`shared`) | `shared_state_unrolled_010` trains cleanly (loss 11.4 → 0.19) but sample is blank — decoder half stuck. `shared_state_unrolled_shared_010` mode-collapses to `eee...` |
| controlled ablation | `src/encoder_state_ablation.py` (run as `encoder_state_ablation_001`, commit `c64f9cf`) | 7-arms, 6568 params, varying state mutability/staticity | real result: see proof in `theories/proofs.md` |

## Three regimes of recurrence shape
Regardless of which attempt, every model here is one of three shapes:

1. **Step-function** — explicit `patch_lengths` from a thresholded
   entropy surrogate. Differentiable only via straight-through or
   curriculum. Tends to diverge in phase 2.
2. **Unrolled RWKV** — `n_encoder_steps` independent encoder
   instances, one patch, `n_decoder_steps` independent decoder
   instances. Encoder/decoder have separate weights. Trains but the
   decoder half doesn't progress beyond ≈ 0.16.
3. **Shared-weights unrolled** — same as (2) but encoder and
   decoder share weights. Mode collapses.

## Design questions we never resolved
1. *From the user/AI commentary in commit messages*: should the byte
   encoder be the *same* network as the decoder (weight-tied via
   transpose), separate, or shared but separate positions?
2. Should the patch-level model be RWKV-style recurrence or a
   one-shot transform (the `transformed_state` in shared_state_v2)?
3. Where does the patch boundary come from? Fixed stride (current
   `patch_size=8`), learned gate (encoder_patcher_decoder), entropy
   surrogate (suggested in `byte-patch-preview.md`), or accumulated
   state threshold?

## Source reading
- [`theories/byte-patch-preview.md`](byte-patch-preview.md) — BLT
  compared to RWKV primitives; only surviving standalone doc.
- [`theories/decoder-ablations.md`](decoder-ablations.md) — report
  on a real controlled ablation (one arm of a 4-variable grid).
- `theories/archive/encoder-decoder-patch.md` — original 3-step-fn
  design. Lands as `src/rnn_patch_model.py`.
- `theories/archive/tiny-blt-rwkv.md` — V0/V1 proposal that proposed
  byte-only-then-byte-with-patcher sequence; the V0 ran, the V1
  didn't reach a clean result.

---

**Research links:** [`research/byte_level_models.md`](../research/byte_level_models.md) — BLT (Meta), MambaByte, ByteFlow, Charformer. See also [`research/state_space_models_mamba.md`](../research/state_space_models_mamba.md) for MambaByte's linear-time byte processing.
