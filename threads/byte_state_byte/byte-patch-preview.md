# byte-patch-preview (revised, after inspecting facebookresearch/blt)

## Actual BLT mechanism (from source)

1. **Byte stream input**. No tokenization. Bytes are the smallest unit.
2. **Patcher** (`bytelatent/data/patcher.py`). A separate small language model
   – the *entropy model* – scores each byte's next-token entropy. A byte at
   position *t* starts a new patch when entropy(*t*) crosses a threshold
   (`patch_start_mask_global_and_monotonicity`). Patches are **dynamic** in
   length and bounded above by `patch_size`.
3. **Local encoder** reads the full byte stream with sliding-window attention
   (`attn_bias_type="local_block_causal"`). Output: byte-level hidden states
   `h_encoder`.
4. **Pool** to patch granularity (typically mean-pool inside each patch).
5. **Global transformer** runs over one token per patch, producing patch-level
   hidden states `h_global`.
6. **Local decoder** rewrites bytes one at a time, cross-attending each
   byte to its patch-level representation from `h_global`.

Architecture diagram (`blt-figure.jpg`): byte → local encoder → pool →
global attention → unpool → local decoder → byte logits.

## What this means for RWKV (and *your* channel-decay insight)

BLT separates two things: a *signal* (entropy, "is this byte surprising?")
and an *aggregation* (pool byte states into patch states, attend on patches).

RWKV already has both — just expressed differently:

- **Signal**: per-channel `time_decay` is a learned "how much to remember
  for this channel". Channels that *want* to retain info across long
  spans are exactly the channels carrying high-surprise bits; channels
  with near-instant decay are carrying predictable/redundant bytes.
  – Insight (yours): BLT's entropy predictor and RWKV's `time_decay`
    capture the same thing. Don't train a separate entropy model; read
    it off the decay gates.
- **Aggregation**: RWKV's time-mixing *is* operating at the sequence
  dimension with a hard-coded window (effectively full sequence). The
  BLT local-encoder windowed-attention analog is just shared parameters
  applied locally; with RWKV's residual recurrent state, that locality
  isn't free.

## Minimal RWKV adaptation

Five components, mapped onto existing RWKV primitives:

| BLT piece | RWKV analog | Effort |
|-----------|-------------|--------|
| Patcher (entropy model) | Read `time_decay` from each layer; threshold | small |
| Local encoder | Byte-level RWKV block(s) with short-wind­owed recurrence | medium |
| Pool to patches | Mean/attention-pool inside each patch span | small |
| Global transformer | RWKV time-mixing over patch tokens | small |
| Local decoder | Decode via cross-recurrence from patch reps | medium |

Concrete minimal viable variant:

1. Train so far: `byte_exp_first` (commit `<pending>`) — byte-level RWKV
   with vocab_size=258, dim=64, num_layers=2, 1500 steps on logic-niiah,
   141K params. Loss curves at 2.05–2.40 region, consistent with the
   char-level `prog_exp_001` (which hit 0.71 within the same step count).
   Byte-level trains; we just didn't show evaluation accuracy yet.
2. **First ablation**: prefix a small patcher (a 1-layer byte classifier
   trained to predict next-byte CE) and feed its threshold-binarised
   prediction as `patch_lengths` to the rest of the model. Compare
   predict-byte loss in (a) vanilla byte-RWKV vs (b) byte-RWKV-with-patches.
3. **Second ablation**: swap the global stack out — comparison is global
   attention over patches vs RWKV time-mixing over patches. This isolates
   recurrence-vs-attention at the *patch* level, byte-level is fixed.

If a and b have comparable loss at matched params/steps, byte-patching
adds something. Otherwise BLT-style patching is not strictly required
on this task.

## What this does NOT prove yet

- That it beats char-level RWKV at the same logic-niiah task. (Need to
  match step count and param count exactly; byte-level adds ~258-74 = 184
  extra output-class weights, which is small in d_emb=64 but not zero.)
- That the patcher helps when the byte stream has high-entropy regions
  (e.g. numbers, variable names, code tokens). Logic-niiah has both; a
  splitter would help there if any model could exploit it.
- That channel-decay as entropy readout is a *learned* signal or just
  a re-statement of the loss the model is already minimizing. We need
  a controlled ablation: train a byte-RWKV, *freeze* its decays, and
  show that those frozen decays still serve as a usable patch signal
  for a downstream patcher module.

## Why this matters for the project (not just BLT)

Patching gives us a clean place to apply progressive expansion. A
patch boundary is a known junction in the computation graph. A
*capacity-pressure signal at a patch boundary* is the perfect insertion
point for a new block in the rapid-expansion frame: every patch is its
own scope, so surgically inserting a new block at the right patch is
local surgery, not global surgery.

---

**Research links:** [`research/byte_level_models.md`](../research/byte_level_models.md) — BLT paper analysis, MambaByte, dynamic patching approaches.
