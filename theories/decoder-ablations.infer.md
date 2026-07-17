# decoder-ablations.infer.md

> **Source**: `decoder-ablations.md` (verbatim).  
> **Date**: 2025-07-17

---

## What this theory is doing

This is a **decoder ablation study** that uses commit `317f4a5` as the experiment seed. The minimal ablation: same encoder + patch-slot + data, only `n_dec_layers` varies.

| Arm | Decoder layers | Params | Final loss (ep 3) |
|---|---|---|---|
| Full decoder | 2 | 673,218 | 1.2032 |
| No decoder | 0 | 431,490 | 1.4021 |

Δ = +0.199, **+14.2% worse** without decoder blocks.

## What the inferred addendum (`decoder-ablations.md` "What the ablation does NOT claim") exposes

The original claim — "decoder RWKV blocks are doing real work" — is **not** proven by this ablation. The reason: the model is so small (~673K total) that gained capacity (any extra depth, any architecture) might explain the 14.2%. The ablation does not isolate *decoder-specific contribution* from *depth-specific contribution*.

What it does isolate: **decoder-blocks-have-non-trivial-content** vs **decoder-blocks-are-pure-overhead**. The 14.2% gap says "the depth contains signal." Not "the decoder-specific mechanics."

## Suggested next ablations (from your words)

1. **Decoder with patch context injected per-block** — like cross-attention at every decoder layer, not one-shot residual at the front
2. **Encoder ablation** — same trick but `n_enc_layers=0` to verify encoder layer-by-layer is load-bearing
3. **Patch-slot ablation** — vary `n_patch_layers` 0→3 with everything else matched

The author (you) flagged these as the **full 4-variable grid** needed before any genuine architecture decomposition. Until those run, +14% is just one arm.

## Latent Assumptions (Infer Notes)

- The encoder is *presumably* also load-bearing — but this is presumption, not measured
- "Decoder is the right place to integrate patch context" — not proven; it's where it currently is, the rightness is a hypothesis
- The 14.2% gap assumes **all decoder blocks carry proportional signal** — but some decoder blocks may carry all the signal and others carry none. Per-block ablation is much sharper.

## Status

Experiment done. Full decomposition pending. Three followups are cheap when GPU is available; ranking them:

| Cost | Ablation | What it tests |
|---|---|---|
| cheap | n_patch_layers ∈ {0,1,2,3} | Is patch depth load-bearing? |
| medium | per-block decoder context injection | Is concentrated-fusion vs interleaved-fusion different? |
| medium | n_enc_layers ∈ {0,1,2,3} | Is encoder depth load-bearing? |

Recommend **patch-slot ablation first** (cheapest) before GPU is needed.
