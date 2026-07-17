# byte-patch-preview.infer.md

> **Source**: `byte-patch-preview.md` (verbatim).  
> **Date**: 2025-07-17

---

## What this theory is doing

A **preview/design note** for a byte-level+patches hybrid architecture, situated *before* the more formalized `byte-state-byte.md` family. The preview asks: can a model ingest byte-level signal with patch-level latency, while staying *cheap*?

It is a **systems-theoretic sketch**, not a measured experiment.

## What the .md assumes vs. declares

| Claim | How stated |
|---|---|
| "Byte-level encoder + patch-level mid-stage is worthwhile" | Inferred from BLT-style literature; the .md is a *design proposal*, not a measurement |
| "Patches need distinct handling from bytes" | By construction |
| Specific timestep definitions, hidden dimensions, etc. | Set conservatively (mimic BLT, smaller) |

## Latent assumptions (filled in here)

- The byte encoder is *sufficient* to compact raw bytes (~256-dim observations) into a fixed-size patch representation
- The patch-level mid-model (RWKV) can predict **next patch** better than it can predict next byte given a patch
- Decoder expands each patch back to multiple bytes coherently
- The factorization `bytes → patches → bytes` is **lossy-but-rich-enough** for downstream tasks

## What I would test first

1. **Lossy compression ceiling.** Does the encoder-decoder round-trip on a hand-curated test set preserve **distinct** outputs across distinct inputs? (sanity test)
2. **Patch-sequence prediction accuracy** vs **byte-sequence prediction accuracy** at matched compute. If patches win by ≥20% on long-range tokens, the architecture is justified.
3. **State accumulation cost.** Verify `state_size ≈ runtime_kbytes` fits constant-memory guarantee of RNNs (doesn't grow with seq length).

## Connection to other theories

| Theory | This .md overlaps |
|---|---|
| `byte-state-byte.md` | Same intuitions; this .md is the *earlier* less-precise sketch |
| `progressive-expansion.md` | "Add a patch stage later" matches the proportional-capability-test design |
| `ultimate_thesis.md` | "Multi-rate" claim U5 directly builds on this .md |

## Status

Preview only. No formal claim proven. Superseded by `byte-state-byte.md` work but kept here as a reasoning snapshot.
