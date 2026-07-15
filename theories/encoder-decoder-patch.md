# encoder-decoder-patch architecture

**Status**: design phase, not yet implemented
**Date**: 2026-07-15
**Inspired by**: BLT (Byte Latent Transformer), but using RNN recurrence instead of attention

## Core idea

Hierarchical byte-level + patch-level architecture using recurrent models (RWKV-style) instead of attention. The architecture takes advantage of RNN's natural sequential processing for byte streams.

## Models

### encoder-model (byte-level)
- **Input**: encoder-state + decoder-state + byte (256-dim one-hot)
- **Output**: encoder-state + trigger
- **Trained on**: next-byte prediction (cross-entropy over 256 bytes)
- **Role**: processes bytes sequentially, accumulates state, fires trigger when patch is complete

### decoder-model (byte-level)
- **Input**: encoder-state + decoder-state + patch
- **Output**: decoder-state + trigger
- **Trained on**: next-byte prediction (cross-entropy over 256 bytes)
- **Role**: reconstructs bytes from the patch, accumulates decoder-state

### patch-model (patch-level)
- **Input**: model-state + patch
- **Output**: model-state + patch
- **Trained on**: next-patch prediction (MSE or similar over 8-float patch vectors)
- **Role**: operates at coarser granularity, transforms patches, accumulates model-state across patches

## State types

- **byte-level-state**: 8 floats (shared between encoder and decoder)
- **patch-level-state**: 8 floats × N patches (used by patch-model)

## Key mechanisms

### Feedback loop
decoder-state feeds back to encoder for the next cycle. This creates continuous recurrence across patch boundaries:
```
encoder ──(patch)──→ patch-model ──(patch)──→ decoder ──(state)──→ encoder ──→ ...
   ↑                                                    │
   └────────────────────────────────────────────────────┘
```

### Trigger mechanism
Patch boundaries are determined by learned triggers, not external entropy models. The encoder learns when to commit a patch (fire trigger) based on its state dynamics.

### Patch = encoder-state
The patch is not a pooled vector — it's the encoder-state at the moment the trigger fires. This is the shared representation that all three models operate on.

## Training phases

### Phase 1: train encoder + decoder
```
byte → encoder (repeat until trigger) → encoder-state
encoder-state → decoder (repeat until trigger) → byte
```
Loss: next-byte cross-entropy. Encoder and decoder learn to agree on the patch representation.

### Phase 2: insert patch-model
```
byte → encoder(frozen) → patch → patch-model → patch → decoder(frozen) → byte
```
Loss: next-byte cross-entropy. Only patch-model parameters are updated. Patch-model must learn to operate in the same patch-space that encoder produces and decoder expects.

## Differences from BLT

| BLT | Our architecture |
|-----|------------------|
| Attention-based | RNN-based (RWKV-style recurrence) |
| Separate entropy model for patch boundaries | Learned trigger mechanism in encoder |
| Pooling from byte-level to patch-level | encoder-state IS the patch (no pooling) |
| No feedback loop | decoder-state feeds back to encoder |
| Patch-level uses global attention | Patch-level uses RNN recurrence |
| Complex (multiple models, attention layers) | Simple (three step-functions, recurrence) |

## Expected result

**Claim**: hierarchical prediction (byte-level + patch-level) enables longer-range prediction than flat byte-level prediction at matched parameter count.

**Reasoning**: patch-level model operates at coarser granularity (patches of 8 bytes). Each step covers 8× more information than byte-level. At same parameter count, patch-level model can maintain coherent state over 8× longer byte ranges.

**Experiment**:
1. Train byte-level model (encoder + decoder, no patch-model), measure prediction error at different horizons
2. Train hierarchical model (encoder + decoder + patch-model), measure prediction error at different horizons
3. Compare: hierarchical model should have lower error at longer horizons (50-byte, 100-byte ahead)
4. Matched parameters, matched training budget, matched data

**Prove one thing**: hierarchical prediction enables longer-range prediction at matched parameter count.

## Dimensionality

- byte-level-state: 8 floats (start simple, adjust if needed)
- patch-level-state: 8 × N patches (N = context window at patch level)
- patch-size: 8 bytes (target)

## Implementation status

Not yet implemented. Design phase.

## Open questions

1. **Trigger mechanism**: how does the encoder decide when to fire? State divergence? Shift-register? Learned threshold? Needs experimentation.
2. **Patch-model loss**: MSE? Cosine? What distance metric in patch-space?
3. **Feedback mechanism**: how does decoder-state feed back to encoder? Concatenation? Addition? Gating?
4. **Context-sensitive surprise**: how does patch-context (from patch-model) inform byte-level surprise? Does encoder receive patch-context as input?

## Why this matters

- **Simpler than BLT**: uses RNN recurrence, no attention, no separate entropy model
- **Takes advantage of RNN**: sequential processing is natural for byte streams, recurrence accumulates state naturally
- **Clear prove-one-thing experiment**: hierarchical prediction vs flat prediction, matched params
- **Small scale**: 8-float state, small models, CPU-trainable
- **Modular**: three simple step-functions, easy to implement and debug
