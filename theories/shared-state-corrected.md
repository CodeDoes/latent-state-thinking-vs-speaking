# Shared-State Architecture: Corrected Design

## Current Implementation (WRONG)

The current `shared_state_model_v2.py` is a feedforward pipeline:
1. Process entire sequence through byte_model → get byte_state
2. Pool byte_state to single vector → send to patch_model once
3. patch_model outputs corrected_state + direction
4. Decoder predicts one byte

**Problems:**
- No sequential processing
- No actual state sharing between encoder and decoder
- No recurrence or feedback loop
- Just three separate models with data flowing through once

## Correct Architecture

### Components

1. **Byte-level encoder (RWKV)**
   - Processes bytes one at a time
   - Accumulates state via RWKV recurrence
   - Tracks surprise (entropy of next-byte prediction)
   - Output: next-byte logits + updated encoder state

2. **Patch-model**
   - Receives encoder state when surprise > threshold
   - Transforms state: corrected_state + predicted next patch (direction)
   - Acts as a "correction layer" between encoder and decoder

3. **Decoder (RWKV)**
   - Receives corrected_state + direction from patch-model
   - Generates bytes sequentially
   - Accumulates its own state
   - Output: next-byte prediction + updated decoder state

### The Flow

```
Byte ingest → encoder (process byte, update state)
           → encoder surprised? 
              YES → send encoder_state to patch_model
                   → patch_model returns corrected_state + direction
                   → decoder receives corrected_state + direction
                   → decoder generates bytes using its state
                   → decoder_state feeds back to encoder
                   → repeat
              NO → continue ingesting bytes
```

### Key Insight

The encoder and decoder are BOTH RWKV models. They both accumulate state. The patch-model sits between them and transforms the state.

The decoder's state feeds back into the encoder for the next cycle. This creates the recurrent loop.

### State Sharing

- **Encoder state**: accumulated from processing bytes
- **Decoder state**: accumulated from generating bytes
- **Patch-model**: transforms encoder state into corrected_state + direction
- **Feedback**: decoder_state → encoder (for next cycle)

The "shared state" is that both encoder and decoder operate on the same state trajectory, mediated by the patch-model.

## Training

### Training objective
- Next-byte prediction (cross-entropy)
- Both encoder and decoder predict next byte
- Patch-model learns to transform state effectively

### Training procedure
1. Process sequence through encoder (byte-by-byte)
2. When surprise > threshold, trigger patch-model
3. patch-model transforms state
4. Decoder generates bytes using transformed state
5. Compute loss on encoder's byte predictions
6. Compute loss on decoder's byte predictions
7. Backprop through all three components

### Ingestion
- Encoder ingests bytes one at a time
- Decoder generates bytes one at a time
- patch-model is triggered dynamically (when surprised)

### Output
- Encoder outputs next-byte logits (for loss)
- Decoder outputs next-byte prediction (for generation)

## What needs to be fixed

The current implementation needs to be rewritten to:
1. Process bytes sequentially (not all at once)
2. Track surprise dynamically
3. Trigger patch-model based on surprise
4. Implement feedback loop (decoder_state → encoder)
5. Both encoder and decoder should predict bytes (for training)
