# Theory: Resolving Decoder Stall & Mode Collapse in Shared-State Unrolled

## 1. Executive Summary
The unrolled shared-state models on disk (`SharedStateUnrolled`) exhibit two failure modes:
- **B3 (Decoder Stall)**: Separate weights, decoder loss fails to decrease below $\approx 0.16$.
- **B4 (Mode Collapse)**: Shared weights, the output collapses to a single repeating character (e.g. `eee...`).

This theory identifies the structural flaw behind these issues: **the decoder does not receive autoregressive sequence feedback during training or generation**. Instead, at every step $i$ of the decoder phase, the model is fed the exact same static `transformed_state`.

We propose **Recurrent Decoder Feedback**—an elegant architectural change where the decoder receives the embedding of the previous step's token (the target token during teacher-forcing training, or the predicted token during inference) as its step input. This introduces the required sequence dependency, resolving the decoder stall and enabling true autoregressive text generation.

---

## 2. The Flaw: Static Inputs over Decoder Steps

In `SharedStateUnrolled.forward`, the decoder phase loops for `n_decoder_steps` steps:

```python
for i in range(self.n_decoder_steps):
    if i == 0:
        input_state = transformed_state
    else:
        # Use decoder's internal state (h from last RWKV block)
        # For now, just use transformed_state (could be improved)
        input_state = transformed_state

    logits, decoder_state = self.decoder(input_state, decoder_state)
```

Because `input_state = transformed_state` at every single step:
1. The input to the decoder is completely static.
2. The decoder blocks have no access to the sequence of bytes they are supposedly predicting.
3. The only variable that changes across steps is the internal hidden state of the blocks (which is currently broken anyway due to the `RWKVBlock` bug).
4. Because the input is constant, the model has no standard autoregressive pathway to learn transition statistics (e.g., $P(y_t \mid y_{<t})$). This forces the decoder to either stall (B3) or mode-collapse (B4).

---

## 3. The Solution: Recurrent Decoder Feedback

To fix this, we implement a proper teacher-forced autoregressive decoder pathway:

### Training Mode (Teacher Forcing):
- At step $i = 0$: The input is the prompt context `transformed_state` (which acts as the start-of-generation signal).
- At step $i > 0$: The input is the embedding of the previous target token:
  $$\text{input\_state} = \text{self.encoder.embed}(\text{targets}[:, n\_encoder\_steps + i])$$

This aligns the model with the classic seq2seq framework:
- Encoder encodes the sequence of inputs into a compressed latent state `transformed_state`.
- Decoder acts as an autoregressive LM, conditioned on `transformed_state` at step 0, and receives previous actual targets as step-by-step feedback.

### Inference/Generation Mode:
- At step $i = 0$: The input is `transformed_state`.
- At step $i > 0$: The input is the embedding of the previously predicted token:
  $$\text{input\_state} = \text{self.encoder.embed}(\text{previous\_predicted\_token})$$

---

## 4. Expected Benefits
- **Avoids Mode Collapse (B4)**: Incorporating actual token transition context removes the degeneracy of constant inputs.
- **Resolves Decoder Stall (B3)**: Decoder optimization becomes a standard sequence-prediction problem, allowing loss to drop cleanly towards zero.
- **Enables True Generation**: Generates coherent, non-blank text.
