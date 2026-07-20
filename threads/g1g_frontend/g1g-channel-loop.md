# G1G Channel Loop

Replace the FFN (channel-mix) sublayer in selected frozen RWKV-7 g1g layers
with a trainable loopy byte-level encoder↔decoder.

## Motivation

RWKV-7 has two sublayers per block: **time-mix** (multi-head matrix state
WKV) and **channel-mix** (ReLU² FFN). The channel-mix is the only
token-independent computation — it processes each token's hidden state
through a fixed 4x expansion.

By replacing the channel-mix with a trainable **loopy byte processor**, we
add adaptive compute where the model can "think in bytes" at specific layers:
project the hidden state to N byte-token IDs, run an RNN encoder over them,
hand off to a decoder via adaptive triggering, then project back to the
hidden dimension. All original time-mix weights stay frozen.

This is a different surgical approach from `token-surgery.md` (which replaces
whole layers at the encoder/decoder boundaries). Here we replace the
**internal** FFN of a block while keeping the time-mix intact.

## Architecture

```
Blocks 0..K: time-mix (frozen) + channel-mix (REPLACED by LoopyChannelMix)
Blocks K+1..31: time-mix (frozen) + channel-mix (frozen FFN)
```

For one replaced block's channel-mix:

```
ln2(x) —(2560-dim)→ hidden_to_byte_logits —(Linear)→ (n_bytes × 258)
  → argmax → byte_ids(0..n_bytes-1)
  → ByteEncoderRNN (reads bytes, accumulates state, emits trigger)
  → ByteDecoderRNN (reads encoder output, produces bytes)
  → pool_decode (top RNN state → Linear → 2560-dim)
  → residual add to x
```

### Components

- **`hidden_to_byte_logits`**: Linear(2560, n_bytes × 258). Projects hidden
  state to byte-token logits. During training, uses gumbel-softmax for
  differentiability.
- **`ByteEncoderRNN`**: Embed + 1-2 RNN cells with receptance gating.
  Reads byte IDs, outputs per-step byte predictions and trigger logits.
- **`ByteDecoderRNN`**: Takes encoder's byte logits + trigger signal,
  processes through its own RNN, outputs decoder byte predictions.
- **`byte_to_hidden`**: Linear(byte_dim, 2560). Pools top decoder state
  back to model dimension.

## Claim C1

The frozen g1g backbone's time-mix computation can be preserved while its
FFN at specific layers is replaced with a byte-level loop. Replacing
deeper layers (closer to the head) is harder because the FFN at those
layers carries token-specific output patterns.

## Experiment Design

### Single variable

**Layer index of channel-mix replacement.** Which layer(s) of the 32 get
the loopy byte channel-mix?

### Arms

| Arm | Layers replaced | Trainable params | Hypothesis |
|-----|----------------|-----------------|------------|
| A (first 3) | 0, 1, 2 | ~32M | Earliest layers are "encoder" layers; byte-level FFN helps encode bytes into state |
| B (middle 3) | 14, 15, 16 | ~32M | Middle layers are "thinker" layers; byte-level FFN distracts from state reasoning |
| C (last 3) | 29, 30, 31 | ~32M | Last layers are "decoder" layers; byte-level FFN helps decode state to bytes |
| D (all 6) | 0-5 | ~64M | More capacity, more risk of overfitting on small task |

### Data

`sum_threshold` rule from `rule_generator.py` via `LogicNiiahGenerator`.
Byte-level (258 vocab). Loss masked to answer spans.

### Size

| Component | Params |
|-----------|--------|
| Frozen g1g backbone (32 layers) | 2.6B |
| LoopyChannelMix per layer (n_bytes=16, byte_dim=64) | ~5.3M |
| Total trainable (3 layers) | ~16M |
| Total trainable (6 layers) | ~32M |

Each LoopyChannelMix layer:
- `hidden_to_byte_logits`: 16 × 258 × 2560 = 10.6M
- Encoder: embed(258 × 64) + cell(3 × 64²) + heads(259 × 64) ≈ 33K
- Decoder: input_proj(259 × 64) + cell(3 × 64²) + heads(259 × 64) ≈ 49K
- `byte_to_hidden`: 2560 × 64 = 164K
- Total: ~10.8M / layer (dominated by hidden_to_byte_logits)

## See also

- `src/g1g_channel_loop.py` — module implementation
- `src/train_g1g_channel_loop.py` — training script
- `threads/tokenizer_family/token-surgery.md` — related layer-level surgery approach
- `threads/byte_state_byte/byte-state-byte.md` — the byte-level encoder/decoder idea
