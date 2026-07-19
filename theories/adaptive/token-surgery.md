# Token Surgery

Replace the token embedding/output layers of a pre-trained RWKV with
byte-level counterparts, freeze the core blocks, and measure knowledge
transfer.

## Motivation

The user wants RWKV-7's *training* (the learned weights in the core
recurrent blocks) to survive a change in tokenization. This tests whether
the core blocks learn modality-agnostic patterns that are independent of
the tokenizer, or whether they are tightly coupled to the specific
vocabulary they were trained on.

A positive result (knowledge transfers through surgery) would mean:
- The core RWKV blocks learn *computational strategies* (how to track
  state, how to compare numbers, how to find patterns) rather than
  *token-specific patterns*.
- You can swap out the tokenizer for any other tokenization scheme
  (bytes, patches, other languages) and keep the training investment.

A negative result (from-scratch byte model matches or beats the surgery
model) would mean:
- The core blocks are tightly coupled to the vocabulary.
- "Surgery" would require re-training the core blocks too — but maybe
  with a different learning rate or adapter layers.

## Experiment Design

### Single variable

**Interface initialization**: pre-trained token weights copied to byte
positions vs. random initialization. *Everything else held fixed*:
architecture (dim=64, 2 layers), task (`sum_threshold` rule), training
steps, optimizer, data distribution.

### Arms

| Arm | Pre-training | Surgery | Core frozen? | What's trained |
|-----|-------------|---------|--------------|----------------|
| A (surgery) | 300 steps char-level | Copy embed/head where chars match bytes | Yes | New embed + head only |
| B (scratch) | None | N/A | No | All weights from scratch |

### Prediction

If the core blocks learned generalizable computation:
- Arm A loss < Arm B loss at matched steps
- Arm A converges faster (steeper initial descent)

If the core blocks are token-bound:
- Arm A loss ≈ Arm B loss (no knowledge transfer)
- Or Arm A loss > Arm B loss (init from partial mapping is misleading)

### Data

`sum_threshold` rule from `rule_generator.py`: sum of numbers in a
sequence >= threshold → predict label character. Same rule for both
phases. Only the tokenization changes (char-level tokens → byte-level
tokens).

### Size

- Model: ~52K params (vocab=74 chars) / ~57K params (vocab=258 bytes)
- Trainable after surgery: ~7K params (embed + head only)
- 300 steps per phase, batch_size=4, ~1 minute on CPU

---

## Source

- `src/token_surgery.py` — the experiment script (Phase 1-4)
- `src/rwkv_nano.py` — the RWKV model
- `src/byte_vocab.py` — byte-level vocabulary (258 tokens)
- `src/rule_generator.py` — synthetic rule generation

## Research backing

See [`research/rwkv_overview.md`](../research/rwkv_overview.md) for RWKV
foundations, particularly DREAMSTATE (state editing shows RWKV state is
modality-independent at the core level).

See [`research/byte_level_models.md`](../research/byte_level_models.md)
for BLT and MambaByte which prove byte-level training works — the
question here is whether pre-trained *token-level* weights help when
switching to bytes.
