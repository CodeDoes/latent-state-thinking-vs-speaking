# AGENTS.md

Project: byte-level front-end for frozen RWKV-7 g1g 2.9B. Replace tokenizer, embedding, and layer 0 with tiny learned models that read raw bytes directly.

Core insight: the g1g's byte-interface model (`byte_embed` + 32 RWKV blocks) already accepts raw bytes. Our job is to compress/accelerate the front-end — replacing expensive components with tiny learned alternatives that produce equivalent hidden states.

## Layout

```
src/                Python modules — models, training, inference
theories/           prose: design, hypotheses, results
experiments/        run artifacts (checkpoints, logs, cached datasets)
```

## Active Components

- **`src/hybrid_tokenizer.py`** — Production tokenizer. TRIE for boundaries, XOR hash for fast ID lookup. 96% story-level match with real tokenizer. Use for any pipeline that needs discrete token IDs.

- **`src/train_byte_ae.py`** — Byte-level auto-encoder. Stacked 2-layer minGRU reads bytes, produces latents. Trigger fires at token boundaries. The latent at trigger position is the compact representation for g1g.

- **`src/train_loopy_timemix.py`** — Predicts layer 0 time-mix state (xx) from bytes. Byte → byte_embed → proj → minGRU → xx(2560). Cos 0.77 vs real time-mix. Replaces byte_embed + layer 0.

- **`src/train_streaming_tokenizer.py`** — Various streaming tokenizer experiments (conv2d, uint8, per-position lookup).

- **`src/auto_tokenizer.py`** — ByteG1GInference: loads quantized g1g, generates byte-by-byte. NF4 cache at `~/Documents/models/rwkv7-g1g-byte-iface/nf4_cache/`.

## How to work

1. **Write scripts, not inline code.** Every training run, test, or experiment goes in a file under `src/` — never in `python3 -c "..."`. Files have easy-to-tweak config constants at the top.

2. **Read first**: check `src/_status.py` for current state, `theories/proofs.md` for proven claims, and the relevant training script for existing approach.

3. **One variable per experiment.** Change one thing, measure, commit. If loss doesn't move, fix the data or architecture — not the step count.

4. **Checkpoint and cache.** Training scripts save model checkpoints to `experiments/<name>/`. Datasets that are expensive to build get cached via pickle/torch.save.

5. **Log at fixed step intervals.** Each training script has a `LOG_EVERY` config. Print loss/metrics every N steps. Don't aggregate across epochs — raw per-step output is easier to grep. Use `train.log | grep step | tail` to check latest.

6. **Commit proven results.** Each working component gets a proof entry in `theories/proofs.md` with the commit hash and key metrics (params, accuracy, speed).

## What not to do

- Inline `python3 -c` scripts for anything beyond a 2-line test.
- Multi-cause experiments where you can't name what changed.
- Scale-up debugging without a named hypothesis.
