"""Quick training test for encoder-patcher-decoder.

Just verifies the architecture trains and see how the loops behave.
Same setup as other BLT experiments: 99KB TinyStories, small model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
import time
import json

from threads.byte_state_byte.encoder_patcher_decoder import EncoderPatcherDecoder
from domains.byte.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, ID_TO_BYTE, UNK_ID
from threads.unsorted.simple_rnn_receptance import SimpleRNNReceptance
import math


def load_text(path: Path) -> list[int]:
    text = path.read_bytes().decode('utf-8', errors='replace')
    return [BYTE_TO_ID.get(ord(c), UNK_ID) for c in text]


def make_batches(stream: list[int], max_len: int, batch_size: int):
    n = (len(stream) - 1) // max_len * max_len
    stream = stream[: n + 1]
    for start in range(0, len(stream) - max_len - 1, batch_size * max_len):
        rows = []
        for i in range(batch_size):
            chunk = stream[start + i * max_len : start + (i + 1) * max_len + 1]
            if len(chunk) < max_len + 1:
                continue
            rows.append(chunk)
        if len(rows) == 0:
            continue
        for r in rows:
            while len(r) < max_len + 1:
                r.append(PAD_ID)
        batch = torch.tensor(rows, dtype=torch.long)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:].contiguous()
        yield input_ids, targets


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def train(steps=300, batch_size=8, max_len=128, lr=3e-4, log_every=50):
    device = torch.device('cpu')
    print(f"Device: {device}")

    # Load data
    text_path = Path('threads/g1g_frontend/experiments/byte_ts_001/text.txt')
    stream = load_text(text_path)
    print(f"Loaded {len(stream):,} bytes")

    # Build model
    model = EncoderPatcherDecoder(dim=64, patch_size=4, max_loops=4).to(device)
    n_params = count_params(model)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    losses = []
    t0 = time.time()

    for step in range(steps):
        batch_iter = make_batches(stream, max_len, batch_size)
        try:
            input_ids, targets = next(batch_iter)
        except StopIteration:
            batch_iter = make_batches(stream, max_len, batch_size)
            input_ids, targets = next(batch_iter)

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits, info = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=PAD_ID,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - t0
            enc_loops = info['encoder']['loop_count']
            dec_loops = info['decoder']['loop_count']
            enc_sup = info['encoder_surprise'].mean().item()
            dec_sup = info['decoder_surprise'].mean().item()
            print(f"step {step:4d} | loss {loss_val:.4f} | enc_loops {enc_loops} | dec_loops {dec_loops} | enc_surprise {enc_sup:.3f} | dec_surprise {dec_sup:.3f} | {elapsed:.1f}s")

    print(f"\nDone. Final loss: {losses[-1]:.4f}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Loss reduction: {(losses[0] - losses[-1]):.4f}")


if __name__ == "__main__":
    train(steps=300)
