#!/usr/bin/env python3
"""THE training loop — one place for stepping, logging, checkpointing.

Extracted from src/__main__.py (which was the canonical copy of a loop that
had been re-implemented in ~40 scripts). Threads call `train(...)` with
their own model; anything train-loop-shaped that isn't here is a bug to
fix by adding a config flag, not by copying the loop.

[meta]
status: active
[/meta]
"""

import math
import time
from pathlib import Path


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def pad_batch(batch):
    """Pad a list of token-id sequences to equal length (0 = pad).

    Returns (x_list, y_list): inputs and next-token targets.
    """
    max_len = max(len(s) for s in batch)
    x_list, y_list = [], []
    for seq in batch:
        x = seq[:-1]
        y = seq[1:]
        pad = (max_len - 1) - len(x)
        if pad > 0:
            x = x + [0] * pad
            y = y + [0] * pad
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def cosine_lr(step, total, base):
    return base * 0.5 * (1 + math.cos(math.pi * min(step / max(total, 1), 1.0)))


def pick_device(want_gpu=False):
    import torch
    if want_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(model, data, config, run_log=None, device=None) -> dict:
    """Next-token cross-entropy loop.

    config keys: steps, lr, batch_size, log_every, save_every(optional).
    Returns the metrics dict; artifacts go through `run_log` (a kit.runlog.Run)
    when given — otherwise just prints.
    """
    import torch
    import torch.nn.functional as F

    device = device or pick_device(bool(config.get("gpu")))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    steps, bs = config["steps"], config["batch_size"]
    log_every = config.get("log_every", 10)
    save_every = config.get("save_every", 0)

    emit = run_log.log if run_log is not None else print
    best = float("inf")
    start = None
    t0 = time.time()

    for step in range(steps):
        ts = time.time()
        total, nb = 0.0, 0
        for i in range(0, len(data), bs):
            x_list, y_list = pad_batch(data[i : i + bs])
            x = torch.tensor(x_list, dtype=torch.long, device=device)
            y = torch.tensor(y_list, dtype=torch.long, device=device)
            logits, info = model(x)
            if not isinstance(info, dict):
                logits, info = info, {}
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            total += loss.item()
            nb += 1
        avg = total / nb
        best = min(best, avg)
        start = avg if start is None else start
        dt = time.time() - ts
        if step % log_every == 0 or step < 3:
            emit(f"step {step:4d} | loss {avg:.4f} | best {best:.4f} "
                 f"| {dt:.2f}s | {time.time() - t0:.1f}s")
        if run_log is not None and save_every and step and step % save_every == 0:
            run_log.save_checkpoint(model, name=f"step_{step:04d}.pt")

    metrics = {
        "start_loss": start, "final_loss": avg, "best_loss": best,
        "total_steps": steps, "params": count_params(model),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if run_log is not None:
        run_log.record(**metrics)
    return metrics
