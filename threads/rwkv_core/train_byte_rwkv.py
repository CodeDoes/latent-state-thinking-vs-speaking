#!/usr/bin/env python3
"""Byte-level RWKV training (no tokenizer cost).

Identical scaffold to ``src/train_rwkv.py`` but with the tokenizer replaced
by an absolute byte mapping (``src/byte_vocab.py``). vocab_size drops from
74 (char-level) to 258 (PAD + UNK + 256 bytes).

Why: BLT-style research needs models that never see a tokenizer. Byte-RWKV
is the minimal first step.

Usage:
    PYTHONPATH=. python3 src/train_byte_rwkv.py --exp_id byte_exp_001 --steps 2000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from domains.byte.byte_vocab import encode, decode, VOCAB_SIZE, PAD_ID, UNK_ID
from threads.memory_growth.logic_niiah_generator import LogicNiiahGenerator
from domains.rwkv.rwkv_nano import RWKVNano, count_params


def example_to_tensor(
    text: str,
    answer_spans: list[tuple[int, int]],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a generator example to (input_ids, loss_mask).

    loss_mask is 1 for answer token positions, 0 for context.
    Byte-level token offsets differ from char offsets: a UTF-8 character
    that encodes to multiple bytes gives multiple token ids per "char".
    For next-token prediction we keep the loss_mask indexing in *token*
    space (i.e., answer-spans need translating). For simplicity in this
    first script we re-use the generator's char offsets as byte offsets
    then truncate — this is approximate but valid for ASCII-heavy
    generator output.
    """
    tokens = encode(text, max_len=max_len + 1)

    # Build loss mask in token space
    mask = torch.zeros(len(tokens), dtype=torch.float)
    for s, e in answer_spans:
        # Char offsets → byte offsets (≈ same in ASCII range)
        for t in range(max(0, s - 1), min(len(tokens) - 1, e - 1)):
            mask[t] = 1.0

    return torch.tensor(tokens, dtype=torch.long), mask


def generate_batch_tensors(
    generator: LogicNiiahGenerator,
    batch_size: int,
    max_len: int,
    gen_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch and return (input_ids, targets, mask) tensors."""
    examples = generator.generate_batch(batch_size, **gen_kwargs)
    batch_ids, batch_masks = [], []
    for ex in examples:
        ids, mask = example_to_tensor(ex["text"], ex["answer_spans"], max_len)
        batch_ids.append(ids)
        batch_masks.append(mask)
    input_ids = torch.stack(batch_ids)
    mask = torch.stack(batch_masks)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = PAD_ID
    return input_ids, targets, mask


def save_checkpoint(exp_dir, model, optimizer, step, metrics, generator_rng_state):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "generator_rng_state": generator_rng_state,
        "config": {
            "vocab_size": VOCAB_SIZE,
            "dim": model.dim,
            "num_layers": model.num_layers,
            "vocab_kind": "byte",
        },
    }
    torch.save(ckpt, exp_dir / "checkpoint.pt")
    torch.save(ckpt, exp_dir / f"checkpoint_step_{step}.pt")
    ckpts = sorted(exp_dir.glob("checkpoint_step_*.pt"))
    for f in ckpts[:-3]:
        f.unlink()


def load_checkpoint(exp_dir, model, optimizer, device):
    ckpt = torch.load(exp_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt.get("metrics", {}), ckpt.get("generator_rng_state")


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Train byte-level RWKV on logic niiah")
    ap.add_argument("--exp_id", default="byte_exp_001")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_vars", type=int, default=3)
    ap.add_argument("--min_transforms", type=int, default=2)
    ap.add_argument("--max_transforms", type=int, default=5)
    ap.add_argument("--noise_min", type=int, default=1)
    ap.add_argument("--noise_max", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(args.device)
    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["git_hash"] = get_git_hash()
    config["vocab_size"] = VOCAB_SIZE
    config["vocab_kind"] = "byte"
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Byte vocab: {VOCAB_SIZE} tokens (PAD=0, UNK=1, 256 bytes)")
    print(f"Config: {exp_dir / 'config.json'}")
    print(f"  git hash: {config['git_hash']}")

    generator = LogicNiiahGenerator(seed=args.seed)
    gen_kwargs = {
        "num_vars": args.num_vars,
        "min_transforms": args.min_transforms,
        "max_transforms": args.max_transforms,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
    }

    model = RWKVNano(
        vocab_size=VOCAB_SIZE, dim=args.dim, num_layers=args.layers, pad_token_id=PAD_ID
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model params: {count_params(model):,}")

    start_step = 1
    best_acc = 0.0
    metrics_log = []

    if args.resume and (exp_dir / "checkpoint.pt").exists():
        start_step, old_metrics, rng_state = load_checkpoint(
            exp_dir, model, optimizer, device
        )
        best_acc = old_metrics.get("best_acc", 0.0)
        metrics_log = old_metrics.get("log", [])
        print(f"Resumed from step {start_step} (best_acc={best_acc:.3f})")

    print("Warming up generator RNG...")
    generator.reseed(args.seed)
    for _ in range(start_step * args.batch_size):
        generator.generate(**gen_kwargs)
    print("Ready.")

    t_start = time.time()
    step = start_step
    while step <= args.steps:
        input_ids, targets, mask = generate_batch_tensors(
            generator, args.batch_size, args.max_len, gen_kwargs
        )
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view_as(mask)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-6)
            print(
                f"step {step:6d}/{args.steps}  "
                f"loss={loss.item():.4f}  speed={steps_per_sec:.1f} st/s"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                exp_dir, model, optimizer, step,
                {"best_acc": best_acc, "log": metrics_log},
                generator.rng.getstate(),
            )
            print(f"  checkpoint at step {step}")

        step += 1

    elapsed = time.time() - t_start
    result = {
        "best_accuracy": best_acc,
        "total_steps": args.steps,
        "elapsed_s": round(elapsed, 1),
        "params": count_params(model),
        "git_hash": config["git_hash"],
        "vocab_size": VOCAB_SIZE,
        "vocab_kind": "byte",
    }
    (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"\nDone. Bytes-lived RWKV produced {count_params(model):,} params")
    print(f"Results saved to: {exp_dir}/")


if __name__ == "__main__":
    main()
