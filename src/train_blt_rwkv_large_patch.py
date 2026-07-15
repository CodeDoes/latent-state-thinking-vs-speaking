"""Train step 5: BLT with RWKV byte encoder/decoder, 3x larger transformer patch mixer.

Changes from step 2 (blt_rwkv_enc_dec):
- RWKV time_decay init: 0 → -0.5 (slower decay, longer memory)
- Patch mixer dim: 32 → 192 (3x larger, more capacity)
- Patch mixer layers: 1 → 2 (deeper)
- Patch mixer heads: 4 → 8 (more attention heads)
- Training steps: 300 → 1000 (longer training)
- Learning rate: cosine schedule (starts at lr, decays to 0)

Hypothesis: patch model was capacity-starved. 3x more params + slower decay + longer training → lower loss.
"""

from __future__ import annotations

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import math

from src.blt_rwkv_large_patch import BLT_RWKV_LargePatch
from src.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, ID_TO_BYTE, UNK_ID
from src.rwkv_nano import count_params


def load_text(path: Path) -> list[int]:
    text = path.read_bytes().decode('utf-8', errors='replace')
    return [BYTE_TO_ID.get(ord(c), UNK_ID) for c in text]


def make_batches(stream: list[int], max_len: int, batch_size: int):
    """Yield (input, target) tensors from a flat byte stream."""
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


@torch.no_grad()
def generate_sample(model, prompt_text: str, max_new_tokens: int = 80, max_len: int = 256):
    model.eval()
    tokens = [BYTE_TO_ID.get(ord(c), UNK_ID) for c in prompt_text[:max_len]]
    tokens = tokens[-max_len:]
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        try:
            logits, _ = model(x)
        except Exception as e:
            return f"[generation error: {e}]"
        last_logits = logits[0, -1]
        next_token = int(last_logits.argmax().item())
        x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
        if x.shape[1] > max_len:
            x = x[:, -max_len:]
    out_ids = x[0].tolist()
    chars = []
    for tid in out_ids:
        b = ID_TO_BYTE.get(tid)
        if b is not None and b != PAD_ID:
            chars.append(chr(b))
    return ''.join(chars)


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int = 50) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def get_git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default='experiments/byte_ts_001/text.txt')
    ap.add_argument('--exp_id', default='blt_rwkv_large_patch_001')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--dim-byte', type=int, default=64)
    ap.add_argument('--dim-patch', type=int, default=192)
    ap.add_argument('--n-layers-inner', type=int, default=2)
    ap.add_argument('--n-layers-outer', type=int, default=2)
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--patch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--log-every', type=int, default=100)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    text_path = Path(args.text)
    if not text_path.exists():
        print(f"Text file not found: {text_path}")
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("Hello world. " * 1000)
    stream = load_text(text_path)
    print(f"Loaded {len(stream):,} bytes")

    # Build model
    model = BLT_RWKV_LargePatch(
        vocab_size=VOCAB_SIZE,
        dim_byte=args.dim_byte,
        dim_patch=args.dim_patch,
        n_layers_inner=args.n_layers_inner,
        n_layers_outer=args.n_layers_outer,
        n_heads=args.n_heads,
        patch_size=args.patch_size,
    ).to(device)

    n_params = count_params(model)
    print(f"Model params: {n_params:,}")
    print(f"  byte_enc (RWKV): {count_params(model.byte_enc):,}")
    print(f"  to_patch: {count_params(model.to_patch):,}")
    print(f"  patch_mixer (transformer): {count_params(model.patch_mixer):,}")
    print(f"  to_byte: {count_params(model.to_byte):,}")
    print(f"  byte_dec: {count_params(model.byte_dec):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Experiment dir
    exp_dir = Path(f"experiments/{args.exp_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "blt_rwkv_large_patch",
        "git_hash": get_git_hash(),
        "text_path": str(text_path),
        "n_bytes": len(stream),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "dim_byte": args.dim_byte,
        "dim_patch": args.dim_patch,
        "n_layers_inner": args.n_layers_inner,
        "n_layers_outer": args.n_layers_outer,
        "n_heads": args.n_heads,
        "patch_size": args.patch_size,
        "lr": args.lr,
        "lr_schedule": "cosine",
        "n_params": n_params,
        "changes": [
            "RWKV time_decay init: 0 → -0.5",
            "Patch dim: 32 → 192 (3x)",
            "Patch layers: 1 → 2",
            "Patch heads: 4 → 8",
            "Steps: 300 → 1000",
            "LR schedule: cosine with warmup",
        ],
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Experiment dir: {exp_dir}")

    # Training loop
    model.train()
    losses = []
    metrics = []
    t0 = time.time()

    for step in range(args.steps):
        # Cosine LR
        lr = cosine_lr(step, args.steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        batch_iter = make_batches(stream, args.max_len, args.batch_size)
        try:
            input_ids, targets = next(batch_iter)
        except StopIteration:
            batch_iter = make_batches(stream, args.max_len, args.batch_size)
            input_ids, targets = next(batch_iter)

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits, _ = model(input_ids)
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

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            bytes_per_sec = (step * args.batch_size * args.max_len) / elapsed if elapsed > 0 else 0
            avg_loss = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
            print(f"step {step:4d} | loss {loss_val:.4f} (avg {avg_loss:.4f}) | lr {lr:.6f} | {bytes_per_sec:.0f} bytes/s | {elapsed:.1f}s")

            metrics.append({
                "step": step,
                "loss": loss_val,
                "avg_loss": avg_loss,
                "lr": lr,
                "bytes_per_sec": bytes_per_sec,
                "elapsed": elapsed,
            })

    # Save final checkpoint
    torch.save(model.state_dict(), exp_dir / "checkpoint.pt")
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Generation sample
    model.eval()
    sample = generate_sample(model, "Once upon a time", max_new_tokens=80)
    print(f"\nGeneration sample:\n{sample}")
    (exp_dir / "sample.txt").write_text(sample)

    print(f"\nDone. Experiment saved to {exp_dir}/")


if __name__ == "__main__":
    main()
