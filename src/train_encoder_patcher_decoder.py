"""Train encoder-patcher-decoder with surprise-router loops.

Tests the full architecture:
- Encoder loop (max 4 iterations, exit on low surprise)
- Patcher (compress bytes to patches)
- Decoder loop (max 4 iterations, exit on low surprise)

Goal: see if the surprise routers learn meaningful behavior and
whether the loops provide any benefit.
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

from src.encoder_patcher_decoder import EncoderPatcherDecoder
from src.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, ID_TO_BYTE, UNK_ID


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
    device = next(model.parameters()).device
    tokens = [BYTE_TO_ID.get(ord(c), UNK_ID) for c in prompt_text[:max_len]]
    tokens = tokens[-max_len:]
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        try:
            logits, _ = model(x)
        except Exception as e:
            return f"[generation error: {e}]"
        last_logits = logits[0, -1]
        next_token = int(last_logits.argmax().item())
        x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
        if x.shape[1] > max_len:
            x = x[:, -max_len:]
    out_ids = x[0].cpu().tolist()
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
    ap.add_argument('--exp_id', default='enc_patch_dec_001')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--patch-size', type=int, default=4)
    ap.add_argument('--max-loops', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--log-every', type=int, default=50)
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
    model = EncoderPatcherDecoder(
        dim=args.dim,
        patch_size=args.patch_size,
        max_loops=args.max_loops,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Info
    print(f"  embed: {sum(p.numel() for p in model.encoder.embed.parameters()):,}")
    print(f"  encoder rnn: {sum(p.numel() for p in model.encoder.rnn.parameters()):,}")
    print(f"  surprise_router: {sum(p.numel() for p in model.encoder.surprise_router.parameters()):,}")
    print(f"  patcher rnn: {sum(p.numel() for p in model.patcher.rnn.parameters()):,}")
    print(f"  decoder rnn: {sum(p.numel() for p in model.decoder.rnn.parameters()):,}")
    print(f"  surprise_router_2: {sum(p.numel() for p in model.decoder.surprise_router_2.parameters()):,}")
    print(f"  head: {sum(p.numel() for p in model.decoder.head.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Experiment dir
    exp_dir = Path(f"experiments/{args.exp_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "encoder_patcher_decoder",
        "git_hash": get_git_hash(),
        "text_path": str(text_path),
        "n_bytes": len(stream),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "dim": args.dim,
        "patch_size": args.patch_size,
        "max_loops": args.max_loops,
        "lr": args.lr,
        "lr_schedule": "cosine",
        "n_params": n_params,
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Experiment dir: {exp_dir}")

    # Training loop
    model.train()
    losses = []
    metrics = []
    t0 = time.time()

    encoder_loops_history = []
    decoder_loops_history = []

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
        
        encoder_loops_history.append(info['encoder']['loop_count'])
        decoder_loops_history.append(info['decoder']['loop_count'])

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            bytes_per_sec = (step * args.batch_size * args.max_len) / elapsed if elapsed > 0 else 0
            avg_loss = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
            
            enc_loops = sum(encoder_loops_history[-args.log_every:]) / len(encoder_loops_history[-args.log_every:])
            dec_loops = sum(decoder_loops_history[-args.log_every:]) / len(decoder_loops_history[-args.log_every:])
            
            print(f"step {step:4d} | loss {loss_val:.4f} (avg {avg_loss:.4f}) | lr {lr:.6f}")
            print(f"         | enc loops {enc_loops:.1f}/{args.max_loops} | dec loops {dec_loops:.1f}/{args.max_loops}")
            print(f"         | {bytes_per_sec:.0f} bytes/s | {elapsed:.1f}s")

            metrics.append({
                "step": step,
                "loss": loss_val,
                "avg_loss": avg_loss,
                "lr": lr,
                "encoder_loops": enc_loops,
                "decoder_loops": dec_loops,
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

    # Summary
    avg_enc_loops = sum(encoder_loops_history) / len(encoder_loops_history)
    avg_dec_loops = sum(decoder_loops_history) / len(decoder_loops_history)
    print(f"\nAvg encoder loops: {avg_enc_loops:.1f}/{args.max_loops}")
    print(f"Avg decoder loops: {avg_dec_loops:.1f}/{args.max_loops}")
    print(f"\nDone. Experiment saved to {exp_dir}/")


if __name__ == "__main__":
    main()
