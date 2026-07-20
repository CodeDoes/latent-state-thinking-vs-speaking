"""Training script for shared-state unrolled architecture.

Shared encoder/decoder weights (true RNN-style unrolling).
Encoder is applied N times, decoder is applied M times.

Architecture:
  encoder(byte_1) → state_1
  encoder(byte_2, state_1) → state_2  (same encoder, shared weights)
  ...
  encoder(byte_N, state_{N-1}) → state_N
  patch_model(state_N) → transformed_state
  decoder(transformed_state) → byte_{N+1}  (same decoder, shared weights)
  decoder(byte_{N+1}, state_{N+1}) → byte_{N+2}  (shared weights)
  ...

Loss: encoder_loss + decoder_loss
"""

from __future__ import annotations

import argparse
import json
import math
import time
import torch
from pathlib import Path

from threads.byte_state_byte.shared_state_unrolled import SharedStateUnrolled
from domains.byte.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, ID_TO_BYTE, UNK_ID
from domains.rwkv.rwkv_nano import count_params


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
    """Generate text using the shared encoder/decoder model."""
    model.eval()
    device = next(model.parameters()).device
    tokens = [BYTE_TO_ID.get(ord(c), UNK_ID) for c in prompt_text[:max_len]]
    
    # Run through encoder phase (shared encoder applied multiple times up to n_encoder_steps)
    rwkv_state = None
    encoder_state = None

    # We will use at most n_encoder_steps tokens from the prompt
    encoder_tokens = tokens[:model.n_encoder_steps]
    if len(encoder_tokens) < model.n_encoder_steps:
        encoder_tokens = encoder_tokens + [PAD_ID] * (model.n_encoder_steps - len(encoder_tokens))
        
    for i in range(model.n_encoder_steps):
        byte = torch.tensor([encoder_tokens[i]], dtype=torch.long, device=device)
        encoder_state, rwkv_state, _ = model.encoder(byte, rwkv_state)
        
    if encoder_state is None:
        return ""

    # Patch phase
    transformed_state = model.patch_model(encoder_state)

    # Decoder phase: sequentially generate new tokens
    decoder_state = None
    generated = []

    for i in range(max_new_tokens):
        if i == 0:
            input_state = transformed_state
        else:
            # Autoregressive feedback: embed the last generated token
            prev_tok = torch.tensor([generated[-1]], dtype=torch.long, device=device)
            input_state = model.encoder.embed(prev_tok)
            
        logits, decoder_state = model.decoder(input_state, decoder_state)
        next_token = int(logits[0].argmax().item())
        generated.append(next_token)

    chars = []
    for tid in generated:
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
    ap.add_argument('--text', default='threads/g1g_frontend/experiments/byte_ts_001/text.txt')
    ap.add_argument('--exp_id', default='shared_state_unrolled_001')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--n-encoder-layers', type=int, default=1)
    ap.add_argument('--n-decoder-layers', type=int, default=1)
    ap.add_argument('--n-patch-layers', type=int, default=1)
    ap.add_argument('--n-encoder-steps', type=int, default=2)
    ap.add_argument('--n-decoder-steps', type=int, default=2)
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
    model = SharedStateUnrolled(
        vocab_size=VOCAB_SIZE,
        dim=args.dim,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_patch_layers=args.n_patch_layers,
        n_encoder_steps=args.n_encoder_steps,
        n_decoder_steps=args.n_decoder_steps,
    ).to(device)

    n_params = count_params(model)
    print(f"Model params: {n_params:,}")
    print(f"  encoder (shared, applied {args.n_encoder_steps}x): {count_params(model.encoder):,}")
    print(f"  decoder (shared, applied {args.n_decoder_steps}x): {count_params(model.decoder):,}")
    print(f"  patch_model: {count_params(model.patch_model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Experiment dir
    exp_dir = Path(f"threads/byte_state_byte/experiments/{args.exp_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "shared_state_unrolled",
        "git_hash": get_git_hash(),
        "text_path": str(text_path),
        "n_bytes": len(stream),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "dim": args.dim,
        "n_encoder_layers": args.n_encoder_layers,
        "n_decoder_layers": args.n_decoder_layers,
        "n_patch_layers": args.n_patch_layers,
        "n_encoder_steps": args.n_encoder_steps,
        "n_decoder_steps": args.n_decoder_steps,
        "lr": args.lr,
        "lr_schedule": "cosine",
        "n_params": n_params,
        "architecture": f"unrolled: {args.n_encoder_steps} encoder steps → patch → {args.n_decoder_steps} decoder steps (separate weights)",
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Experiment dir: {exp_dir}")

    # Training loop
    model.train()
    losses = []
    metrics_log = []
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

        loss, metrics = model(input_ids, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            bytes_per_sec = (step * args.batch_size * args.max_len) / elapsed if elapsed > 0 else 0
            avg_loss = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
            print(f"step {step:4d} | loss {loss_val:.4f} (avg {avg_loss:.4f}) | lr {lr:.6f} | "
                  f"enc {metrics['encoder_loss']:.4f} | dec {metrics['decoder_loss']:.4f} | "
                  f"{bytes_per_sec:.0f} bytes/s")

            metrics_log.append({
                "step": step,
                "loss": loss_val,
                "avg_loss": avg_loss,
                "lr": lr,
                **metrics,
                "bytes_per_sec": bytes_per_sec,
                "elapsed": elapsed,
            })

    # Save final checkpoint
    torch.save(model.state_dict(), exp_dir / "checkpoint.pt")
    (exp_dir / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics_log))

    # Generation sample
    model.eval()
    sample = generate_sample(model, "Once upon a time", max_new_tokens=80)
    print(f"\nGeneration sample:\n{sample}")
    (exp_dir / "sample.txt").write_text(sample)

    print(f"\nDone. Experiment saved to {exp_dir}/")


if __name__ == "__main__":
    main()
