"""Train BLT-RWKV on a small TinyStories slice.

Goal: find technical glitches before scaling up.
Single-stream text training, no patcher — fixed-size windows.
Each step is one byte-prediction cross-entropy over a sliding window.
"""

from __future__ import annotations

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from threads.blt.blt_rwkv import BLT_RWKV
from domains.byte.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, ID_TO_BYTE, UNK_ID
from domains.rwkv.rwkv_nano import count_params


def load_text(path: Path) -> list[int]:
    text = path.read_bytes().decode('utf-8', errors='replace')
    return [BYTE_TO_ID.get(ord(c), UNK_ID) for c in text]


def make_batches(stream: list[int], max_len: int, batch_size: int):
    """Yield (input, target) tensors from a flat byte stream."""
    n = (len(stream) - 1) // max_len * max_len
    # truncate so we get clean windows
    stream = stream[: n + 1]
    # reshape into rows of max_len
    for start in range(0, len(stream) - max_len - 1, batch_size * max_len):
        rows = []
        for i in range(batch_size):
            chunk = stream[start + i * max_len : start + (i + 1) * max_len + 1]
            if len(chunk) < max_len + 1:
                continue
            rows.append(chunk)
        if len(rows) == 0:
            continue
        # pad short rows
        for r in rows:
            while len(r) < max_len + 1:
                r.append(PAD_ID)
        batch = torch.tensor(rows, dtype=torch.long)  # [B, T+1]
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
        # greedy generation; no sampling — we want texture
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


def get_git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default='threads/g1g_frontend/experiments/byte_ts_001/text.txt')
    ap.add_argument('--exp_id', default='byte_ts_001')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--dim-byte', type=int, default=64)
    ap.add_argument('--dim-patch', type=int, default=32)
    ap.add_argument('--n-layers-inner', type=int, default=2)
    ap.add_argument('--n-layers-outer', type=int, default=1)
    ap.add_argument('--patch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--sample-every', type=int, default=60)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    exp_dir = Path('experiments') / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['git_hash'] = get_git_hash()
    (exp_dir / 'config.json').write_text(json.dumps(config, indent=2))

    print(f"Loading text from {args.text} ...")
    stream = load_text(Path(args.text))
    print(f"  {len(stream)} byte tokens")
    n_batches_per_epoch = max(1, (len(stream) - args.max_len) // (args.batch_size * args.max_len))
    print(f"  ~{n_batches_per_epoch} batches per pass")

    # Build model
    model = BLT_RWKV(
        vocab_size=VOCAB_SIZE,
        dim_byte=args.dim_byte,
        dim_patch=args.dim_patch,
        n_layers_inner=args.n_layers_inner,
        n_layers_outer=args.n_layers_outer,
        patch_size=args.patch_size,
    ).to(device)
    n_params = count_params(model)
    print(f"\nBLT_RWKV params: {n_params:,}")
    print(f"  byte_enc: {count_params(model.byte_enc):,}")
    print(f"  patch_mixer: {count_params(model.patch_mixer):,}")
    print(f"  byte_dec: {count_params(model.byte_dec):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nTraining {args.steps} steps, batch={args.batch_size}, len={args.max_len} ...")
    t0 = time.time()
    log = []
    step = 0
    while step < args.steps:
        for input_ids, targets in make_batches(stream, args.max_len, args.batch_size):
            if step >= args.steps:
                break
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            try:
                logits, _ = model(input_ids)
                # cross-entropy on each position
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction='mean',
                )
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except Exception as e:
                print(f"  TRAIN ERROR at step {step}: {type(e).__name__}: {e}")
                raise

            step += 1
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                print(f"  step {step:4d}/{args.steps}  loss={loss.item():.4f}  "
                      f"grad_norm={float(grad_norm):.3f}  speed={(step/elapsed):.1f} st/s",
                      flush=True)
                log.append({'step': step, 'loss': float(loss.item()), 'grad': float(grad_norm)})

            if step % args.sample_every == 0:
                model.eval()
                sample = generate_sample(model, "Once upon a", max_new_tokens=60, max_len=args.max_len)
                print(f"  >> sample: {sample!r}")
                model.train()

    elapsed = time.time() - t0
    print(f"\nDone. {step} steps in {elapsed:.1f}s")
    (exp_dir / 'metrics.json').write_text(json.dumps({
        'final_step': step,
        'final_loss': float(log[-1]['loss']) if log else None,
        'elapsed_s': round(elapsed, 1),
        'params': n_params,
        'git_hash': config['git_hash'],
    }, indent=2))

    # Save checkpoint
    torch.save({'model_state': model.state_dict(), 'config': config, 'step': step},
               exp_dir / 'checkpoint.pt')
    print(f"Checkpoint saved.")


if __name__ == '__main__':
    main()
