"""Train ProtoBLT_RWKV on text.

Predecessor to GPU work, designed to surface training-time glitches:
- shape mismatches at patch boundaries
- NaNs in the decoder's gated fusion
- gradient stalls when decoder RWKV depth is non-trivial
"""

from __future__ import annotations

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.proto_blt_rwkv import ProtoBLT_RWKV
from src.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, ID_TO_BYTE, UNK_ID
from src.rwkv_nano import count_params


def load_text(path: Path) -> list[int]:
    text = path.read_bytes().decode('utf-8', errors='replace')
    return [BYTE_TO_ID.get(ord(c), UNK_ID) for c in text]


def batches(stream, max_len, batch_size):
    n = ((len(stream) - 1) // max_len) * max_len
    stream = stream[: n + 1]
    for start in range(0, len(stream) - max_len - 1, batch_size * max_len):
        rows = []
        for i in range(batch_size):
            chunk = stream[start + i * max_len : start + (i + 1) * max_len + 1]
            if len(chunk) < max_len + 1:
                break
            rows.append(chunk)
        if not rows:
            continue
        for r in rows:
            while len(r) < max_len + 1:
                r.append(PAD_ID)
        batch = torch.tensor(rows, dtype=torch.long)
        yield batch[:, :-1], batch[:, 1:].contiguous()


@torch.no_grad()
def sample(model, prompt="Once upon a", max_new=80, max_len=256):
    model.eval()
    tokens = [BYTE_TO_ID.get(ord(c), UNK_ID) for c in prompt[:max_len]]
    tokens = tokens[-max_len:]
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    for _ in range(max_new):
        try:
            logits, _ = model(x)
        except Exception as e:
            return f"[gen err: {e}]"
        last = logits[0, -1]
        nxt = int(last.argmax().item())
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
        if x.shape[1] > max_len:
            x = x[:, -max_len:]
    out = x[0].tolist()
    chars = [chr(ID_TO_BYTE[t]) for t in out if t in ID_TO_BYTE and ID_TO_BYTE[t] != PAD_ID]
    return ''.join(chars)


def git_hash():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default='experiments/byte_ts_001/text.txt')
    ap.add_argument('--exp_id', default='proto_blt_ts_001')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--dim', type=int, default=96)
    ap.add_argument('--n-enc-layers', type=int, default=2)
    ap.add_argument('--n-patch-layers', type=int, default=1)
    ap.add_argument('--n-dec-layers', type=int, default=2)
    ap.add_argument('--patch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--sample-every', type=int, default=100)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    exp_dir = Path('experiments') / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['git_hash'] = git_hash()
    (exp_dir / 'config.json').write_text(json.dumps(config, indent=2))

    print(f"Loading text from {args.text}")
    stream = load_text(Path(args.text))
    print(f"  {len(stream)} byte tokens")

    model = ProtoBLT_RWKV(
        vocab_size=VOCAB_SIZE,
        dim=args.dim,
        n_enc_layers=args.n_enc_layers,
        n_patch_layers=args.n_patch_layers,
        n_dec_layers=args.n_dec_layers,
        patch_size=args.patch_size,
    ).to(device)
    n_params = count_params(model)
    print(f"\nProtoBLT_RWKV params: {n_params:,}")
    print(f"  byte_enc   : {count_params(model.byte_enc):,}")
    print(f"  patch_slot : {count_params(model.patch_slot):,}")
    print(f"  byte_dec   : {count_params(model.byte_dec):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nTraining {args.steps} steps, batch={args.batch_size}, len={args.max_len} ...")
    t0 = time.time()
    log = []
    step = 0
    for input_ids, targets in batches(stream, args.max_len, args.batch_size):
        if step >= args.steps:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        try:
            logits, _ = model(input_ids)
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
                  f"grad={float(grad_norm):.3f}  speed={(step/elapsed):.1f} st/s",
                  flush=True)
            log.append({'step': step, 'loss': float(loss.item()), 'grad': float(grad_norm)})

        if step % args.sample_every == 0:
            print(f"  >> sample: {sample(model)!r}")

    elapsed = time.time() - t0
    print(f"\nDone. {step} steps in {elapsed:.1f}s")
    (exp_dir / 'metrics.json').write_text(json.dumps({
        'final_step': step,
        'final_loss': float(log[-1]['loss']) if log else None,
        'elapsed_s': round(elapsed, 1),
        'params': n_params,
        'git_hash': config['git_hash'],
    }, indent=2))
    torch.save({'model_state': model.state_dict(), 'config': config, 'step': step},
               exp_dir / 'checkpoint.pt')
    print(f"Checkpoint saved to {exp_dir}/")


if __name__ == '__main__':
    main()
