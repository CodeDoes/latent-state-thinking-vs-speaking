"""GPU training script for ProtoBLT_RWKV.

Scaled-up version of src/train_proto_blt.py:
- larger model (~1M params target)
- longer training (semantic gradient budget, not 200 steps)
- bf16 autocast for memory
- structured logging
"""

from __future__ import annotations

import argparse
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path

from threads.blt.proto_blt_rwkv import ProtoBLT_RWKV
from domains.byte.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, ID_TO_BYTE, UNK_ID
from domains.rwkv.rwkv_nano import count_params


def load_text(path: Path) -> list[int]:
    text = path.read_bytes().decode('utf-8', errors='replace')
    return [BYTE_TO_ID.get(ord(c), UNK_ID) for c in text]


def batches(stream, max_len, batch_size):
    n = ((len(stream) - 1) // max_len) * max_len
    stream = stream[: n + 1]
    cursor = 0
    while True:
        rows = []
        for b in range(batch_size):
            i = (cursor + b * max_len) % (len(stream) - max_len - 1)
            chunk = stream[i:i + max_len + 1]
            if len(chunk) < max_len + 1:
                break
            rows.append(chunk)
        if not rows:
            return
        for r in rows:
            while len(r) < max_len + 1:
                r.append(PAD_ID)
        yield torch.tensor(rows, dtype=torch.long)[:, :-1].contiguous(), \
              torch.tensor(rows, dtype=torch.long)[:, 1:].contiguous()
        cursor += batch_size * max_len
        if cursor > len(stream) * 2:  # epoch wrap
            cursor = 0


def git_hash():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default='threads/g1g_frontend/experiments/byte_ts_001/text.txt')
    ap.add_argument('--exp_id', default='gpu_proto_blt_001')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--n-enc-layers', type=int, default=2)
    ap.add_argument('--n-patch-layers', type=int, default=1)
    ap.add_argument('--n-dec-layers', type=int, default=2)
    ap.add_argument('--patch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    exp_dir = Path('experiments') / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['git_hash'] = git_hash()
    (exp_dir / 'config.json').write_text(json.dumps(config, indent=2))

    print(f"Loading text from {args.text}")
    stream = load_text(Path(args.text))
    print(f"  {len(stream)} byte tokens (~{len(stream)/1024:.1f} KB)")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\nTraining {args.steps} steps on {device}: "
          f"batch={args.batch_size} len={args.max_len}")
    print(f"Estimated peak VRAM at this size: ~0.4 GB (well under 4 GB budget)")

    t0 = time.time()
    log = []
    step = 0
    last_loss = None
    for input_ids, targets in batches(stream, args.max_len, args.batch_size):
        if step >= args.steps:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='mean',
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.item()

        step += 1
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            speed = step / max(elapsed, 1e-6)
            print(f"  step {step:4d}/{args.steps}  loss={loss.item():.4f}  "
                  f"grad={float(grad_norm):.3f}  speed={speed:.1f} st/s  "
                  f"elapsed={elapsed:.0f}s", flush=True)
            log.append({
                'step': step, 'loss': float(loss.item()), 'grad': float(grad_norm),
                'elapsed': elapsed,
            })

    elapsed = time.time() - t0
    print(f"\nDone. {step} steps in {elapsed:.1f}s ({step/elapsed:.1f} st/s)")
    print(f"Final loss: {last_loss:.4f}")

    (exp_dir / 'metrics.json').write_text(json.dumps({
        'final_step': step,
        'final_loss': float(last_loss) if last_loss is not None else None,
        'elapsed_s': round(elapsed, 1),
        'speed_st_per_s': step / max(elapsed, 1e-6),
        'params': n_params,
        'git_hash': config['git_hash'],
    }, indent=2))
    torch.save({'model_state': model.state_dict(), 'config': config, 'step': step},
               exp_dir / 'checkpoint.pt')
    print(f"Checkpoint saved to {exp_dir}/")


if __name__ == '__main__':
    main()
