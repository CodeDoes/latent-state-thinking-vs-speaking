#!/usr/bin/env python3
"""Fast comparison: Standard RWKV vs Dendritic RWKV on learnable rules.

Runs ~1 minute per model with actual patterns to learn.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.rule_generator import (
    generate_batch, encode, decode, VOCAB, char_to_id, PAD_ID, UNK_ID, RULES
)
from src.rwkv_nano import RWKVNano, count_params
from src.rwkv_dendritic import RWKVNanoDendritic


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def train_one(model: nn.Module, rule_name: str, device: torch.device,
              steps: int = 500, batch_size: int = 16, max_len: int = 256,
              lr: float = 3e-4, seed: int = 42) -> dict:
    """Train on a single rule and return metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device).train()

    best_acc = 0.0
    step_times = []

    for step in range(1, steps + 1):
        # Generate fresh batch
        batch = generate_batch(rule_name, batch_size, seed=seed + step)

        # Encode and pad
        max_t = max(len(encode(ex['text'])) for ex in batch)
        max_t = min(max_t, max_len)

        input_ids = []
        targets = []
        masks = []

        for ex in batch:
            tokens = encode(ex['text'])
            if len(tokens) > max_t:
                tokens = tokens[:max_t]

            # Build mask for answer position (last token before padding)
            mask = torch.zeros(max_t)
            if len(tokens) > 1:
                mask[len(tokens) - 2] = 1.0  # predict last answer char

            # Next-token targets
            tgt = torch.roll(torch.tensor(tokens), -1)
            tgt[-1] = PAD_ID

            # Pad
            pad_len = max_t - len(tokens)
            if pad_len > 0:
                tokens = tokens + [PAD_ID] * pad_len
                tgt = torch.cat([tgt, torch.full((pad_len,), PAD_ID)])
                mask = torch.cat([mask, torch.zeros(pad_len)])

            input_ids.append(tokens)
            targets.append(tgt.tolist())
            masks.append(mask.tolist())

        input_ids = torch.tensor(input_ids, device=device)
        targets = torch.tensor(targets, device=device)
        masks = torch.tensor(masks, device=device)

        # Forward
        t0 = time.time()
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='none',
        )
        loss = loss.view_as(masks)
        loss = (loss * masks).sum() / (masks.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_times.append(time.time() - t0)

        # Quick eval every 100 steps
        if step % 100 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                eval_batch = generate_batch(rule_name, 32, seed=9999)
                correct = total = 0
                for ex in eval_batch:
                    tokens = encode(ex['text'])
                    if len(tokens) > max_len:
                        tokens = tokens[:max_len]
                    if len(tokens) < 2:
                        continue
                    inp = torch.tensor([tokens[:-1]], device=device)
                    tgt_token = tokens[-1]
                    out, _ = model(inp)
                    pred = out[0, -1].argmax().item()
                    if pred == tgt_token:
                        correct += 1
                    total += 1
                acc = correct / total if total > 0 else 0
                best_acc = max(best_acc, acc)
                print(f"  step {step:4d}/{steps}  loss={loss.item():.4f}  "
                      f"acc={acc:.3f}  best={best_acc:.3f}  "
                      f"{sum(step_times[-100:])/max(1,len(step_times[-100:]))*1000:.1f}ms/step")
            model.train()

    return {'final_acc': best_acc, 'avg_step_ms': sum(step_times)/len(step_times)*1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['standard', 'dendritic', 'both'], default='both')
    ap.add_argument('--rule', choices=list(RULES.keys()), default='sum_threshold')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_len', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--exp_id', default='rule_compare')
    args = ap.parse_args()

    device = torch.device(args.device)
    exp_dir = Path('experiments') / args.exp_id / args.rule
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['git_hash'] = get_git_hash()
    config['vocab_size'] = len(VOCAB)
    (exp_dir / 'config.json').write_text(json.dumps(config, indent=2))

    print(f"Rule: {args.rule} | Device: {device} | Git: {config['git_hash']}")
    print(f"Vocab: {len(VOCAB)} | Dim: {args.dim} | Layers: {args.layers}")

    results = {}

    if args.model in ('standard', 'both'):
        print(f"\n{'='*50}")
        print(f"STANDARD RWKV on {args.rule}")
        print(f"{'='*50}")
        m = RWKVNano(
            vocab_size=len(VOCAB), dim=args.dim, num_layers=args.layers,
            pad_token_id=PAD_ID,
        )
        print(f"Params: {count_params(m):,}")
        results['standard'] = train_one(
            m, args.rule, device, args.steps, args.batch_size,
            args.max_len, args.lr, args.seed
        )

    if args.model in ('dendritic', 'both'):
        print(f"\n{'='*50}")
        print(f"DENDRITIC RWKV on {args.rule}")
        print(f"{'='*50}")
        m = RWKVNanoDendritic(
            vocab_size=len(VOCAB), dim=args.dim, num_layers=args.layers,
            pad_token_id=PAD_ID,
        )
        print(f"Params: {count_params(m):,}")
        results['dendritic'] = train_one(
            m, args.rule, device, args.steps, args.batch_size,
            args.max_len, args.lr, args.seed
        )

    # Summary
    if args.model == 'both':
        print(f"\n{'='*50}")
        print(f"SUMMARY on {args.rule}")
        print(f"{'='*50}")
        for name, res in results.items():
            print(f"  {name:12s}: acc={res['final_acc']:.3f}  "
                  f"step={res['avg_step_ms']:.1f}ms  params={count_params(m):,}")

    (exp_dir / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()