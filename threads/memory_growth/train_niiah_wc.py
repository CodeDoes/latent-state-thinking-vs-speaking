#!/usr/bin/env python3
"""Train Standard vs Dendritic RWKV on Logic-NIIAH + Word Count.

Standard: trains on Logic-NIIAH only.
Dendritic: trains on Logic-NIIAH + Word Count task.


[meta]
status: triage-needed
[/meta]
"""

import argparse
import json
import subprocess
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from threads.memory_growth.logic_niiah_generator import LogicNiiahGenerator
from domains.rwkv.rwkv_nano import RWKVNano, count_params
from threads.memory_growth.rwkv_dendritic import RWKVNanoDendritic


# ── Character tokenizer ──────────────────────────────────────────────────

CHARS = [
    '\n', ' ', '!', ',', '-', '.', ':', '=',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
]
SPECIAL = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
VOCAB = SPECIAL + CHARS
char_to_id = {c: i for i, c in enumerate(VOCAB)}
id_to_char = {i: c for c, i in char_to_id.items()}
PAD_ID = char_to_id['<PAD>']
UNK_ID = char_to_id['<UNK>']
EOS_ID = char_to_id['<EOS>']


def encode(text: str):
    return [char_to_id.get(c, UNK_ID) for c in text]


def get_git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


# ── Logic-NIIAH generator ────────────────────────────────────────────────

def gen_logic_niiah(batch_size, max_len, seed, **gen_kwargs):
    gen = LogicNiiahGenerator(seed=seed)
    input_ids, targets, masks = [], [], []

    for _ in range(batch_size):
        ex = gen.generate(**gen_kwargs)
        tokens = encode(ex['text'])
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if len(tokens) < 2:
            continue

        mask = torch.zeros(max_len)
        for s, e in ex['answer_spans']:
            if s >= max_len:
                continue
            for t in range(max(s - 1, 0), min(e - 1, max_len - 1)):
                mask[t] = 1.0

        inp = tokens
        tgt = tokens[1:] + [PAD_ID]
        pad = max_len - len(inp)
        if pad > 0:
            inp = inp + [PAD_ID] * pad
            tgt = tgt + [PAD_ID] * pad

        input_ids.append(inp)
        targets.append(tgt)
        masks.append(mask.tolist())

    if not input_ids:
        return None, None, None

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.stack([torch.tensor(m) for m in masks]),
    )


# ── Word Count generator ────────────────────────────────────────────────

def gen_word_count(batch_size, max_len, seed):
    random.seed(seed)
    words = [
        'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
        'cat', 'bird', 'fish', 'tree', 'house', 'car', 'book', 'pen',
        'apple', 'orange', 'banana', 'grape', 'melon', 'pear', 'peach',
        'red', 'blue', 'green', 'yellow', 'black', 'white', 'big', 'small',
    ]

    input_ids, targets, masks = [], [], []

    for _ in range(batch_size):
        n_words = random.randint(5, 30)
        sentence = ' '.join(random.choices(words, k=n_words))
        count_str = str(n_words)
        text = f"count words: {sentence} <EOS> {count_str}"
        tokens = encode(text)

        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if len(tokens) < 3:
            continue

        mask = torch.zeros(max_len)
        eos_pos = None
        for i, t in enumerate(tokens):
            if t == EOS_ID:
                eos_pos = i
                break

        if eos_pos is not None and eos_pos + 1 < len(tokens):
            for t in range(eos_pos, min(len(tokens) - 1, max_len - 1)):
                mask[t] = 1.0

        inp = tokens
        tgt = tokens[1:] + [PAD_ID]
        pad = max_len - len(inp)
        if pad > 0:
            inp = inp + [PAD_ID] * pad
            tgt = tgt + [PAD_ID] * pad

        input_ids.append(inp)
        targets.append(tgt)
        masks.append(mask.tolist())

    if not input_ids:
        return None, None, None

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.stack([torch.tensor(m) for m in masks]),
    )


# ── Evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, gen_fn, max_len, device, seed, num_batches=2, batch_size=32):
    model.eval()
    correct = masked = 0
    for b in range(num_batches):
        input_ids, targets, mask = gen_fn(batch_size, max_len, seed + b + 10000)
        if input_ids is None:
            continue
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        logits, _ = model(input_ids)
        pred = logits.argmax(-1)
        m = mask.bool()
        correct += ((pred == targets) & m).sum().item()
        masked += m.sum().item()

    model.train()
    return correct / masked if masked > 0 else 0.0


# ── Training ────────────────────────────────────────────────────────────

def train_one(
    model, name, device, steps, batch_size, max_len, lr, seed,
    gen_kwargs, extra_gen_fn=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device).train()

    best_logic = 0.0
    best_wc = 0.0
    step_times = []

    for step in range(1, steps + 1):
        t0 = time.time()

        # Logic-NIIAH
        input_ids, targets, mask = gen_logic_niiah(
            batch_size, max_len, seed * 1000 + step, **gen_kwargs)

        loss = 0.0
        if input_ids is not None:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            logits, _ = model(input_ids)
            l = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            l = l.view_as(mask)
            loss = (l * mask).sum() / (mask.sum() + 1e-8)

            # Extra task for dendritic
            if extra_gen_fn is not None:
                input_ids2, targets2, mask2 = extra_gen_fn(
                    batch_size, max_len, seed * 1000 + step + 5000)
                if input_ids2 is not None:
                    input_ids2 = input_ids2.to(device)
                    targets2 = targets2.to(device)
                    mask2 = mask2.to(device)

                    logits2, _ = model(input_ids2)
                    l2 = F.cross_entropy(
                        logits2.view(-1, logits2.size(-1)), targets2.view(-1), reduction='none')
                    l2 = l2.view_as(mask2)
                    loss2 = (l2 * mask2).sum() / (mask2.sum() + 1e-8)
                    loss = loss + loss2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step_times.append(time.time() - t0)

        # Eval
        if step % 100 == 0 or step == 1 or step == steps:
            logic_acc = evaluate(model, lambda b, m, s: gen_logic_niiah(b, m, s, **gen_kwargs),
                                 max_len, device, seed, num_batches=2)
            wc_acc = 0.0
            if extra_gen_fn:
                wc_acc = evaluate(model, extra_gen_fn, max_len, device, seed, num_batches=2)

            if logic_acc > best_logic:
                best_logic = logic_acc
            if wc_acc > best_wc:
                best_wc = wc_acc

            avg_ms = sum(step_times[-100:]) / max(1, len(step_times[-100:])) * 1000
            loss_str = f"{loss.item():.4f}" if isinstance(loss, torch.Tensor) else "0.0000"
            print(
                f"  [{name}] step {step:5d}/{steps}  loss={loss_str}  "
                f"logic_acc={logic_acc:.3f}  wc_acc={wc_acc:.3f}  "
                f"best_logic={best_logic:.3f}  best_wc={best_wc:.3f}  "
                f"{avg_ms:.1f}ms/step",
                flush=True,
            )

    final_logic = evaluate(model, lambda b, m, s: gen_logic_niiah(b, m, s, **gen_kwargs),
                           max_len, device, seed, num_batches=4)
    final_wc = evaluate(model, extra_gen_fn, max_len, device, seed, num_batches=4) if extra_gen_fn else 0.0

    return {
        'final_logic_acc': final_logic,
        'best_logic_acc': best_logic,
        'final_wc_acc': final_wc,
        'best_wc_acc': best_wc,
        'avg_step_ms': sum(step_times) / len(step_times) * 1000,
        'params': count_params(model),
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['standard', 'dendritic', 'both'], default='both')
    ap.add_argument('--exp_id', default='niiah_wc')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')

    # Logic-NIIAH config
    ap.add_argument('--num_vars', type=int, default=2)
    ap.add_argument('--min_transforms', type=int, default=1)
    ap.add_argument('--max_transforms', type=int, default=2)
    ap.add_argument('--noise_min', type=int, default=0)
    ap.add_argument('--noise_max', type=int, default=1)

    args = ap.parse_args()

    device = torch.device(args.device)
    exp_dir = Path('experiments') / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['git_hash'] = get_git_hash()
    config['vocab_size'] = len(VOCAB)
    (exp_dir / 'config.json').write_text(json.dumps(config, indent=2))

    gen_kwargs = {
        'num_vars': args.num_vars,
        'min_transforms': args.min_transforms,
        'max_transforms': args.max_transforms,
        'noise_min': args.noise_min,
        'noise_max': args.noise_max,
    }

    print(f"Experiment: {args.exp_id}")
    print(f"Git: {config['git_hash']}")
    print(f"Vocab: {len(VOCAB)} | Dim: {args.dim} | Layers: {args.layers}")
    print(f"Logic-NIIAH: {gen_kwargs}")

    results = {}

    if args.model in ('standard', 'both'):
        print(f"\n{'='*60}")
        print(f"STANDARD RWKV (Logic-NIIAH only)")
        print(f"{'='*60}")
        model = RWKVNano(vocab_size=len(VOCAB), dim=args.dim, num_layers=args.layers, pad_token_id=PAD_ID)
        print(f"Params: {count_params(model):,}")
        results['standard'] = train_one(
            model, 'standard', device, args.steps, args.batch_size,
            args.max_len, args.lr, args.seed, gen_kwargs, extra_gen_fn=None,
        )

    if args.model in ('dendritic', 'both'):
        print(f"\n{'='*60}")
        print(f"DENDRITIC RWKV (Logic-NIIAH + Word Count)")
        print(f"{'='*60}")
        model = RWKVNanoDendritic(vocab_size=len(VOCAB), dim=args.dim, num_layers=args.layers, pad_token_id=PAD_ID)
        print(f"Params: {count_params(model):,}")
        results['dendritic'] = train_one(
            model, 'dendritic', device, args.steps, args.batch_size,
            args.max_len, args.lr, args.seed, gen_kwargs, extra_gen_fn=gen_word_count,
        )

    if args.model == 'both':
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        for name, res in results.items():
            print(f"  {name:12s}: logic={res['final_logic_acc']:.3f} (best={res['best_logic_acc']:.3f})  "
                  f"wc={res['final_wc_acc']:.3f}  params={res['params']:,}  step={res['avg_step_ms']:.1f}ms")

    (exp_dir / 'results.json').write_text(json.dumps(results, indent=2))
    print(f"\nSaved to: {exp_dir}/results.json")


if __name__ == '__main__':
    main()