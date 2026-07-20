#!/usr/bin/env python3
"""Compare Standard RWKV vs Dendritic RWKV on Logic NIIAH task.

[meta]
status: triage-needed
[/meta]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from threads.memory_growth.logic_niiah_generator import LogicNiiahGenerator
from domains.rwkv.rwkv_nano import RWKVNano, count_params
from threads.memory_growth.rwkv_dendritic import RWKVNanoDendritic


# ── Character-level tokenizer ─────────────────────────────────────────────

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


def encode(text: str) -> list[int]:
    return [char_to_id.get(c, UNK_ID) for c in text]


def decode(ids: list[int]) -> str:
    return ''.join(id_to_char.get(i, '<UNK>') for i in ids)


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


# ── Data preparation ──────────────────────────────────────────────────────

def example_to_tensor(
    text: str,
    answer_spans: list[tuple[int, int]],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = encode(text)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [PAD_ID] * (max_len - len(tokens))

    input_ids = torch.tensor(tokens, dtype=torch.long)

    mask = torch.zeros(max_len, dtype=torch.float)
    for s, e in answer_spans:
        for t in range(max(0, s - 1), min(max_len - 1, e - 1)):
            mask[t] = 1.0

    return input_ids, mask


def generate_batch_tensors(
    generator: LogicNiiahGenerator,
    batch_size: int,
    max_len: int,
    gen_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = generator.generate_batch(batch_size, **gen_kwargs)
    batch_ids = []
    batch_masks = []
    for ex in examples:
        ids, mask = example_to_tensor(ex['text'], ex['answer_spans'], max_len)
        batch_ids.append(ids)
        batch_masks.append(mask)

    input_ids = torch.stack(batch_ids)
    mask = torch.stack(batch_masks)

    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = PAD_ID

    return input_ids, targets, mask


# ── Checkpointing ─────────────────────────────────────────────────────────

def save_checkpoint(
    exp_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict,
    generator_rng_state,
):
    ckpt = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': step,
        'metrics': metrics,
        'generator_rng_state': generator_rng_state,
        'config': {
            'vocab_size': len(VOCAB),
            'dim': model.dim,
            'num_layers': model.num_layers,
        },
    }
    torch.save(ckpt, exp_dir / 'checkpoint.pt')
    torch.save(ckpt, exp_dir / f'checkpoint_step_{step}.pt')
    ckpts = sorted(exp_dir.glob('checkpoint_step_*.pt'))
    for f in ckpts[:-3]:
        f.unlink()


def load_checkpoint(
    exp_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, dict, object]:
    ckpt = torch.load(exp_dir / 'checkpoint.pt', map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    step = ckpt['step']
    metrics = ckpt.get('metrics', {})
    rng_state = ckpt.get('generator_rng_state')
    return step, metrics, rng_state


# ── Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    generator: LogicNiiahGenerator,
    max_len: int,
    gen_kwargs: dict,
    num_examples: int = 64,
    device: torch.device = torch.device('cpu'),
) -> dict:
    model.eval()
    exact_correct = 0
    total_answers = 0
    digit_correct = 0
    digit_total = 0

    generator.reseed(9999)
    examples = generator.generate_batch(num_examples, **gen_kwargs)

    for ex in examples:
        tokens = encode(ex['text'])
        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        input_t = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, _ = model(input_t)

        for s, e in ex['answer_spans']:
            pred_chars = []
            target_chars = []
            for ti in range(s, min(e, max_len)):
                if ti > 0 and ti < len(tokens):
                    pred_t = logits[0, ti - 1].argmax().item()
                    pred_chars.append(id_to_char.get(pred_t, '<ERR>'))
                    target_chars.append(ex['text'][ti])
                    digit_total += 1
                    if id_to_char.get(pred_t, '<ERR>') == ex['text'][ti]:
                        digit_correct += 1

            pred_str = ''.join(pred_chars)
            target_str = ''.join(target_chars)
            total_answers += 1
            if pred_str == target_str:
                exact_correct += 1

    model.train()
    exact_acc = exact_correct / total_answers if total_answers > 0 else 0.0
    digit_acc = digit_correct / digit_total if digit_total > 0 else 0.0
    return {
        'accuracy': exact_acc,
        'digit_acc': digit_acc,
        'exact_correct': exact_correct,
        'total_answers': total_answers,
        'digit_correct': digit_correct,
        'digit_total': digit_total,
    }


# ── Training ──────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    exp_dir: Path,
    args,
    generator: LogicNiiahGenerator,
    gen_kwargs: dict,
    device: torch.device,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model params: {count_params(model):,}")

    start_step = 1
    best_acc = 0.0
    metrics_log = []

    if args.resume and (exp_dir / 'checkpoint.pt').exists():
        start_step, old_metrics, rng_state = load_checkpoint(
            exp_dir, model, optimizer, device)
        generator.reseed(args.seed)
        for _ in range(start_step * args.batch_size):
            generator.generate(**gen_kwargs)
        best_acc = old_metrics.get('best_acc', 0.0)
        metrics_log = old_metrics.get('log', [])
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
            generator, args.batch_size, args.max_len, gen_kwargs)
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='none',
        )
        loss = loss.view_as(mask)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            sps = (step - start_step + 1) / max(elapsed, 1e-6)
            print(
                f"step {step:6d}/{args.steps}  "
                f"loss={loss.item():.4f}  "
                f"speed={sps:.1f} st/s  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        if step % args.eval_every == 0 or step == 1:
            eval_metrics = evaluate(
                model, generator, args.max_len, gen_kwargs,
                num_examples=32, device=device,
            )
            acc = eval_metrics['accuracy']
            if acc > best_acc:
                best_acc = acc
            print(
                f"  eval: exact_acc={acc:.3f}  "
                f"digit_acc={eval_metrics['digit_acc']:.3f}  "
                f"({eval_metrics['exact_correct']}/{eval_metrics['total_answers']} ans, "
                f"{eval_metrics['digit_correct']}/{eval_metrics['digit_total']} dig)  "
                f"best={best_acc:.3f}",
                flush=True,
            )

            gen_sample = generator.generate(**gen_kwargs)
            input_t = torch.tensor(
                [encode(gen_sample['text'][:args.max_len])], dtype=torch.long,
                device=device,
            )
            model.eval()
            with torch.no_grad():
                logits, _ = model(input_t)
                pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
                pred_text = decode(pred_ids)
            model.train()

            sample_log = {
                'step': step,
                'accuracy': acc,
                'best_acc': best_acc,
                'loss': loss.item(),
                'sample_input': gen_sample['text'][:300],
                'sample_prediction': pred_text[:300],
                'sample_answers': gen_sample['answers'],
            }
            metrics_log.append(sample_log)
            (exp_dir / 'samples.json').write_text(
                json.dumps(metrics_log[-10:], indent=2))

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                exp_dir, model, optimizer, step,
                {'best_acc': best_acc, 'log': metrics_log},
                generator.rng.getstate(),
            )
            print(f"  checkpoint saved at step {step}", flush=True)

        step += 1

    elapsed = time.time() - t_start
    final_eval = evaluate(
        model, generator, args.max_len, gen_kwargs,
        num_examples=64, device=device,
    )
    result = {
        'final_accuracy': final_eval['accuracy'],
        'best_accuracy': best_acc,
        'total_steps': args.steps,
        'elapsed_s': round(elapsed, 1),
        'params': count_params(model),
        'git_hash': get_git_hash(),
    }
    (exp_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(f"\n{'='*50}")
    print(f"Done! Final accuracy: {final_eval['accuracy']:.3f}")
    print(f"Best accuracy:       {best_acc:.3f}")
    print(f"Elapsed:             {elapsed:.1f}s")
    print(f"Results saved to:    {exp_dir}/")
    print(f"{'='*50}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Compare Standard RWKV vs Dendritic RWKV on Logic NIIAH')
    ap.add_argument('--model', choices=['standard', 'dendritic', 'both'],
                    default='both', help='Which model(s) to train')
    ap.add_argument('--exp_id', default='compare', help='experiment directory name')
    ap.add_argument('--steps', type=int, default=3000, help='total training steps')
    ap.add_argument('--batch_size', type=int, default=8, help='examples per step')
    ap.add_argument('--max_len', type=int, default=512, help='max sequence length')
    ap.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    ap.add_argument('--dim', type=int, default=128, help='RWKV embedding dimension')
    ap.add_argument('--layers', type=int, default=3, help='number of RWKV layers')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')

    # Generator difficulty
    ap.add_argument('--num_vars', type=int, default=3)
    ap.add_argument('--min_transforms', type=int, default=2)
    ap.add_argument('--max_transforms', type=int, default=5)
    ap.add_argument('--noise_min', type=int, default=1)
    ap.add_argument('--noise_max', type=int, default=3)

    # Dendritic specific
    ap.add_argument('--n_active', type=int, default=None,
                    help='Number of active units per step (default: 10% of hidden)')
    ap.add_argument('--duty_decay', type=float, default=0.99)
    ap.add_argument('--boost_strength', type=float, default=1.0)

    args = ap.parse_args()

    device = torch.device(args.device)
    base_dir = Path('experiments') / args.exp_id
    base_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['git_hash'] = get_git_hash()
    config['vocab_size'] = len(VOCAB)
    (base_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f"Config: {base_dir / 'config.json'}")
    print(f"  git hash: {config['git_hash']}")

    generator = LogicNiiahGenerator(seed=args.seed)
    gen_kwargs = {
        'num_vars': args.num_vars,
        'min_transforms': args.min_transforms,
        'max_transforms': args.max_transforms,
        'noise_min': args.noise_min,
        'noise_max': args.noise_max,
    }

    results = {}

    # Train standard RWKV
    if args.model in ('standard', 'both'):
        print(f"\n{'='*60}")
        print(f"Training STANDARD RWKV")
        print(f"{'='*60}\n")
        std_dir = base_dir / 'standard'
        std_dir.mkdir(parents=True, exist_ok=True)

        model = RWKVNano(
            vocab_size=len(VOCAB),
            dim=args.dim,
            num_layers=args.layers,
            pad_token_id=PAD_ID,
        ).to(device)

        results['standard'] = train_model(
            model, std_dir, args, generator, gen_kwargs, device)

    # Train dendritic RWKV
    if args.model in ('dendritic', 'both'):
        print(f"\n{'='*60}")
        print(f"Training DENDRITIC RWKV")
        print(f"{'='*60}\n")
        den_dir = base_dir / 'dendritic'
        den_dir.mkdir(parents=True, exist_ok=True)

        model = RWKVNanoDendritic(
            vocab_size=len(VOCAB),
            dim=args.dim,
            num_layers=args.layers,
            n_active=args.n_active,
            duty_cycle_momentum=args.duty_decay,
            boost_strength=args.boost_strength,
        ).to(device)

        results['dendritic'] = train_model(
            model, den_dir, args, generator, gen_kwargs, device)

    # Summary
    if args.model == 'both':
        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        for name, res in results.items():
            print(f"  {name:12s}: final={res['final_accuracy']:.3f}  "
                  f"best={res['best_accuracy']:.3f}  "
                  f"params={res['params']:,}  "
                  f"time={res['elapsed_s']:.1f}s")

        (base_dir / 'comparison.json').write_text(json.dumps(results, indent=2))
        print(f"\nSaved to: {base_dir}/comparison.json")


if __name__ == '__main__':
    main()