"""Train the g1g loopy channel-mix replacement on a synthetic byte task.

Frozen g1g 2.9B backbone with loopy byte-level channel-mix replacements
in the first N layers. Only the LoopyChannelMix parameters are trained.

Usage:
    PYTHONPATH=. python src/train_g1g_channel_loop.py --exp_id cl_smoke_001 \\
        --steps 100 --batch_size 4 --lr 1e-4 \\
        --layers_to_replace 0 1 2 --n_bytes 16 --byte_dim 32
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.byte_vocab import encode, decode, VOCAB_SIZE, PAD_ID
from src.logic_niiah_generator import LogicNiiahGenerator
from src.g1g_channel_loop import (
    G1GWithLoopyChannel, DEFAULT_MODEL_PATH,
    BYTE_VOCAB_SIZE, BYTE_PAD, BYTE_TO_ID,
)


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _text_to_byte_ids(text: str, max_len: int = 512) -> list[int]:
    """Encode text to actual UTF-8 byte IDs (258 vocab)."""
    raw = text.encode("utf-8")
    ids = [BYTE_TO_ID[b] for b in raw]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [BYTE_PAD] * (max_len - len(ids))
    return ids


def _char_span_to_byte_span(text: str, char_span: tuple[int, int]) -> tuple[int, int]:
    """Convert character-offset span to byte-offset span for a UTF-8 string."""
    raw = text.encode("utf-8")
    byte_positions = [0]
    for ch in text:
        byte_positions.append(byte_positions[-1] + len(ch.encode("utf-8")))
    s = byte_positions[char_span[0]] if char_span[0] < len(byte_positions) else len(raw)
    e = byte_positions[char_span[1]] if char_span[1] < len(byte_positions) else len(raw)
    return (s, e)


def byte_example_to_tensor(
    text: str,
    answer_spans: list[tuple[int, int]],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert generator example to (byte_ids, loss_mask) using UTF-8 byte encoding."""
    tokens = _text_to_byte_ids(text, max_len=max_len + 1)
    mask = torch.zeros(len(tokens), dtype=torch.float)
    for cs, ce in answer_spans:
        bs, be = _char_span_to_byte_span(text, (cs, ce))
        for t in range(max(0, bs - 1), min(len(tokens) - 1, be - 1)):
            mask[t] = 1.0
    return torch.tensor(tokens, dtype=torch.long), mask


def main():
    ap = argparse.ArgumentParser(description="Train g1g with loopy channel mixes")
    ap.add_argument("--exp_id", default="cl_smoke_001")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--layers_to_replace", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_bytes", type=int, default=16)
    ap.add_argument("--byte_dim", type=int, default=64)
    ap.add_argument("--expansion_factor", type=float, default=1.0)
    ap.add_argument("--max_loops", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_vars", type=int, default=3)
    ap.add_argument("--min_transforms", type=int, default=2)
    ap.add_argument("--max_transforms", type=int, default=4)
    ap.add_argument("--noise_min", type=int, default=1)
    ap.add_argument("--noise_max", type=int, default=2)
    args = ap.parse_args()

    device = torch.device(args.device)
    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["git_hash"] = get_git_hash()
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Config: {exp_dir / 'config.json'}")
    print(f"  git: {config['git_hash']}")

    # Generator
    generator = LogicNiiahGenerator(seed=args.seed)
    gen_kwargs = {
        "num_vars": args.num_vars,
        "min_transforms": args.min_transforms,
        "max_transforms": args.max_transforms,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
    }

    # Model
    print("Loading G1GWithLoopyChannel...")
    t0 = time.time()
    model = G1GWithLoopyChannel(
        model_path=DEFAULT_MODEL_PATH,
        layers_to_replace=args.layers_to_replace,
        n_bytes=args.n_bytes,
        byte_dim=args.byte_dim,
        expansion_factor=args.expansion_factor,
        max_loops=args.max_loops,
    ).to(device)
    model.train()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    trainable = sum(p.numel() for p in model.loopy_channels.parameters())
    print(f"  Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"  Layers replaced: {args.layers_to_replace}")

    optimizer = torch.optim.AdamW(
        model.get_trainable_params(), lr=args.lr, weight_decay=0.01,
    )

    # Training loop
    t_start = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        # Generate batch
        examples = generator.generate_batch(args.batch_size, **gen_kwargs)
        batch_ids, batch_masks = [], []
        for ex in examples:
            ids, mask = byte_example_to_tensor(
                ex["text"], ex["answer_spans"], args.max_len,
            )
            batch_ids.append(ids)
            batch_masks.append(mask)

        input_ids = torch.stack(batch_ids).to(device)
        mask = torch.stack(batch_masks).to(device)
        targets = torch.roll(input_ids, shifts=-1, dims=1)
        targets[:, -1] = BYTE_PAD

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none",
        )
        loss = loss.view_as(mask)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            recent = sum(losses[-min(len(losses), args.log_every):]) / min(len(losses), args.log_every)
            print(f"  step {step:4d}/{args.steps}  loss={recent:.4f}  {sps:.1f} st/s")

    elapsed = time.time() - t_start
    result = {
        "final_loss": losses[-1] if losses else None,
        "avg_loss_last_10": sum(losses[-10:]) / min(len(losses), 10) if losses else None,
        "total_steps": args.steps,
        "elapsed_s": round(elapsed, 1),
        "trainable_params": trainable,
        "git_hash": config["git_hash"],
    }
    (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"\nDone. Trainable: {trainable:,} params")
    print(f"Final loss: {result.get('final_loss', 'N/A')}")
    print(f"Results saved to: {exp_dir}/")


if __name__ == "__main__":
    main()
