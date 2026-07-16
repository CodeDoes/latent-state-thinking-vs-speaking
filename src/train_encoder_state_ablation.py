"""Train encoder state ablation: 7 variants, which state components matter?

Each variant uses a different combination of state components as encoder input:
1. static-patch-state
2. encoder-state
3. static-patch-state + encoder-state
4. byte-level-state
5. static-patch-state + byte-level-state
6. mutable-full-state
7. static-full-state

All trained on the same data with the same hyperparams. Final loss
compared to determine which state components provide useful context.
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

from src.encoder_state_ablation import EncoderVariant, StateBuffer, PatchModel
from src.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, ID_TO_BYTE, UNK_ID


def load_text(path: Path) -> list[int]:
    text = path.read_bytes().decode('utf-8', errors='replace')
    return [BYTE_TO_ID.get(ord(c), UNK_ID) for c in text]


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int = 50) -> float:
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


def train_one_variant(state_comps, exp_dir, args, device):
    """Train one encoder variant and return final loss."""
    print(f"\n{'='*60}")
    print(f"Training: {'+'.join(state_comps)}")
    print(f"{'='*60}")
    
    # Load data
    text_path = Path(args.text)
    stream = load_text(text_path)
    
    # Build model
    dim = args.dim
    state_buffer = StateBuffer(dim, device=device)
    patch_model = PatchModel(dim).to(device)
    model = EncoderVariant(dim, state_comps, VOCAB_SIZE).to(device)
    
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in patch_model.parameters())
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(patch_model.parameters()),
        lr=args.lr
    )
    
    # Create batches (fixed streams for determinism)
    batch_size = args.batch_size
    max_len = args.max_len
    n = (len(stream) - 1) // max_len * max_len
    stream = stream[:n + 1]
    
    def get_batch():
        start = (step * batch_size * max_len) % (len(stream) - max_len - 1)
        rows = []
        for i in range(batch_size):
            chunk = stream[start + i * max_len : start + (i + 1) * max_len + 1]
            if len(chunk) < max_len + 1:
                chunk = chunk + [PAD_ID] * (max_len + 1 - len(chunk))
            rows.append(chunk)
        batch = torch.tensor(rows, dtype=torch.long, device=device)
        return batch[:, :-1], batch[:, 1:].contiguous()
    
    # Save config
    config = {
        "model": "encoder_state_ablation",
        "variant": '+'.join(state_comps),
        "git_hash": get_git_hash(),
        "n_bytes": len(stream),
        "steps": args.steps,
        "batch_size": batch_size,
        "max_len": max_len,
        "dim": dim,
        "lr": args.lr,
        "lr_schedule": "cosine",
        "n_params": n_params,
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    # Training loop
    model.train()
    patch_model.train()
    losses = []
    metrics = []
    t0 = time.time()
    step = -1
    
    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        input_ids, targets = get_batch()
        
        # Process sequence with sliding encoder
        # For each position, use encoder + states to predict
        total_loss = 0
        n_positions = 0
        
        for pos in range(max_len):
            byte_input = input_ids[:, pos]  # [batch]
            target = targets[:, pos]  # [batch]
            
            logits, surprise, should_patch = model(
                byte_input, state_buffer, patch_model, return_surprise=True
            )
            
            loss = F.cross_entropy(logits, target, ignore_index=PAD_ID)
            total_loss = total_loss + loss
            n_positions += 1
            
            # If surprising, update patch model
            if should_patch:
                with torch.no_grad():
                    new_state = patch_model(state_buffer.patch_state)
                    state_buffer.patch_state = new_state.detach()
            
            # Update byte-level state for variants that use it
            if 'byte-level-state' in state_comps:
                old_state = state_buffer.byte_level_state
                new_state = (old_state + model.embed(byte_input).mean(dim=0).detach()) / 2
                state_buffer.byte_level_state = new_state.detach()
        
        avg_loss = total_loss / n_positions
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        loss_val = avg_loss.item()
        losses.append(loss_val)
        
        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            avg = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
            print(f"  step {step:4d} | loss {loss_val:.4f} (avg {avg:.4f}) | lr {lr:.6f} | {elapsed:.1f}s")
            metrics.append({"step": step, "loss": loss_val, "avg_loss": avg, "lr": lr})
    
    # Save results
    torch.save(model.state_dict(), exp_dir / "checkpoint.pt")
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    
    final_avg = sum(losses[-50:]) / min(len(losses), 50)
    print(f"  Final avg loss: {final_avg:.4f}")
    return final_avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default='experiments/byte_ts_001/text.txt')
    ap.add_argument('--exp_id', default='encoder_state_ablation_001')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--log-every', type=int, default=20)
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Variants
    variants = [
        ['static-patch-state'],
        ['encoder-state'],
        ['static-patch-state', 'encoder-state'],
        ['byte-level-state'],
        ['static-patch-state', 'byte-level-state'],
        ['mutable-full-state'],
        ['static-full-state'],
    ]
    
    base_dir = Path(f"experiments/{args.exp_id}")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for state_comps in variants:
        variant_name = '+'.join(state_comps).replace('-', '_')
        exp_dir = base_dir / variant_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Re-init global seed for determinism
        torch.manual_seed(42)
        final_loss = train_one_variant(state_comps, exp_dir, args, device)
        results['+'.join(state_comps)] = final_loss
    
    # Summary
    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print(f"{'='*60}")
    for comps, loss in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {comps:50s} → {loss:.4f}")
    
    # Save summary
    summary = {
        "git_hash": get_git_hash(),
        "results": results,
        "best": min(results, key=results.get),
        "worst": max(results, key=results.get),
    }
    (base_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest: {summary['best']} ({results[summary['best']]:.4f})")
    print(f"Worst: {summary['worst']} ({results[summary['worst']]:.4f})")


if __name__ == "__main__":
    main()
