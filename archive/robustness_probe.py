"""Robustness probe: does the learned world model survive pruning / weight shaking?

Motivation (see FUTURE_RESEARCH.md + the "generalize past the classic program"
critique): our benchmarks are *within-program* -- data and targets both come from
the same deterministic generator, so a model could ace them by memorizing template
grammar rather than tracking entities. A fragile memorized solution collapses under
weight perturbation / pruning; a distributed structured representation degrades
gracefully (and pruning can even act as a regularizer). This script measures that.

Usage:
  python3 robustness_probe.py --device cpu --d_state 64 --epochs 12 --n_samples 500
"""
import argparse, json, copy, time
import torch
import torch.nn.functional as F
from src.tokenizer import CharTokenizer, build_tokenizer_from_dataset
from src.dataset import generate_dataset
from src.world_state import (WorldTrainConfig, train_world, run_world_qa,
                             NAME_TO_I, extract_query)


def clip_state(state, frac):
    """Magnitude pruning: zero the smallest |w| fraction of each weight matrix."""
    out = {}
    for k, v in state.items():
        if v.dim() >= 2 and "weight" in k and v.numel() > 0:
            flat = v.abs().flatten()
            kth = int(frac * flat.numel())
            if kth <= 0:
                out[k] = v.clone()
            else:
                thr = flat.kthvalue(max(1, kth)).values
                out[k] = v * (v.abs() >= thr)
        else:
            out[k] = v.clone()
    return out


def shake_state(state, sigma):
    """Add N(0, sigma) to every floating param (absolute shaking)."""
    out = {}
    for k, v in state.items():
        out[k] = v + torch.randn_like(v) * sigma if v.is_floating_point() else v.clone()
    return out


def reinit_state(state, frac):
    """Randomly reinitialize a fraction of each weight matrix's entries."""
    out = {}
    for k, v in state.items():
        if v.dim() >= 2 and "weight" in k and v.numel() > 0:
            mask = (torch.rand(v.shape) < frac)
            nv = v.clone()
            nv[mask] = torch.randn(mask.sum().item()) * 0.05
            out[k] = nv
        else:
            out[k] = v.clone()
    return out


def eval_acc(model, qa, tokenizer, device):
    _, acc, n = run_world_qa(model, qa, tokenizer, device)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d_state", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    qa = list(generate_dataset(n_samples=args.n_samples, seed=args.seed,
                               task_weights={"location": 1.0}))
    tokenizer = build_tokenizer_from_dataset(qa, max_vocab=256)

    cfg = WorldTrainConfig(d_state=args.d_state, d_model=args.d_state,
                           epochs=args.epochs, batch_size=64,
                           lr=1e-3, ans_w=1.0, field_w=1.0,
                           loc_tok_w=1.0, item_tok_w=1.0, seed=args.seed)
    model, _ = train_world(qa, tokenizer, device, cfg, verbose=False)
    base = eval_acc(model, qa, tokenizer, device)
    print(f"\nBASE location acc = {base:.3f}  (n={len(qa)}, d={args.d_state})\n")

    rows = [("intact", 0.0, base)]

    # Magnitude pruning sweep
    for frac in (0.25, 0.5, 0.75, 0.9):
        m = copy.deepcopy(model)
        m.load_state_dict(clip_state(model.state_dict(), frac))
        rows.append((f"prune {int(frac*100)}%", frac, eval_acc(m, qa, tokenizer, device)))

    # Weight shaking sweep (absolute Gaussian)
    for sigma in (1e-3, 5e-3, 1e-2, 5e-2, 1e-1):
        m = copy.deepcopy(model)
        m.load_state_dict(shake_state(model.state_dict(), sigma))
        rows.append((f"shake σ={sigma:g}", sigma, eval_acc(m, qa, tokenizer, device)))

    # Random reinitialization sweep
    for frac in (0.1, 0.3, 0.5):
        m = copy.deepcopy(model)
        m.load_state_dict(reinit_state(model.state_dict(), frac))
        rows.append((f"reinit {int(frac*100)}%", frac, eval_acc(m, qa, tokenizer, device)))

    print(f"{'perturbation':<16}{'param':>10}{'acc':>10}{'Δ vs base':>12}")
    print("-" * 50)
    for name, p, acc in rows:
        print(f"{name:<16}{p:>10.3f}{acc:>10.3f}{acc - base:>+12.3f}")

    out = {"base_acc": base, "rows": [{"name": n, "param": p, "acc": a} for n, p, a in rows]}
    with open("robustness_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote robustness_probe.json")


if __name__ == "__main__":
    main()
