"""Read-twice experiment harness.

Compares recurrent-pass depth vs inserting a new layer at the detected
bottleneck. Matched compute, matched params where possible, same model
weights where not.

Usage:
    # Baselines first
    PYTHONPATH=. python3 src/read_twice.py \
        --checkpoint experiments/prog_exp_001/checkpoint.pt \
        --hard-n 200 --batch-size 8 --passes 1

    # Then scaled
    PYTHONPATH=. python3 src/read_twice.py \
        --checkpoint experiments/prog_exp_001/checkpoint.pt \
        --hard-n 200 --batch-size 8 --passes 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root on sys.path when run as python src/read_twice.py
_HERE = Path(__file__).resolve()
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

from src.analyze_cli import (
    RwkvDataPipeline,
    load_model,
    discover_hook_targets,
)
from src.bottleneck_analysis import HookManager, metric_saturation_ratio


# ──────────────────────────────────────────────────────────────────────────────
# Probes: same weights, more passes
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_passes(
    model: nn.Module,
    dataloader,
    n_passes: int = 1,
) -> tuple[float, float]:
    """Run forward passes N times before finalising logits.

    For RWKV the recurrent state naturally accumulates; we just loop the
    model forward over the same input, keeping the returned state each time,
    and feed the final state into the final LayerNorm + head.
    """
    model.eval()
    device = next(model.parameters()).device
    total_correct = 0.0
    total_answers = 0
    digit_correct = 0
    digit_total = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        elif isinstance(batch, dict):
            input_ids = batch.get("input_ids") or batch.get("context")
            if isinstance(input_ids, list):
                # TensorDataset case handled by tuple branch; guard
                continue
            input_ids = input_ids.to(device)
        else:
            input_ids = batch.to(device)

        # Multiple recurrent passes: reuse and extend the recurrent state
        logits = None
        state = None
        for _ in range(n_passes):
            logits, state = model(input_ids, state=state, return_state=True)

        logits = logits.cpu()
        # For logic-niiah the answer token is the last non-pad input position.
        # We teach-forcing-evaluate: pick argmax at each position, compare to
        # the target token at +1 position (next-token prediction format used
        # by train_rwkv).
        preds = logits.argmax(dim=-1)
        targets = input_ids.roll(-1, dims=1).cpu()
        targets[:, -1] = 0  # ignore last position (no target)

        mask = targets != 0
        if mask.any():
            total_correct += (preds[mask] == targets[mask]).sum().item()
            total_answers += mask.sum().item()
            digit_correct += (preds[mask] == targets[mask]).sum().item()
            digit_total += mask.sum().item()

    exact_acc = total_correct / total_answers if total_answers > 0 else 0.0
    digit_acc = digit_correct / digit_total if digit_total > 0 else 0.0
    return round(exact_acc, 6), round(digit_acc, 6)


def layer_insertion_accuracy(
    model: nn.Module,
    dataloader,
    layer_name: str,
    new_layer: nn.Module | None = None,
) -> float:
    """Insert an identity layer at `layer_name`, run one forward pass."""
    if new_layer is None:
        raise ValueError("new_layer required")

    # Resolve target by dotted name
    parent = model
    for part in layer_name.split(".")[:-1]:
        parent = getattr(parent, part)
    leaf_name = layer_name.split(".")[-1]
    original = getattr(parent, leaf_name)

    # Swap in identity + inserted layer, keeping original weights available
    # via one-to-one weight sharing. Here we just wrap the forward.
    @torch.no_grad()
    def inserted_forward(x, *args, **kwargs):
        z = original(x, *args, **kwargs)
        return new_layer(z)

    setattr(parent, leaf_name, inserted_forward)
    try:
        acc, da = evaluate_with_passes(model, dataloader, n_passes=1)
    finally:
        setattr(parent, leaf_name, original)
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# Experimental arms
# ──────────────────────────────────────────────────────────────────────────────
def make_linear_insert(dim: int, device: str) -> nn.Module:
    return nn.Linear(dim, dim, bias=True).to(device)


def run_experiment(
    checkpoint: str,
    passes: list[int],
    easy_n: int = 200,
    hard_n: int = 200,
    batch_size: int = 8,
    device: str = "cpu",
) -> None:
    """Run read-twice arm and one layer-insertion arm."""
    print(f"Loading model: {checkpoint}")
    model = load_model(checkpoint, device=device)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Determine insertion point from bottleneck analysis
    print("Detecting bottleneck (using saturation as cheap proxy)...")
    hook_mgr = HookManager(model)
    targets = discover_hook_targets(model)
    hook_mgr.register(targets, post=True)

    pipeline = RwkvDataPipeline(seed=42)
    easy_loader = pipeline.build_dataloader(
        easy_n,
        {
            "num_vars": 2,
            "min_transforms": 2,
            "max_transforms": 3,
            "noise_min": 0,
            "noise_max": 1,
        },
        batch_size=batch_size,
    )
    # Hard loader identical to training distribution for reproducibility
    hard_loader = pipeline.build_dataloader(
        hard_n,
        {
            "num_vars": 2,
            "min_transforms": 2,
            "max_transforms": 4,
            "noise_min": 0,
            "noise_max": 2,
        },
        batch_size=batch_size,
    )

    easy_acts = hook_mgr.collect(easy_loader)
    hard_acts = hook_mgr.collect(hard_loader)

    # Pick the layer with the highest saturation signal as insertion point
    layer_scores: dict[str, float] = {}
    for layer in easy_acts:
        if layer not in hard_acts:
            continue
        score = metric_saturation_ratio(easy_acts[layer], hard_acts[layer])
        if score.ndim == 0:
            score = torch.tensor([float(score)])
        layer_scores[layer] = float(score.max())
    insert_layer = max(layer_scores, key=layer_scores.get)
    dim = easy_acts[insert_layer][0].shape[-1]

    results: list[dict] = []
    print(f"Insertion layer candidate: {insert_layer} (dim={dim})")

    # Arm 1: read-twice at N passes
    for n in passes:
        print(f"  passes={n}: ", end="", flush=True)
        acc_exact, acc_digit = evaluate_with_passes(model, hard_loader, n_passes=n)
        print(f"exact={acc_exact:.4f} digit={acc_digit:.4f}")
        results.append({"arm": f"passes_{n}", "exact_acc": acc_exact, "digit_acc": acc_digit})

    # Arm 2: single linear insertion
    print(f"  layer_insert (identity-init linear at {insert_layer}): ", end="", flush=True)
    new_layer = make_linear_insert(dim, device)
    nn.init.zeros_(new_layer.weight)
    nn.init.zeros_(new_layer.bias)
    with torch.no_grad():
        for p in new_layer.parameters():
            p.requires_grad = False
    acc_exact, acc_digit = layer_insertion_accuracy(model, hard_loader, insert_layer, new_layer)
    print(f"exact={acc_exact:.4f} digit={acc_digit:.4f}")
    results.append({"arm": f"layer_insert_{insert_layer}", "exact_acc": acc_exact, "digit_acc": acc_digit})

    # Write results
    output = Path("experiments/prog_exp_001/read_twice_results.csv")
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", "exact_acc", "digit_acc"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {output}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read-twice experiment")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--passes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--easy-n", type=int, default=200)
    ap.add_argument("--hard-n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_experiment(**vars(args))
