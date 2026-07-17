"""Training script for DendriteRWKV - one adapter per synthetic rule."""

import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.dendrite_rwkv import (
    DendriteRWKV,
    count_base_params,
    count_lora_params,
    apply_lora_to_rwkv,
    train_adapter,
    gen_sum_threshold,
    gen_vowel_majority,
    gen_endpoint_match,
    gen_count_trigger,
)


# ── Config ─────────────────────────────────────────────────────────────

VOCAB_SIZE = 128
DIM = 128
NUM_LAYERS = 3
HIDDEN_SCALE = 4
LORA_RANK = 8
LORA_ALPHA = 16

RULES = [
    {'name': 'sum_threshold', 'gen': gen_sum_threshold, 'args': (1000, 200), 'val_args': (200, 200)},
    {'name': 'vowel_majority', 'gen': gen_vowel_majority, 'args': (1000,), 'val_args': (200,)},
    {'name': 'endpoint_match', 'gen': gen_endpoint_match, 'args': (1000,), 'val_args': (200,)},
    {'name': 'count_trigger', 'gen': gen_count_trigger, 'args': (1000,), 'val_args': (200,)},
]

STEPS_PER_ADAPTER = 500
LR = 3e-4
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EXP_DIR = Path("experiments/dendrite_rwkv_001")
EXP_DIR.mkdir(parents=True, exist_ok=True)


def collate_fn(batch):
    """Pad sequences to max length in batch."""
    max_len = max(len(x) for x, _ in batch)
    padded_x = []
    labels = []
    for x, y in batch:
        pad_len = max_len - len(x)
        padded = x + [0] * pad_len
        padded_x.append(padded)
        labels.append(y)
    return torch.tensor(padded_x, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def prepare_data(gen_fn, args) -> Tuple[DataLoader, DataLoader]:
    """Generate data and create loaders."""
    data = gen_fn(*args)
    # Split into sequences and labels
    sequences = [seq for seq, _ in data]
    labels = [label for _, label in data]
    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


def main():
    print(f"Device: {DEVICE}")
    print(f"Experiment dir: {EXP_DIR}")

    # Create model
    configs = [{'name': r['name']} for r in RULES]
    model = DendriteRWKV(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=NUM_LAYERS,
        hidden_scale=HIDDEN_SCALE,
        adapter_configs=configs,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
    ).to(DEVICE)

    print(f"Backbone (frozen): {count_base_params(model.backbone):,}")
    for name in model.adapter_names:
        print(f"Adapter {name} LoRA: {count_lora_params(model.adapters[name]):,}")

    # Save config
    config = {
        'vocab_size': VOCAB_SIZE,
        'dim': DIM,
        'num_layers': NUM_LAYERS,
        'hidden_scale': HIDDEN_SCALE,
        'lora_rank': LORA_RANK,
        'lora_alpha': LORA_ALPHA,
        'rules': [r['name'] for r in RULES],
        'steps_per_adapter': STEPS_PER_ADAPTER,
        'lr': LR,
        'batch_size': BATCH_SIZE,
    }
    with open(EXP_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Train each adapter
    results = {}
    for rule in RULES:
        name = rule['name']
        print(f"\n{'='*50}")
        print(f"Training adapter: {name}")
        print(f"{'='*50}")

        train_loader = prepare_data(rule['gen'], rule['args'])
        val_loader = prepare_data(rule['gen'], rule['val_args'])

        result = train_adapter(
            model, name,
            train_loader, val_loader,
            steps=STEPS_PER_ADAPTER,
            lr=LR,
            device=DEVICE,
        )
        results[name] = result
        print(f"Result: {result}")

        # Save adapter weights
        adapter = model.adapters[name]
        torch.save(adapter.state_dict(), EXP_DIR / f"adapter_{name}.pt")

    # Save results
    with open(EXP_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results: {results}")


if __name__ == "__main__":
    main()