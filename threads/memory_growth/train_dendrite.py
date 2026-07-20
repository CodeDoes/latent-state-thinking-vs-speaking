"""Training script for DendriteRWKV - one adapter per synthetic rule.

[meta]
status: triage-needed
[/meta]
"""

from __future__ import annotations

import json
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from threads.memory_growth.dendrite_model import (
    DendriteRWKV,
    count_base_params,
    count_lora_params,
    get_lora_params,
    AdapterRegistry,
    train_adapter,
)


# ── Synthetic Rules (matching Dendritron tasks) ──────────────────────

VOCAB_SIZE = 128
PAD_ID = 0
MAX_LEN = 64

# Binary classification: 0 = false, 1 = true
# Labels are appended as last token


def gen_sum_threshold(n_samples: int, threshold: int = 200) -> List[Tuple[List[int], int]]:
    """Sum of digits in sequence >= threshold?"""
    data = []
    for _ in range(n_samples):
        length = random.randint(8, 20)
        seq = [random.randint(1, 9) for _ in range(length)]
        total = sum(seq)
        label = 1 if total >= threshold else 0
        # Append label as last token (shift by +10 to avoid collision with digits)
        seq.append(label + 10)
        data.append((seq, label))
    return data


def gen_vowel_majority(n_samples: int) -> List[Tuple[List[int], int]]:
    """Vowels (a,e,i,o,u) > consonants?"""
    vowel_ids = {1, 5, 9, 15, 21}  # a,e,i,o,u in 1-26 mapping
    data = []
    for _ in range(n_samples):
        length = random.randint(8, 20)
        seq = [random.randint(1, 26) for _ in range(length)]
        vowels = sum(1 for c in seq if c in vowel_ids)
        label = 1 if vowels > len(seq) - vowels else 0
        seq.append(label + 10)
        data.append((seq, label))
    return data


def gen_endpoint_match(n_samples: int) -> List[Tuple[List[int], int]]:
    """First and last char match?"""
    data = []
    for _ in range(n_samples):
        length = random.randint(8, 20)
        seq = [random.randint(1, 26) for _ in range(length)]
        label = 1 if seq[0] == seq[-1] else 0
        seq.append(label + 10)
        data.append((seq, label))
    return data


def gen_count_trigger(n_samples: int, trigger_char: int = 7, count: int = 3) -> List[Tuple[List[int], int]]:
    """Does char 7 appear >= count times?"""
    data = []
    for _ in range(n_samples):
        length = random.randint(8, 20)
        seq = [random.randint(1, 26) for _ in range(length)]
        occurrences = sum(1 for c in seq if c == trigger_char)
        label = 1 if occurrences >= count else 0
        seq.append(label + 10)
        data.append((seq, label))
    return data


RULES = {
    'sum_threshold': lambda n: gen_sum_threshold(n, threshold=200),
    'vowel_majority': gen_vowel_majority,
    'endpoint_match': gen_endpoint_match,
    'count_trigger': lambda n: gen_count_trigger(n, trigger_char=7, count=3),
}


def pad_sequences(data: List[Tuple[List[int], int]], max_len: int = MAX_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to max_len, return (input_ids, labels)."""
    inputs = []
    labels = []
    for seq, label in data:
        if len(seq) > max_len:
            seq = seq[:max_len]
        padded = seq + [PAD_ID] * (max_len - len(seq))
        inputs.append(padded)
        labels.append(label)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def make_dataloader(data: List[Tuple[List[int], int]], batch_size: int = 32, shuffle: bool = True):
    inputs, labels = pad_sequences(data)
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ── Main Training ────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Generate data
    n_train = 2000
    n_val = 500
    rule_names = list(RULES.keys())

    print("Generating synthetic data...")
    train_data = {name: RULES[name](n_train) for name in rule_names}
    val_data = {name: RULES[name](n_val) for name in rule_names}

    # Create model
    configs = [{'name': name} for name in rule_names]
    model = DendriteRWKV(
        vocab_size=VOCAB_SIZE,
        dim=128,
        num_layers=3,
        adapter_configs=configs,
    ).to(device)

    print(f"Backbone params: {count_base_params(model.backbone):,}")
    for name in model.adapter_names:
        print(f"  Adapter {name}: {count_lora_params(model.adapters[name]):,} LoRA params")

    # Train each adapter
    registry = AdapterRegistry("experiments/dendrite_rwkv_001/registry")

    results = {}
    for name in rule_names:
        print(f"\n{'='*50}")
        print(f"Training adapter: {name}")
        print(f"{'='*50}")

        train_loader = make_dataloader(train_data[name], batch_size=32)
        val_loader = make_dataloader(val_data[name], batch_size=32, shuffle=False)

        # Train adapter
        result = train_adapter(
            model, name,
            train_data=train_loader,
            val_data=val_loader,
            steps=500,
            lr=3e-4,
            device=device,
        )
        results[name] = result
        print(f"  Val acc: {result['val_acc']:.4f}")

        # Install to registry with gates
        # Use a probe batch for functional equivalence
        probe_x, _ = next(iter(val_loader))
        probe_data = (probe_x.to(device),)
        gate_result = registry.install(name, model.adapters[name], probe_data)
        print(f"  Registry gates: {gate_result}")

    # Save results
    exp_dir = Path("experiments/dendrite_rwkv_001")
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model config
    config = {
        "vocab_size": VOCAB_SIZE,
        "dim": 128,
        "num_layers": 3,
        "adapter_configs": configs,
        "rules": rule_names,
        "n_train": n_train,
        "n_val": n_val,
        "steps": 500,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Test routing (address heads + verifiers need to be fitted)
    print("\nFitting routing on validation hidden states...")
    model.eval()
    with torch.no_grad():
        for name in rule_names:
            val_loader = make_dataloader(val_data[name], batch_size=32, shuffle=False)
            adapter = model.adapters[name].to(device).eval()
            backbone = model.backbone.to(device).eval()

            # Collect hidden states from backbone at tap layer
            all_pools = []
            for x, y in val_loader:
                x = x.to(device)
                # Get backbone hidden states
                h = backbone.embed(x)
                for i, block in enumerate(backbone.blocks):
                    if i == 1:  # tap at layer 1 (35% depth)
                        h, _ = block(h)
                        break
                    h, _ = block(h)
                # Pool
                mean_pool = h.mean(dim=1)
                last_pool = h[:, -1]
                combined = torch.cat([mean_pool, last_pool], dim=1)
                all_pools.append(combined.cpu())

            pools = torch.cat(all_pools, dim=0).numpy()
            labels = torch.cat([y for _, y in val_loader]).numpy()

            # Fit address head + verifier for this adapter
            model.fit_address_head(name, pools, labels)
            model.fit_verifier(name, pools, labels)
            print(f"  {name}: address head + verifier fitted")

    # Save backbone
    torch.save(model.backbone.state_dict(), exp_dir / "backbone.pt")
    print(f"\nDone. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()