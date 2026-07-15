"""Training script for RNN Patch Model.

Phase 1: Train encoder+decoder as autoencoder (reconstruct bytes)
Phase 2: Freeze encoder+decoder, train patch-model

This tests the hierarchical prediction hypothesis:
- Byte-level autoencoder compresses N bytes → D floats → N bytes
- Patch-level model transforms the compressed representation
- Does hierarchical structure enable longer-range prediction?
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rnn_patch_model import RNXPatchModel, BYTE_STATE_DIM, PATCH_DIM, VOCAB_SIZE
from src.byte_vocab import encode, PAD_ID
from src.logic_niiah_generator import LogicNiiahGenerator
from src.rwkv_nano import count_params


def example_to_bytes(text: str, max_len: int) -> torch.Tensor:
    """Convert text to byte tensor."""
    tokens = encode(text, max_len=max_len)
    return torch.tensor(tokens, dtype=torch.long)


def generate_batch(
    generator: LogicNiiahGenerator,
    batch_size: int,
    max_len: int,
    gen_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch and return (input_ids, targets) tensors."""
    examples = generator.generate_batch(batch_size, **gen_kwargs)
    batch_ids = []
    for ex in examples:
        ids = example_to_bytes(ex["text"], max_len)
        batch_ids.append(ids)
    input_ids = torch.stack(batch_ids)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = PAD_ID
    return input_ids, targets


def train_phase1(
    model: RNXPatchModel,
    generator: LogicNiiahGenerator,
    optimizer,
    device,
    args,
) -> dict:
    """Phase 1: Train encoder+decoder as autoencoder."""
    print("\n=== Phase 1: Training encoder+decoder (autoencoder) ===")
    
    # Freeze patch-model
    for param in model.patch_model.parameters():
        param.requires_grad = False
    
    # Unfreeze encoder+decoder
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True
    
    gen_kwargs = {
        "num_vars": args.num_vars,
        "min_transforms": args.min_transforms,
        "max_transforms": args.max_transforms,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
    }
    
    model.train()
    log = []
    t0 = time.time()
    
    for step in range(1, args.steps_phase1 + 1):
        input_ids, targets = generate_batch(
            generator, args.batch_size, args.max_len, gen_kwargs
        )
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Forward: autoencoder (encoder → decoder)
        logits = model.forward_phase1(input_ids, patch_size=args.patch_size)
        
        # Loss: next-byte prediction
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            targets.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{args.steps_phase1}  loss={loss.item():.4f}  elapsed={elapsed:.1f}s")
            log.append({"step": step, "loss": float(loss.item()), "phase": 1})
    
    return {"log": log, "elapsed": time.time() - t0}


def train_phase2(
    model: RNXPatchModel,
    generator: LogicNiiahGenerator,
    optimizer,
    device,
    args,
) -> dict:
    """Phase 2: Train patch-model while encoder+decoder are frozen."""
    print("\n=== Phase 2: Training patch-model ===")
    
    # Freeze encoder+decoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    # Unfreeze patch-model
    for param in model.patch_model.parameters():
        param.requires_grad = True
    
    gen_kwargs = {
        "num_vars": args.num_vars,
        "min_transforms": args.min_transforms,
        "max_transforms": args.max_transforms,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
    }
    
    model.train()
    log = []
    t0 = time.time()
    
    for step in range(1, args.steps_phase2 + 1):
        input_ids, targets = generate_batch(
            generator, args.batch_size, args.max_len, gen_kwargs
        )
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Forward: encoder → patch-model → decoder
        logits = model.forward_phase2(input_ids, patch_size=args.patch_size)
        
        # Loss: next-byte prediction
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            targets.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{args.steps_phase2}  loss={loss.item():.4f}  elapsed={elapsed:.1f}s")
            log.append({"step": step, "loss": float(loss.item()), "phase": 2})
    
    return {"log": log, "elapsed": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="rnn_patch_002")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps_phase1", type=int, default=1000)
    parser.add_argument("--steps_phase2", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--byte_state_dim", type=int, default=BYTE_STATE_DIM)
    parser.add_argument("--patch_state_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_vars", type=int, default=2)
    parser.add_argument("--min_transforms", type=int, default=2)
    parser.add_argument("--max_transforms", type=int, default=4)
    parser.add_argument("--noise_min", type=int, default=0)
    parser.add_argument("--noise_max", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = RNXPatchModel(
        byte_state_dim=args.byte_state_dim,
        patch_dim=args.byte_state_dim,
        patch_state_dim=args.patch_state_dim,
        vocab_size=VOCAB_SIZE
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Total params: {count_params(model):,}")
    print(f"    encoder:     {count_params(model.encoder):,}")
    print(f"    decoder:     {count_params(model.decoder):,}")
    print(f"    patch_model: {count_params(model.patch_model):,}")
    
    # Create dataset generator
    generator = LogicNiiahGenerator(seed=args.seed)
    
    # Save config
    config = vars(args)
    config["model_params"] = count_params(model)
    config["encoder_params"] = count_params(model.encoder)
    config["decoder_params"] = count_params(model.decoder)
    config["patch_model_params"] = count_params(model.patch_model)
    
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Phase 1: train encoder+decoder as autoencoder
    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    results1 = train_phase1(model, generator, optimizer1, device, args)
    
    # Phase 2: train patch-model
    optimizer2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    results2 = train_phase2(model, generator, optimizer2, device, args)
    
    # Save model
    torch.save(model.state_dict(), exp_dir / "model.pt")
    
    # Save results
    results = {
        "phase1": results1,
        "phase2": results2,
        "final_model": str(exp_dir / "model.pt"),
    }
    
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Experiment complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
