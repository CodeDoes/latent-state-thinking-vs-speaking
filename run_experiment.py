#!/usr/bin/env python3
"""
Run experiments for the hybrid latent-state language model.

Usage:
    python run_experiment.py --exp_id exp001 --model baseline
    python run_experiment.py --exp_id exp002 --model latent_ssm --latent_steps 4
    python run_experiment.py --exp_id exp003 --model latent_ssm_decoder --latent_steps 4

This script:
1. Generates synthetic dataset
2. Builds tokenizer
3. Creates model
4. Trains and evaluates
5. Saves checkpoint, metrics, and samples to experiments/<exp_id>/
"""

import argparse
import os
import sys
import random
import json
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import generate_dataset, format_for_training
from src.tokenizer import build_tokenizer_from_dataset, CharTokenizer
from src.models import BaselineTransformer, LatentSSM, LatentSSMDecoder
from src.trainer import Trainer, ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Run latent-state model experiments")
    parser.add_argument("--exp_id", type=str, default="exp001", help="Experiment ID")
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "latent_ssm", "latent_ssm_decoder"],
                        help="Model type")
    parser.add_argument("--latent_steps", type=int, default=4, help="Number of latent thinking steps")
    parser.add_argument("--think_every", type=int, default=4, help="Run thinking steps every N tokens")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--max_seq_len", type=int, default=768, help="Maximum sequence length")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--recall_max_chars", type=int, default=600, help="Max recall narrative length (lower = shorter context)")
    parser.add_argument("--location_max_chars", type=int, default=600, help="Max location narrative length")
    parser.add_argument("--inventory_max_chars", type=int, default=600, help="Max inventory narrative length")
    parser.add_argument("--transfer_max_chars", type=int, default=600, help="Max transfer narrative length")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"=== Experiment: {args.exp_id} ===")
    print(f"Model: {args.model}")
    print(f"Latent steps: {args.latent_steps}")
    print(f"Device: {args.device}")
    print()

    # Generate dataset
    print("Generating dataset...")
    dataset = generate_dataset(
        n_samples=args.n_samples, seed=args.seed,
        location_max_chars=args.location_max_chars,
        inventory_max_chars=args.inventory_max_chars,
        transfer_max_chars=args.transfer_max_chars,
        recall_max_chars=args.recall_max_chars,
    )
    print(f"Generated {len(dataset)} samples")

    # Split
    train_data = dataset[:int(0.8 * len(dataset))]
    val_data = dataset[int(0.8 * len(dataset)):]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Build tokenizer from the *formatted* texts so format markers
    # ("Question:", "Answer:", newline) are in-vocab, not UNK.
    print("Building tokenizer...")
    train_texts = [format_for_training(s) for s in train_data]
    val_texts = [format_for_training(s) for s in val_data]
    tokenizer = CharTokenizer(train_texts, max_vocab=256)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    # Create config
    config = ExperimentConfig(
        exp_id=args.exp_id,
        model=args.model,
        d_model=args.d_model,
        d_state=args.d_model,
        latent_steps=args.latent_steps,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        device=args.device,
    )

    # Build model
    print(f"\nBuilding model: {args.model}")
    if args.model == "baseline":
        model = BaselineTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            num_layers=4,
            nhead=8,
        )
    elif args.model == "latent_ssm":
        model = LatentSSM(
            vocab_size=tokenizer.vocab_size,
            d_state=args.d_model,
            d_model=args.d_model,
            num_ssm_layers=2,
            latent_steps=args.latent_steps,
            think_every=args.think_every,
        )
    elif args.model == "latent_ssm_decoder":
        model = LatentSSMDecoder(
            vocab_size=tokenizer.vocab_size,
            d_state=args.d_model,
            d_model=args.d_model,
            num_ssm_layers=2,
            latent_steps=args.latent_steps,
            tokens_per_step=8,
            think_every=args.think_every,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Create datasets
    from src.trainer import TextDataset, DataLoader
    train_dataset = TextDataset(train_texts, tokenizer, max_len=args.max_seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, max_len=args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Train
    trainer = Trainer(model, config, tokenizer, output_dir=args.output_dir)
    metrics = trainer.train(train_loader, val_loader, qa_dataset=val_data)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment {args.exp_id} complete")
    print(f"Final train loss: {metrics['train_loss'][-1]:.4f}")
    if metrics['val_loss']:
        print(f"Final val loss: {metrics['val_loss'][-1]:.4f}")
    if metrics['eval_accuracy']:
        last_qa = metrics['eval_accuracy'][-1]
        print(f"Final QA accuracy: {last_qa['overall_accuracy']:.3f}")
        for task, acc in last_qa['task_accuracy'].items():
            print(f"  {task}: {acc:.3f}")
        print("  Stratified (by difficulty bucket):")
        for bkey, acc in sorted(last_qa['stratified'].items()):
            print(f"    {bkey:<28} {acc:.3f}")
    print(f"Results saved to: {trainer.output_dir}")
    print(f"{'='*60}")

    # Kaggle: persist the model + metrics into /kaggle/working (the only
    # directory Kaggle keeps as "Output" after the run). Guarded so local
    # runs are unaffected.
    import shutil, glob
    kaggle_working = "/kaggle/working"
    if os.path.isdir(kaggle_working):
        out_dir = os.path.join(kaggle_working, args.exp_id)
        os.makedirs(out_dir, exist_ok=True)
        for f in glob.glob(os.path.join(trainer.output_dir, "*")):
            shutil.copy(f, out_dir)
        # Also drop a clearly-named copy at the top level of the Output panel.
        best = os.path.join(trainer.output_dir, "best_model.pt")
        if os.path.exists(best):
            shutil.copy(best, os.path.join(kaggle_working, f"{args.exp_id}_best_model.pt"))
        print(f"Saved model + metrics to {kaggle_working} (Kaggle Output)")

    return metrics


if __name__ == "__main__":
    main()
