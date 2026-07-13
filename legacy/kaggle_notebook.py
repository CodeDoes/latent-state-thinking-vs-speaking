"""
Kaggle Notebook: Hybrid Latent-State Language Model

This notebook runs experiments on Kaggle's free GPU (T4).
It downloads the latest code, trains models, and uploads results.

Setup:
1. kaggle kernels push -p .
2. Monitor at: https://www.kaggle.com/kitastro/code
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# ============================================================
# Experiment Configuration
# ============================================================

EXPERIMENTS = [
    {
        "exp_id": "exp001",
        "model": "baseline",
        "description": "Standard autoregressive transformer baseline",
        "latent_steps": 0,
        "d_model": 256,
        "epochs": 30,
        "batch_size": 32,
        "n_samples": 5000,
    },
    {
        "exp_id": "exp002",
        "model": "latent_ssm",
        "description": "SSM only - pure recurrent baseline",
        "latent_steps": 1,
        "d_model": 256,
        "epochs": 30,
        "batch_size": 32,
        "n_samples": 5000,
    },
    {
        "exp_id": "exp003",
        "model": "latent_ssm",
        "description": "SSM with 4 latent thinking steps",
        "latent_steps": 4,
        "d_model": 256,
        "epochs": 30,
        "batch_size": 32,
        "n_samples": 5000,
    },
    {
        "exp_id": "exp004",
        "model": "latent_ssm",
        "description": "SSM with 8 latent thinking steps",
        "latent_steps": 8,
        "d_model": 256,
        "epochs": 30,
        "batch_size": 32,
        "n_samples": 5000,
    },
    {
        "exp_id": "exp005",
        "model": "latent_ssm_decoder",
        "description": "SSM + FFN decoder - decoupled generation",
        "latent_steps": 4,
        "d_model": 256,
        "epochs": 30,
        "batch_size": 32,
        "n_samples": 5000,
    },
]

# ============================================================
# Helper Functions
# ============================================================

def save_checkpoint(exp_id, epoch, model, optimizer, metrics, output_dir="experiments"):
    """Save checkpoint with timestamp."""
    path = Path(output_dir) / exp_id
    path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }, path / "checkpoint.pt")

    # Also save as latest
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }, path / "latest.pt")

    print(f"  Checkpoint saved: {path}/checkpoint.pt")


def upload_to_kaggle(output_dir="experiments"):
    """Upload results to Kaggle dataset."""
    try:
        import kaggle
        kaggle.api.dataset_create_version(
            folder=output_dir,
            version_notes=f"Experiment results {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        print("Results uploaded to Kaggle!")
    except Exception as e:
        print(f"Upload failed (not critical): {e}")


# ============================================================
# Main Training Loop
# ============================================================

def main():
    """Run all experiments sequentially."""
    start_time = time.time()

    # Import project modules
    # In Kaggle, these will be in the notebook's working directory
    from src.dataset import generate_dataset
    from src.tokenizer import build_tokenizer_from_dataset
    from src.models import BaselineTransformer, LatentSSM, LatentSSMDecoder
    from src.trainer import Trainer, ExperimentConfig, TextDataset
    from torch.utils.data import DataLoader

    all_results = {}

    for exp_config in EXPERIMENTS:
        exp_id = exp_config["exp_id"]
        print(f"\n{'='*70}")
        print(f"Starting {exp_id}: {exp_config['description']}")
        print(f"{'='*70}")

        # Set seed
        torch.manual_seed(42)

        # Generate dataset
        print("Generating dataset...")
        dataset = generate_dataset(n_samples=exp_config["n_samples"], seed=42)
        train_data = dataset[:int(0.8 * len(dataset))]
        val_data = dataset[int(0.8 * len(dataset)):]
        print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

        # Build tokenizer
        tokenizer = build_tokenizer_from_dataset(dataset)
        print(f"  Vocab size: {tokenizer.vocab_size}")

        # Prepare text data
        train_texts = [f"{s['narrative']} {s['question']} {s['answer']}" for s in train_data if s.get("question")]
        val_texts = [f"{s['narrative']} {s['question']} {s['answer']}" for s in val_data if s.get("question")]

        # Build model
        model_type = exp_config["model"]
        if model_type == "baseline":
            model = BaselineTransformer(
                vocab_size=tokenizer.vocab_size,
                d_model=exp_config["d_model"],
                num_layers=4,
                nhead=8,
            )
        elif model_type == "latent_ssm":
            model = LatentSSM(
                vocab_size=tokenizer.vocab_size,
                d_state=exp_config["d_model"],
                d_model=exp_config["d_model"],
                num_ssm_layers=2,
                latent_steps=exp_config["latent_steps"],
            )
        elif model_type == "latent_ssm_decoder":
            model = LatentSSMDecoder(
                vocab_size=tokenizer.vocab_size,
                d_state=exp_config["d_model"],
                d_model=exp_config["d_model"],
                num_ssm_layers=2,
                latent_steps=exp_config["latent_steps"],
                tokens_per_step=8,
            )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {model_type}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Latent steps: {exp_config['latent_steps']}")

        # Create config
        config = ExperimentConfig(
            exp_id=exp_id,
            model=model_type,
            d_model=exp_config["d_model"],
            d_state=exp_config["d_model"],
            latent_steps=exp_config["latent_steps"],
            batch_size=exp_config["batch_size"],
            num_epochs=exp_config["epochs"],
            save_every=5,
            eval_every=5,
        )

        # Create data loaders
        train_dataset = TextDataset(train_texts, tokenizer, max_len=256)
        val_dataset = TextDataset(val_texts, tokenizer, max_len=256)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        # Train
        trainer = Trainer(model, config, tokenizer)
        metrics = trainer.train(train_loader, val_loader, qa_dataset=val_data)

        # Save results
        all_results[exp_id] = {
            "final_train_loss": metrics["train_loss"][-1] if metrics["train_loss"] else None,
            "final_val_loss": metrics["val_loss"][-1] if metrics["val_loss"] else None,
            "final_qa_accuracy": metrics["eval_accuracy"][-1]["overall_accuracy"] if metrics["eval_accuracy"] else None,
            "n_parameters": n_params,
            "latent_steps": exp_config["latent_steps"],
        }

        # Save checkpoint after each experiment
        save_checkpoint(
            exp_id=exp_id,
            epoch=config.num_epochs,
            model=model,
            optimizer=trainer.optimizer,
            metrics={"train_loss": metrics["train_loss"][-1], "val_loss": metrics["val_loss"][-1] if metrics["val_loss"] else None},
        )

        print(f"\n  {exp_id} complete!")

    # ============================================================
    # Summary
    # ============================================================

    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Exp':<10} {'Model':<20} {'Steps':<8} {'Params':<10} {'Train Loss':<12} {'Val Loss':<12} {'QA Acc':<8}")
    print("-" * 80)

    for exp_config in EXPERIMENTS:
        exp_id = exp_config["exp_id"]
        r = all_results[exp_id]
        print(
            f"{exp_id:<10} "
            f"{exp_config['model']:<20} "
            f"{exp_config['latent_steps']:<8} "
            f"{r['n_parameters']:<10,} "
            f"{r['final_train_loss']:<12.4f} "
            f"{(r['final_val_loss'] or 0):<12.4f} "
            f"{(r['final_qa_accuracy'] or 0):<8.3f}"
        )

    # Save summary
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved to results.json")
    print(f"Checkpoints saved to experiments/")

    # Try uploading to Kaggle
    # upload_to_kaggle()

    return all_results


if __name__ == "__main__":
    main()
