"""
Training loop for latent-state language model experiments.

Handles:
  - Training with multiple loss types
  - Experiment tracking
  - Checkpointing
  - Evaluation
  - Sample generation
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    exp_id: str = "exp001"
    model: str = "baseline"  # baseline, latent_ssm, latent_ssm_decoder

    # Model params
    d_model: int = 256
    d_state: int = 256
    nhead: int = 8
    num_layers: int = 4
    num_ssm_layers: int = 2
    latent_steps: int = 4
    tokens_per_step: int = 8

    # Training params
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 20
    max_seq_len: int = 256
    gradient_clip: float = 1.0
    warmup_steps: int = 100

    # Loss weights
    token_loss_weight: float = 1.0
    latent_consistency_weight: float = 0.1
    state_evolution_weight: float = 0.1

    # Misc
    seed: int = 42
    device: str = "auto"
    save_every: int = 5
    eval_every: int = 1

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TextDataset(Dataset):
    """Simple text dataset for training."""

    def __init__(self, texts: List[str], tokenizer, max_len: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer.encode(text, max_len=self.max_len)
        ids = ids + [self.tokenizer.vocab[self.tokenizer.eos_token]]
        ids = ids[:self.max_len + 1]

        # Pad to max_len + 1
        while len(ids) < self.max_len + 1:
            ids.append(self.tokenizer.vocab[self.tokenizer.pad_token])

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


class Trainer:
    """Training loop with experiment tracking and checkpointing."""

    def __init__(self, model: nn.Module, config: ExperimentConfig, tokenizer, output_dir: str = "experiments"):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir) / config.exp_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=config.num_epochs * 100,  # approximate
            pct_start=0.1,
        )

        # Metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "eval_accuracy": [],
            "samples": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Training [{self.config.exp_id}]")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if hasattr(self.model, 'forward'):
                output = self.model(batch_x)
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output

                # Handle different output shapes
                if logits.dim() == 3:
                    # [batch, seq, vocab] or [batch, n_tokens, vocab]
                    logits = logits.reshape(-1, logits.size(-1))
                    targets = batch_y.reshape(-1)
                elif logits.dim() == 2:
                    # [batch, vocab] — sequence-level prediction
                    # Use first target token
                    logits = logits
                    targets = batch_y[:, 0]
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")

                loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.vocab[self.tokenizer.pad_token])
            else:
                raise ValueError("Model must have a forward method")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Run evaluation."""
        self.model.eval()
        total_loss = 0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            output = self.model(batch_x)
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output

            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.size(-1))
                targets = batch_y.reshape(-1)
            elif logits.dim() == 2:
                logits = logits
                targets = batch_y[:, 0]

            loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.vocab[self.tokenizer.pad_token])
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text from a prompt."""
        self.model.eval()

        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, max_len=self.config.max_seq_len)
        generated = list(prompt_ids)

        for _ in range(max_tokens):
            x = torch.tensor([generated], dtype=torch.long).to(self.device)

            output = self.model(x)
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output

            # Get next token
            if logits.dim() == 3:
                next_logits = logits[0, -1, :]
            else:
                next_logits = logits[0]

            next_token = torch.argmax(next_logits).item()

            if self.tokenizer.inv_vocab.get(next_token) == self.tokenizer.eos_token:
                break

            generated.append(next_token)

        # Decode only the generated part (after prompt)
        generated_text = self.tokenizer.decode(generated[len(prompt_ids):])
        return generated_text

    def run_qa_eval(self, dataset: List[dict]) -> Dict[str, float]:
        """Evaluate on QA tasks."""
        self.model.eval()
        correct = 0
        total = 0
        by_task = {}

        for sample in dataset:
            narrative = sample["narrative"]
            question = sample["question"]
            answer = sample["answer"]
            task_type = sample["task_type"]

            if task_type not in by_task:
                by_task[task_type] = {"correct": 0, "total": 0}

            prompt = f"{narrative} {question}"
            generated = self.generate(prompt, max_tokens=30)

            # Simple string matching
            if answer.lower() in generated.lower():
                correct += 1
                by_task[task_type]["correct"] += 1

            total += 1
            by_task[task_type]["total"] += 1

        # Compute accuracy by task
        task_accuracy = {}
        for task, counts in by_task.items():
            task_accuracy[task] = counts["correct"] / max(counts["total"], 1)

        return {
            "overall_accuracy": correct / max(total, 1),
            "task_accuracy": task_accuracy,
        }

    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint and metrics."""
        checkpoint_path = self.output_dir / f"model_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }, checkpoint_path)

        # Save latest as best_model.pt
        latest_path = self.output_dir / "best_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }, latest_path)

    def save_results(self):
        """Save final results."""
        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        qa_dataset: Optional[List[dict]] = None,
    ):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.config.exp_id}")
        print(f"Model: {self.config.model}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)
            self.metrics["train_loss"].append(train_loss)

            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.metrics["val_loss"].append(val_loss)

            # QA evaluation
            qa_results = None
            if qa_dataset and epoch % self.config.eval_every == 0:
                qa_results = self.run_qa_eval(qa_dataset)
                self.metrics["eval_accuracy"].append(qa_results)

                # Generate samples
                for sample in qa_dataset[:3]:
                    prompt = f"{sample['narrative']} {sample['question']}"
                    generated = self.generate(prompt, max_tokens=30)
                    self.metrics["samples"].append({
                        "epoch": epoch,
                        "prompt": prompt[:100] + "...",
                        "expected": sample["answer"],
                        "generated": generated,
                        "task_type": sample["task_type"],
                    })

            epoch_time = time.time() - epoch_start

            # Log
            log_msg = f"Epoch {epoch}/{self.config.num_epochs} | "
            log_msg += f"train_loss: {train_loss:.4f} | "
            if val_loss is not None:
                log_msg += f"val_loss: {val_loss:.4f} | "
            if qa_results:
                log_msg += f"qa_acc: {qa_results['overall_accuracy']:.3f} | "
            log_msg += f"time: {epoch_time:.1f}s"
            print(log_msg)

            # Save checkpoint
            if epoch % self.config.save_every == 0 or epoch == self.config.num_epochs:
                self.save_checkpoint(epoch, {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "qa_accuracy": qa_results["overall_accuracy"] if qa_results else None,
                })

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")

        # Save final results
        self.save_results()

        # Save samples to text file
        samples_path = self.output_dir / "samples.txt"
        with open(samples_path, 'w') as f:
            for sample in self.metrics["samples"]:
                f.write(f"\n--- Epoch {sample['epoch']} [{sample['task_type']}] ---\n")
                f.write(f"Prompt: {sample['prompt']}\n")
                f.write(f"Expected: {sample['expected']}\n")
                f.write(f"Generated: {sample['generated']}\n")

        return self.metrics
