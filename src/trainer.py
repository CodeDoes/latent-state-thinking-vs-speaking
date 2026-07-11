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

try:
    from src.dataset import build_prompt, parse_answer, _bucket
except ImportError:  # allow running trainer standalone
    from dataset import build_prompt, parse_answer, _bucket


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

            # Forward pass - all models produce [batch, seq, vocab]
            logits = self.model(batch_x)
            
            # Reshape for loss computation
            logits = logits.reshape(-1, logits.size(-1))
            targets = batch_y.reshape(-1)
            
            loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.vocab[self.tokenizer.pad_token])

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

            logits = self.model(batch_x)
            logits = logits.reshape(-1, logits.size(-1))
            targets = batch_y.reshape(-1)
            
            loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.vocab[self.tokenizer.pad_token])
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.8, top_k: int = 40) -> str:
        """Generate text from a prompt.

        temperature <= 0 disables sampling and uses greedy argmax (used for
        deterministic, exact-match evaluation). Generation stops at EOS or a
        newline so we capture just the answer in the "Answer:" slot.
        """
        self.model.eval()

        prompt_ids = self.tokenizer.encode(prompt, max_len=self.config.max_seq_len)
        generated = list(prompt_ids)

        eos_id = self.tokenizer.vocab[self.tokenizer.eos_token]
        newline_id = self.tokenizer.vocab.get("\n", None)

        for _ in range(max_tokens):
            x = torch.tensor([generated], dtype=torch.long).to(self.device)
            logits = self.model(x)
            next_logits = logits[0, -1, :]

            if temperature <= 0:
                next_token = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    kth = torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[next_logits < kth] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            if next_token == eos_id:
                break
            if newline_id is not None and next_token == newline_id:
                break
            generated.append(next_token)

        return self.tokenizer.decode(generated[len(prompt_ids):])

    @torch.no_grad()
    def run_qa_eval(self, dataset: List[dict], max_new_tokens: int = 20) -> Dict[str, float]:
        """Strict exact-match evaluation on QA tasks, with stratified breakdowns.

        Returns overall accuracy, per-task accuracy, and accuracy stratified by
        difficulty bucket (interference level / decoy presence / etc.) so we can
        see *where* the model fails -- not just a single number. This also
        exposes "low loss but useless output" that the old substring check hid.
        """
        self.model.eval()
        correct = 0
        total = 0
        by_task = {}
        by_bucket = {}

        for sample in dataset:
            if not sample.get("question"):  # skip story tasks
                continue

            task_type = sample["task_type"]
            by_task.setdefault(task_type, {"correct": 0, "total": 0})
            bucket = _bucket(sample)
            bkey = f"{task_type}|{bucket}"
            by_bucket.setdefault(bkey, {"correct": 0, "total": 0})

            prompt = build_prompt(sample)
            expected = sample["answer"].strip().lower()

            generated = self.generate(prompt, max_tokens=max_new_tokens, temperature=0.0)
            predicted = parse_answer(generated).strip().lower()

            ok = (predicted == expected)
            if ok:
                correct += 1
                by_task[task_type]["correct"] += 1
                by_bucket[bkey]["correct"] += 1
            total += 1
            by_task[task_type]["total"] += 1
            by_bucket[bkey]["total"] += 1

        task_accuracy = {
            task: counts["correct"] / max(counts["total"], 1)
            for task, counts in by_task.items()
        }
        stratified = {
            bkey: counts["correct"] / max(counts["total"], 1)
            for bkey, counts in by_bucket.items()
        }
        return {
            "overall_accuracy": correct / max(total, 1),
            "task_accuracy": task_accuracy,
            "stratified": stratified,
            "n": total,
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

                # Print + record sample outputs so we can SEE whether the
                # model's output is actually useful, not just low-loss.
                print(f"\n--- SAMPLE OUTPUTS (epoch {epoch}) ---")
                for sample in qa_dataset[:4]:
                    if not sample.get("question"):
                        continue
                    prompt = build_prompt(sample)
                    generated = self.generate(prompt, max_tokens=20, temperature=0.0)
                    predicted = parse_answer(generated).strip().lower()
                    expected = sample["answer"].strip().lower()
                    ok = "✓" if predicted == expected else "✗"
                    print(f"  [{ok}] ({sample['task_type']}) Q: {sample['question']}")
                    print(f"       expected: {sample['answer']!r}  got: {generated!r}")
                    self.metrics["samples"].append({
                        "epoch": epoch,
                        "task_type": sample["task_type"],
                        "prompt": prompt,
                        "expected": sample["answer"],
                        "generated": generated,
                        "correct": predicted == expected,
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
            print(f"STAGE: train exp={self.config.exp_id} ep={epoch}/{self.config.num_epochs} "
                  f"val_loss={val_loss if val_loss is not None else '-'} "
                  f"qa_acc={qa_results['overall_accuracy'] if qa_results else '-'}")

            # Save checkpoint
            if epoch % self.config.save_every == 0 or epoch == self.config.num_epochs:
                self.save_checkpoint(epoch, {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "qa_accuracy": qa_results["overall_accuracy"] if qa_results else None,
                })

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")

        # Diagnostic: flag the "low loss but useless output" failure mode.
        if self.metrics["val_loss"] and self.metrics["eval_accuracy"]:
            final_val = self.metrics["val_loss"][-1]
            final_acc = self.metrics["eval_accuracy"][-1]["overall_accuracy"]
            if final_val < 2.0 and final_acc < 0.5:
                print("\n  ⚠ LOW LOSS BUT USELESS OUTPUT")
                print(f"    val_loss={final_val:.3f} (low) yet exact-match acc={final_acc:.3f} (low).")
                print("    The model minimized cross-entropy without learning to answer.")

        # Save final results
        self.save_results()

        # Save samples to text file
        samples_path = self.output_dir / "samples.txt"
        with open(samples_path, 'w') as f:
            for sample in self.metrics["samples"]:
                mark = "CORRECT" if sample.get("correct") else "WRONG"
                f.write(f"\n--- Epoch {sample['epoch']} [{sample['task_type']}] {mark} ---\n")
                f.write(f"Prompt:\n{sample['prompt']}\n")
                f.write(f"Expected:  {sample['expected']}\n")
                f.write(f"Generated: {sample['generated']}\n")

        return self.metrics
