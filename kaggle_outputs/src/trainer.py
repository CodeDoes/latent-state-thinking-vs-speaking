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
    answer_loss_weight: float = 0.0   # 0 = standard next-token loss over all chars
                                       # >0 = up-weight loss on the answer-slot tokens
                                       #      so the model gets sharp signal on what to fill
                                       #      Reward *only* comes from predicting the right
                                       #      answer chars, not just fitting filler.
    latent_consistency_weight: float = 0.1
    state_evolution_weight: float = 0.1

    # Visibility
    print_every_batches: int = 50     # visible "TRAIN" log every N batches with loss + a sample
    gen_sample_every: int = 200       # generate a sample every N batches (heavy)
    save_initial_loss_baseline: bool = True

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

    def _answer_positions(self, ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Find the token positions right AFTER 'Answer: ' in each batch row.

        Returns a [batch, seq_len] bool mask (True at the answer positions)
        OR None if 'Answer:' isn't a contiguous pattern in the vocab (then
        we silently fall back to plain next-token loss).

        Why focus on the answer slot?  Most training characters are easy filler
        (spaces, the name John, the verb "moved"). Gradient from those drowns
        out the signal for "what goes in the answer slot." Focusing loss on
        the answer slot means the model gets a sharp, focused signal that
        directly improves exact-match accuracy.
        """
        ans_lit = self.tokenizer.encode("Answer: ", max_len=None)
        if len(ans_lit) < 3:
            return None  # can't pattern-match
        L = len(ans_lit)
        bsz, sl = ids.shape
        if sl <= L:
            return None
        mask = torch.zeros((bsz, sl), dtype=torch.bool, device=ids.device)
        # Slide a window per row.
        ids_cpu = ids.detach()
        for b in range(bsz):
            row = ids_cpu[b].tolist()
            # Build rolling "last L tokens equal ans_lit"
            i = L - 1
            in_run = False
            for t in range(L, sl):
                if row[t - L:t] == ans_lit:
                    # tokens indices [t, t+1, ...] are answer positions
                    # but we want the *target* positions (y = ids[1:]), which
                    # are answer positions [t-1, t, t+1, ...]. Align with y:
                    # y[t-1] is the first answer char.
                    mask[b, max(0, t - 1):] = True
                    break
        return mask

    def train_epoch(self, dataloader: DataLoader, eval_during_epoch: bool = False) -> Tuple[float, dict]:
        """Run one training epoch with mid-epoch visibility.

        Two loss numbers are reported mid-epoch so the user can SEE whether
        the model is improving on filler (loss_full) or on the answer slot
        (loss_answer) specifically. Modest mid-epoch generation + cheap QA
        snapshots make training legible instead of "just a number at the end."
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        info = {"mid_epoch_samples": [], "mid_epoch_qa": []}

        pb_every = self.config.print_every_batches
        g_every   = self.config.gen_sample_every

        pbar = tqdm(dataloader, desc=f"Training [{self.config.exp_id}]")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass - all models produce [batch, seq, vocab]
            logits = self.model(batch_x)
            pad_id = self.tokenizer.vocab[self.tokenizer.pad_token]

            # Build per-token loss. We compute the answer-slot mask *over the
            # input positions* (batch_x) but use it with the target positions
            # (batch_y) by sliding one step earlier.
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = batch_y.reshape(-1)
            loss_per_token = F.cross_entropy(
                flat_logits, flat_targets, ignore_index=pad_id, reduction='none',
            ).reshape(batch_y.shape)

            # Standard uniform cross-entropy weight is 1.0. Optional answer-focus:
            # weight = 1 + answer_loss_weight * mask_of_answer_positions. Default 0.
            ans_w = self.config.answer_loss_weight
            if ans_w > 0:
                mask_in = self._answer_positions(batch_x)
                if mask_in is not None:
                    # align mask with batch_y: answer chars in batch_y are
                    # shifted one position earlier than in batch_x.
                    mask = torch.zeros_like(mask_in)
                    mask[:, :-1] = mask_in[:, 1:]  # y[:, t] is x[:, t+1]
                else:
                    mask = torch.zeros_like(batch_y, dtype=torch.bool)
                mask = mask & (batch_y != pad_id)  # don't double-count pad
                weights = 1.0 + ans_w * mask.to(loss_per_token.dtype)
            else:
                weights = torch.ones_like(loss_per_token)

            non_pad = (batch_y != pad_id).float()
            loss = (loss_per_token * weights * non_pad).sum() / non_pad.sum().clamp(min=1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # -------- visible TRAIN progress --------
            if pb_every > 0 and (n_batches % pb_every == 0):
                with torch.no_grad():
                    if ans_w > 0 and mask.any():
                        ans_l = (loss_per_token * mask.to(loss_per_token.dtype)).sum() / mask.float().sum().clamp(min=1.0)
                    else:
                        ans_l = loss_per_token.mean()
                    full_l = loss_per_token.mean()
                print(f"  [TRAIN] exp={self.config.exp_id} ep={len(self.metrics['train_loss'])+1} "
                      f"batch={n_batches}/{len(dataloader)} "
                      f"loss_full={full_l.item():.3f} loss_answer={ans_l.item():.3f} "
                      f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

            # -------- visible mid-epoch generation ----------
            if eval_during_epoch and g_every > 0 and (n_batches % g_every == 0):
                snap = self._mid_epoch_sample()
                if snap:
                    info["mid_epoch_samples"].append(snap)

            # -------- mid-epoch cheap QA ----------
            if eval_during_epoch and g_every > 0 and (n_batches % (2 * g_every) == 0):
                qa = self.run_qa_eval_quick()
                if qa:
                    info["mid_epoch_qa"].append(qa)
                    print(f"  [MID-QA] n_eval={qa['n']} acc={qa['overall_accuracy']:.3f}")

        return total_loss / max(n_batches, 1), info

    @torch.no_grad()
    def _mid_epoch_sample(self) -> Optional[dict]:
        """Generate one sample mid-epoch so we can see what the model is doing."""
        from src.dataset import build_prompt, parse_answer
        ds = getattr(self, "_qa_snapshot_dataset", None)
        if not ds:
            return None
        sample = ds[0]
        if not sample.get("question"):
            return None
        prompt = build_prompt(sample)
        # greedy + a tiny chance of sampling is bad; use greedy for visibility
        gen = self.generate(prompt, max_tokens=24, temperature=0.0)
        print(f"  [GEN ] Q={sample['question']!r} "
              f"expected={sample['answer']!r} got={gen.strip()!r}")
        return {"prompt": prompt, "expected": sample["answer"], "generated": gen.strip()}

    @torch.no_grad()
    def run_qa_eval_quick(self, n: int = 32) -> Optional[dict]:
        """Cheap, fast QA on first n val samples for live-monitoring."""
        from src.dataset import build_prompt, parse_answer
        ds = getattr(self, "_qa_snapshot_dataset", None)
        if not ds:
            return None
        self.model.eval()
        ok, tot = 0, 0
        for s in ds[:n]:
            if not s.get("question"):
                continue
            prompt = build_prompt(s)
            gen = self.generate(prompt, max_tokens=20, temperature=0.0)
            pred = parse_answer(gen).strip().lower()
            exp = s["answer"].strip().lower()
            ok += int(pred == exp); tot += 1
        return {"overall_accuracy": ok / max(tot, 1), "n": tot}

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
        """Full training loop.

        Stages:
          STAGE: data    -- dataset / tokenizer / answer-slot coverage stats
          STAGE: token   -- vocab info (size, special-token presence)
          STAGE: init    -- model class & param count vs same-size baseline
          STAGE: train   -- per-epoch: train_loss, val_loss, qa_acc
          STAGE: gen     -- mid-epoch sample generations (visible)
          STAGE: done    -- final summary
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {self.config.exp_id}")
        print(f"Model: {self.config.model}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")

        # ---- STAGE: data ----
        print(f"STAGE: data exp={self.config.exp_id} "
              f"train_batches={len(train_loader)} val_batches={len(val_loader) if val_loader else 0} "
              f"max_seq_len={self.config.max_seq_len}")
        if qa_dataset:
            from collections import Counter
            ttc = Counter(s["task_type"] for s in qa_dataset if s.get("question"))
            print(f"  qa={len([s for s in qa_dataset if s.get('question')])} per_task={dict(ttc)}")

        # ---- STAGE: token ----
        ans_lit = self.tokenizer.encode("Answer: ", max_len=None)
        print(f"STAGE: token exp={self.config.exp_id} "
              f"vocab_size={self.tokenizer.vocab_size} "
              f"'Answer: ' encodes_to={ans_lit}")

        # ---- STAGE: init ----
        print(f"STAGE: init exp={self.config.exp_id} optimizer=AdamW "
              f"lr={self.config.learning_rate} clip={self.config.gradient_clip}")

        # store the qa snapshot for mid-epoch visibility
        self._qa_snapshot_dataset = list(qa_dataset) if qa_dataset else None

        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()

            # Train (with mid-epoch visibility)
            train_loss, info = self.train_epoch(
                train_loader,
                eval_during_epoch=(qa_dataset is not None and epoch % self.config.eval_every == 0),
            )
            self.metrics["train_loss"].append(train_loss)
            # stash visible mid-epoch snapshots in metrics for offline review
            if info["mid_epoch_samples"]:
                self.metrics.setdefault("mid_epoch_samples", []).extend(
                    [{"epoch": epoch, **s} for s in info["mid_epoch_samples"]])
            if info["mid_epoch_qa"]:
                self.metrics.setdefault("mid_epoch_qa", []).extend(
                    [{"epoch": epoch, **q} for q in info["mid_epoch_qa"]])

            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.metrics["val_loss"].append(val_loss)

            # Full QA evaluation
            qa_results = None
            if qa_dataset and epoch % self.config.eval_every == 0:
                qa_results = self.run_qa_eval(qa_dataset)
                self.metrics["eval_accuracy"].append(qa_results)

                # Print + record sample outputs so we can SEE whether the
                # model's output is actually useful, not just low-loss.
                print(f"\n--- [EVAL] exp={self.config.exp_id} epoch {epoch} ---")
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
                  f"train_loss={train_loss:.4f} "
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
