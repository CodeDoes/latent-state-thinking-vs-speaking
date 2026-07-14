#!/usr/bin/env python3
"""
Train the structured WorldModel (src/world_state.py).

This is the complexity-aware counterpart to train_modules.py: the latent state
is an EXPLICIT table of per-entity / per-item slots (the structured world that
reverse_templates.py proved the model must track), and reverse_templates is used
AS THE TEACHER to supervise each slot with dense, fact-level targets.

Usage (canonical):
  python train_world.py --quick                 # tiny CPU sanity only
  python train_world.py --device cuda --epochs 25   # real Kaggle run
  python train_world.py --analyze               # report last run
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import generate_dataset
from src.tokenizer import CharTokenizer
from src.world_state import (
    WorldModel, WorldTrainConfig, train_world, run_world_qa,
)
from src import diagnostics as diag

QUICK = dict(d_state=48, d_model=32, n_samples=400, epochs=4, batch_size=32, lr=3e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n_samples", type=int, default=6000)
    ap.add_argument("--d_state", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    if args.analyze:
        report_analyze()
        return

    if args.quick:
        for k, v in QUICK.items():
            setattr(args, k, v)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    print(f"STAGE: world-train device={device} d_state={args.d_state} "
          f"n_samples={args.n_samples} epochs={args.epochs}")

    dataset = generate_dataset(n_samples=args.n_samples, seed=args.seed)
    texts = [s["narrative"] + "\nQuestion: " + (s.get("question", "") or "")
             + "\nAnswer: " + s["answer"] for s in dataset]
    tokenizer = CharTokenizer(texts, max_vocab=256)

    cfg = WorldTrainConfig(d_state=args.d_state, d_model=args.d_model,
                           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                           seed=args.seed)

    model, hist = train_world(dataset, tokenizer, device, cfg)

    # Final eval
    qa = [s for s in dataset if s.get("question")]
    acc, task_acc, n = run_world_qa(model, qa, tokenizer, device)
    hist["qa_acc"].append(acc)
    print(f"\nStructured world-state exact-match: {acc:.3f} (n={n})")
    for t, a in task_acc.items():
        print(f"  {t}: {a:.3f}")
    print(f"STAGE: done world_qa_acc={acc:.3f}")

    # Save
    Path("world_report.json").write_text(json.dumps(
        {"qa_accuracy": acc, "task_accuracy": task_acc, "n": n,
         "config": vars(cfg), "history": hist}, indent=2))
    print("Saved world_report.json")


def report_analyze():
    p = Path("world_report.json")
    if not p.exists():
        print("world_report.json not found (run first).")
        return
    d = json.loads(p.read_text())
    print("=== LAST WORLD RUN ===")
    print(f"acc={d['qa_accuracy']:.3f} task_acc={d['task_accuracy']}")
    if d["history"].get("qa_acc"):
        print("qa_acc over time:", ["%.2f" % x for x in d["history"]["qa_acc"]])


if __name__ == "__main__":
    main()
