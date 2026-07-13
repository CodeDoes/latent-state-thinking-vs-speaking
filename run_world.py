#!/usr/bin/env python3
"""
Reusable experiment runner for the structured WorldModel (src/world_state.py).

This is the canonical local entry point for evaluating Model C (the
complexity-aware structured world-state model). It is small/CPU-or-GPU friendly
-- no Kaggle required -- and writes a full experiment record under
experiments/expNNN/ (config.json, metrics.json, samples.txt, model.pt) so every
run is preserved and comparable (append-only, per AGENTS.md).

Usage:
  python experiments/run_world.py --task location --device cuda
  python experiments/run_world.py --task all --device cuda --epochs 30
  python experiments/run_world.py --task transfer --d_state 96 --exp_dir experiments/exp_world_03

Tasks: location | inventory | transfer | recall | all
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import generate_dataset
from src.tokenizer import CharTokenizer
from src.world_state import (WorldTrainConfig, train_world, run_world_qa,
                            NAME_TO_I, NAME_POOL, I_TO_LOC, I_TO_ITEM, I_TO_NAME,
                            ITEM_TO_I, N_LOCS, N_ITEMS, N_NAMES, extract_query)
from reverse_templates import reverse_templates


def next_exp_dir(base: str = "experiments") -> str:
    """Find the next free experiments/expNNN/ dir (append-only)."""
    base = Path(base)
    base.mkdir(exist_ok=True)
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name[:3] == "exp" and p.name[3:].isdigit()]
    n = max([int(p[3:]) for p in existing], default=0) + 1
    d = base / f"exp{n:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    choices=["location", "inventory", "transfer", "recall", "all"])
    ap.add_argument("--d_state", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--slot_w", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exp_dir", default=None, help="override auto expNNN dir")
    ap.add_argument("--max_chars", type=int, default=600)
    args = ap.parse_args()

    device = torch.device(args.device)
    exp_dir = args.exp_dir or next_exp_dir()
    print(f"STAGE: run_world task={args.task} device={device} d_state={args.d_state} "
          f"epochs={args.epochs} n={args.n_samples} -> {exp_dir}")

    # ---- data ----
    if args.task == "all":
        weights = None
    else:
        weights = {t: 0.0 for t in ["location", "inventory", "transfer", "recall"]}
        weights[args.task] = 1.0
    dataset = generate_dataset(n_samples=args.n_samples, seed=args.seed,
                               task_weights=weights,
                               location_max_chars=args.max_chars,
                               inventory_max_chars=args.max_chars,
                               transfer_max_chars=args.max_chars,
                               recall_max_chars=args.max_chars)
    qa = [s for s in dataset if s.get("question")]
    # keep only the requested task for single-task runs (cleaner signal)
    if args.task != "all":
        qa = [s for s in qa if s["task_type"] == args.task]

    texts = [s["narrative"] + "\nQuestion: " + (s.get("question", "") or "")
             + "\nAnswer: " + s["answer"] for s in qa]
    tokenizer = CharTokenizer(texts, max_vocab=256)

    # ---- train ----
    cfg = WorldTrainConfig(d_state=args.d_state, d_model=args.d_model,
                           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                           seed=args.seed)
    t0 = time.time()
    model, hist = train_world(qa, tokenizer, device, cfg, qa_hook=lambda e: None)
    train_s = time.time() - t0

    # ---- eval ----
    task_acc, acc, n = run_world_qa(model, qa, tokenizer, device)

    # ---- samples ----
    samples = []
    dbg = []
    for s in qa[:12]:
        ent_slots, item_slots, holder_logits = model.write(s["narrative"], tokenizer, device)
        subj_name, item_name = extract_query(s)
        subj = NAME_TO_I.get(subj_name, 0) if subj_name in NAME_TO_I else 0
        ent = ent_slots[subj]                       # [d_state]
        hl = holder_logits                          # [N_ITEMS, N_NAMES]
        task = s["task_type"]
        if task == "location":
            gen = I_TO_LOC[model.loc_head(ent.unsqueeze(0)).argmax(-1).item()]
            if len(dbg) < 6 and subj_name in NAME_POOL:
                topk = model.loc_head(ent.unsqueeze(0))[0].topk(3).indices.tolist()
                dbg.append((subj_name, s["answer"], gen,
                            [(I_TO_LOC[k], round(model.loc_head(ent.unsqueeze(0))[0][k].item(), 2)) for k in topk]))
        elif task == "inventory":
            probs = model.inv_head(ent.unsqueeze(0)).sigmoid()[0]
            items = [I_TO_ITEM[i] for i in range(N_ITEMS) if probs[i] > 0.5]
            gen = " and ".join(items) if items else "nothing"
            if len(dbg) < 6 and subj_name in NAME_POOL:
                dbg.append((subj_name, "inv", s["answer"], gen,
                            [(I_TO_ITEM[i], round(probs[i].item(), 2))
                             for i in range(N_ITEMS) if probs[i] > 0.3]))
        elif task == "transfer":
            if item_name in ITEM_TO_I:
                iidx = ITEM_TO_I[item_name]
                islot = item_slots[iidx, :N_NAMES].unsqueeze(0)
                holder = model.holder_head(islot).argmax(-1).item()
                ent2 = ent_slots[holder]
                gen = I_TO_LOC[model.loc_head(ent2.unsqueeze(0)).argmax(-1).item()]
            else:
                gen = ""
        else:
            item = item_slots[0]
            gen = model.generate_answer(ent, item, hl, s.get("question", ""), tokenizer)
        samples.append({"task": task, "question": s.get("question"),
                        "expected": s["answer"], "generated": gen,
                        "correct": gen.strip().lower() == s["answer"].strip().lower()})

    # ---- save (append-only experiment record) ----
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    (Path(exp_dir) / "config.json").write_text(json.dumps({
        "task": args.task, "d_state": args.d_state, "d_model": args.d_model,
        "epochs": args.epochs, "n_samples": args.n_samples, "batch_size": args.batch_size,
        "lr": args.lr, "slot_w": args.slot_w, "seed": args.seed,
        "device": str(device), "train_seconds": round(train_s, 1),
    }, indent=2))
    (Path(exp_dir) / "metrics.json").write_text(json.dumps({
        "overall_accuracy": acc, "task_accuracy": task_acc, "n": n,
        "history": hist,
    }, indent=2))
    with open(Path(exp_dir) / "samples.txt", "w") as f:
        for sm in samples:
            f.write(f"[{'OK' if sm['correct'] else 'X'}] ({sm['task']}) Q: {sm['question']}\n")
            f.write(f"    expected: {sm['expected']!r}  generated: {sm['generated']!r}\n")
    torch.save(model.state_dict(), Path(exp_dir) / "model.pt")

    for d in dbg:
        print(f"  [dbg] subj={d[0]} task={d[1]} expected={d[2]} pred={d[3]} probs={d[4]}")

    print(f"\nSTAGE: done task={args.task} acc={acc:.3f} n={n} ({train_s:.0f}s)")
    print(f"  per-task: " + " ".join(f"{t}={a:.3f}" for t, a in task_acc.items()))
    print(f"  saved -> {exp_dir}")


if __name__ == "__main__":
    main()
