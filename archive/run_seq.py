#!/usr/bin/env python3
"""Clean seq2seq runner (src/seqmodel.py).

Usage:
  python run_seq.py --task all --device cpu --epochs 20 --n_samples 2000
  python run_seq.py --task recall --device cpu --d_model 128 --epochs 25

Unlike run_world.py, this model generates answers as a token sequence, so it
handles generative recall/story natively (no read-heads, no reverse_templates
supervision). All tasks share one objective.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.dataset import generate_dataset
from src.tokenizer import CharTokenizer
from src.seqmodel import SeqWorldModel, train_seq, run_seq_qa, PAD_ID


def next_exp_dir(base="experiments"):
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
                    choices=["location", "inventory", "transfer", "recall",
                             "holder", "colocation", "count_people",
                             "which_loc_most", "most_items", "empty_loc",
                             "has_item", "story", "all"])
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_enc", type=int, default=2)
    ap.add_argument("--n_dec", type=int, default=2)
    ap.add_argument("--latent_len", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exp_dir", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    exp_dir = args.exp_dir or next_exp_dir()
    print(f"STAGE: run_seq task={args.task} device={device} d_model={args.d_model} "
          f"epochs={args.epochs} n={args.n_samples} -> {exp_dir}")

    weights = None if args.task == "all" else {args.task: 1.0}
    dataset = generate_dataset(n_samples=args.n_samples, seed=args.seed,
                               task_weights=weights)
    qa = [s for s in dataset if s.get("question") or s["task_type"] == "story"]
    if args.task != "all":
        qa = [s for s in qa if s["task_type"] == args.task]

    texts = [s["narrative"] + "\nQuestion: " + (s.get("question", "") or "")
             + "\nAnswer: " + s["answer"] for s in qa]
    tokenizer = CharTokenizer(texts, max_vocab=256)

    model = SeqWorldModel(tokenizer.vocab_size, d_model=args.d_model,
                          n_enc=args.n_enc, n_dec=args.n_dec,
                          latent_len=args.latent_len).to(device)

    t0 = time.time()
    model, hist = train_seq(model, qa, tokenizer, device,
                            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    train_s = time.time() - t0

    task_acc, acc, n = run_seq_qa(model, qa, tokenizer, device)

    nl_id = tokenizer.encode("\n", max_len=None)[0]
    print("  --- sample generations ---")
    for s in qa[:10]:
        nids = torch.tensor(tokenizer.encode(s["narrative"], max_len=None)[:600],
                            dtype=torch.long).unsqueeze(0).to(device)
        pids = torch.tensor(tokenizer.encode(f"Question: {s['question']}\nAnswer: ",
                            max_len=None), dtype=torch.long).unsqueeze(0).to(device)
        z = model.encode(nids)
        ids = pids
        for _ in range(40):
            logits = model.decode_logits(z, ids)[:, -1, :]
            nxt = int(logits.argmax(-1).item())
            if nxt == nl_id or nxt == PAD_ID:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        gen = tokenizer.decode(ids[0, pids.size(1):].tolist()).strip()
        print(f"  [{s['task_type']}] Q: {s['question']} | exp={s['answer']!r} gen={gen!r}")

    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    (Path(exp_dir) / "config.json").write_text(json.dumps({
        "task": args.task, "d_model": args.d_model, "n_enc": args.n_enc,
        "n_dec": args.n_dec, "latent_len": args.latent_len,
        "epochs": args.epochs, "n_samples": args.n_samples,
        "batch_size": args.batch_size, "lr": args.lr,
        "device": str(device), "train_seconds": round(train_s, 1),
    }, indent=2))
    (Path(exp_dir) / "metrics.json").write_text(json.dumps({
        "overall_accuracy": acc, "task_accuracy": task_acc, "n": n,
        "history": hist,
    }, indent=2))
    torch.save(model.state_dict(), Path(exp_dir) / "model.pt")

    print(f"\nSTAGE: done task={args.task} acc={acc:.3f} n={n} ({train_s:.0f}s)")
    print("  per-task: " + " ".join(f"{t}={a:.3f}" for t, a in sorted(task_acc.items())))
    print(f"  saved -> {exp_dir}")


if __name__ == "__main__":
    main()
