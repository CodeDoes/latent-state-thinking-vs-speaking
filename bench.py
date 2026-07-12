#!/usr/bin/env python3
"""
Reusable benchmark for the hybrid latent-state language model.

One entry point for everything:
  * Train one or more model variants.
  * Strict greedy exact-match eval (per task AND by difficulty bucket).
  * Print + save a rich report (loss curves, per-task acc, stratified acc,
    sample outputs, params, "low loss but useless output" flag).

Usage:
  # Fast CPU smoke of every model (tiny):
  python bench.py --quick

  # One model, more serious:
  python bench.py --models baseline --epochs 20 --n_samples 4000 --device cuda

  # Full battery on GPU:
  python bench.py --models baseline,latent_ssm,latent_ssm_decoder --epochs 30 --n_samples 8000 --device cuda

  # Just report existing experiment dirs (e.g. downloaded from Kaggle):
  python bench.py --analyze

The point of the strict, multi-dimensional eval is to catch the failure mode
where cross-entropy loss drops but the model still can't answer (low loss,
useless output). We never report a single number.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import (
    generate_dataset, format_for_training, build_prompt, parse_answer, _bucket,
)
from src.tokenizer import CharTokenizer
from src.models import BaselineTransformer, LatentSSM, LatentSSMDecoder
from src.trainer import Trainer, ExperimentConfig, TextDataset, DataLoader


# Model registry: key -> (model_type, latent_steps, think_every, tokens_per_step, d_model)
# `d_model` here is the per-spec override; latent models also use it as d_state.
# baseline_big is scaled to ~33M params to give a *same-size* autoregressive
# reference for the 34M latent models (the "first night win condition" needs
# an equal-parameter comparison, not 2M baseline vs 34M latent).
MODELS = {
    "baseline":             dict(model="baseline",           ls=0, te=0, tps=8, dm=256),
    "baseline_big":         dict(model="baseline_big",       ls=0, te=0, tps=8, dm=768),
    "latent_ssm":           dict(model="latent_ssm",         ls=0, te=0, tps=8, dm=256),
    "latent_ssm_think":     dict(model="latent_ssm",         ls=4, te=4, tps=8, dm=256),
    "latent_ssm_think8":    dict(model="latent_ssm",         ls=4, te=8, tps=8, dm=256),
    "latent_ssm_decoder":   dict(model="latent_ssm_decoder", ls=4, te=4, tps=8, dm=256),
}

QUICK = dict(d_model=32, n_samples=64, epochs=1, max_seq_len=160,
             location_max_chars=200, inventory_max_chars=200,
             transfer_max_chars=200, recall_max_chars=200, eval_every=1, batch_size=16)

# In --quick mode, force the oversized 'baseline_big' (dm=768 hardcoded in
# MODELS) down to the small d_model so the sanity run actually finishes fast.
QUICK_DM_OVERRIDE = True


def build_model(spec: dict, vocab_size: int, d_model: int):
    m = spec["model"]
    dm = spec.get("dm", d_model)
    if m == "baseline":
        return BaselineTransformer(vocab_size=vocab_size, d_model=dm, num_layers=4, nhead=8)
    if m == "baseline_big":
        return BaselineTransformer(vocab_size=vocab_size, d_model=dm, num_layers=6,
                                   nhead=12, dim_ff=2048)
    if m == "latent_ssm":
        return LatentSSM(vocab_size=vocab_size, d_state=dm, d_model=dm,
                         num_ssm_layers=2, latent_steps=spec["ls"], think_every=spec["te"])
    if m == "latent_ssm_decoder":
        return LatentSSMDecoder(vocab_size=vocab_size, d_state=dm, d_model=dm,
                                num_ssm_layers=2, latent_steps=spec["ls"],
                                tokens_per_step=spec["tps"], think_every=spec["te"])
    raise ValueError(m)


def run_one(key: str, args, dataset, device) -> dict:
    spec = MODELS[key]
    dm = spec.get("dm", args.d_model)
    if getattr(args, "quick", False) and QUICK_DM_OVERRIDE:
        dm = args.d_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"STAGE: start exp={key} model={spec['model']} dm={dm} "
          f"steps={spec['ls']} think_every={spec['te']}")

    train_data = dataset[:int(0.8 * len(dataset))]
    val_data = dataset[int(0.8 * len(dataset)):]

    train_texts = [format_for_training(s) for s in train_data]
    val_texts = [format_for_training(s) for s in val_data]
    tokenizer = CharTokenizer(train_texts, max_vocab=256)

    model = build_model(spec, tokenizer.vocab_size, args.d_model).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    config = ExperimentConfig(
        exp_id=key, model=spec["model"], d_model=dm,
        latent_steps=spec["ls"], batch_size=args.batch_size,
        num_epochs=args.epochs, max_seq_len=args.max_seq_len,
        eval_every=args.eval_every, seed=args.seed, device=str(device),
        # Visibility: mid-epoch loss numbers + a fresh generation sample so the
        # user can actually watch the model learn (instead of staring at a
        # blank console for an hour and then a single accuracy=0 result).
        print_every_batches=args.print_every_batches,
        gen_sample_every=args.gen_sample_every,
        # Answer-slot focus: up-weight loss on the "Answer:" continuation so
        # the model gets sharp signal on the *answer* chars, not just filler.
        # 0 disables. 1.0 doubles the loss weight on answer positions.
        answer_loss_weight=args.answer_loss_weight,
    )
    trainer = Trainer(model, config, tokenizer, output_dir=args.output_dir)
    train_loader = DataLoader(TextDataset(train_texts, tokenizer, max_len=args.max_seq_len),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_texts, tokenizer, max_len=args.max_seq_len),
                            batch_size=args.batch_size)

    t0 = time.time()
    metrics = trainer.train(train_loader, val_loader, qa_dataset=val_data)
    elapsed = time.time() - t0

    qa = metrics["eval_accuracy"][-1] if metrics["eval_accuracy"] else None
    final_val = metrics["val_loss"][-1] if metrics["val_loss"] else None
    low_but_useless = bool(final_val is not None and final_val < 2.0
                           and qa is not None and qa["overall_accuracy"] < 0.5)

    # A few visible sample outputs (expected vs generated).
    samples = []
    for s in val_data[:6]:
        if not s.get("question"):
            continue
        prompt = build_prompt(s)
        gen = trainer.generate(prompt, max_tokens=20, temperature=0.0)
        samples.append(dict(task=s["task_type"], question=s["question"],
                            expected=s["answer"], generated=gen.strip()))

    return dict(
        key=key, model=spec["model"], latent_steps=spec["ls"], think_every=spec["te"],
        params=n_params, elapsed_s=elapsed,
        final_train_loss=metrics["train_loss"][-1] if metrics["train_loss"] else None,
        final_val_loss=final_val,
        overall_accuracy=qa["overall_accuracy"] if qa else None,
        task_accuracy=qa["task_accuracy"] if qa else None,
        stratified=qa["stratified"] if qa else None,
        n_eval=qa["n"] if qa else None,
        low_but_useless=low_but_useless,
        samples=samples,
    )
    print(f"STAGE: done exp={key} val_loss={final_val} acc={qa['overall_accuracy'] if qa else None}")


def render_report(results: list) -> str:
    _f = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—"
    lines = []
    lines.append("# Benchmark Report\n")
    lines.append(f"Models evaluated: {len(results)}\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| model | steps | params | train_loss | val_loss | "
                 "exact_acc | low_loss_useless |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        params = f"{r['params']:,}" if isinstance(r["params"], int) else str(r["params"])
        lines.append(
            f"| {r['key']} | {r['latent_steps']} | {params} | "
            f"{_f(r['final_train_loss'])} | {_f(r['final_val_loss'])} | "
            f"{_f(r['overall_accuracy'])} | {'YES' if r['low_but_useless'] else 'no'} |"
        )

    # Per-task accuracy
    lines.append("\n## Per-task exact-match accuracy\n")
    tasks = ["location", "inventory", "transfer", "recall"]
    lines.append("| model | " + " | ".join(tasks) + " |")
    lines.append("|---|" + "|".join(["---"] * len(tasks)) + "|")
    for r in results:
        ta = r["task_accuracy"] or {}
        lines.append("| " + r["key"] + " | " +
                     " | ".join(_f(ta.get(t)) for t in tasks) + " |")

    # Stratified
    lines.append("\n## Stratified by difficulty bucket\n")
    for r in results:
        if not r["stratified"]:
            continue
        lines.append(f"**{r['key']}**")
        for bkey, acc in sorted(r["stratified"].items()):
            lines.append(f"  - {bkey:<26} {_f(acc)}")
        lines.append("")

    # Sample outputs
    lines.append("\n## Sample outputs (expected vs generated)\n")
    for r in results:
        lines.append(f"**{r['key']}** (exact_acc={_f(r['overall_accuracy'])})")
        for s in r["samples"]:
            lines.append(f"  - [{s['task']}] Q: {s['question']}")
            lines.append(f"      expected: {s['expected']!r}")
            lines.append(f"      generated: {s['generated']!r}")
        lines.append("")

    return "\n".join(lines)


def analyze_existing(output_dir: str = "experiments") -> list:
    """Build a report from existing experiment dirs (no training)."""
    results = []
    for d in sorted(Path(output_dir).iterdir()):
        mfile = d / "metrics.json"
        if not mfile.exists():
            continue
        with open(mfile) as f:
            m = json.load(f)
        qa = m["eval_accuracy"][-1] if m.get("eval_accuracy") else None
        final_val = m["val_loss"][-1] if m.get("val_loss") else None
        samples = []
        stxt = d / "samples.txt"
        if stxt.exists():
            txt = stxt.read_text()
            for blk in txt.split("--- Epoch")[1:]:
                for line in blk.splitlines():
                    if line.strip().startswith("Expected:"):
                        samples.append(line.strip())
        results.append(dict(
            key=d.name, model=m.get("config", {}).get("model", "?"),
            latent_steps=m.get("config", {}).get("latent_steps", "?"),
            params="?", elapsed_s=0,
            final_train_loss=m["train_loss"][-1] if m.get("train_loss") else None,
            final_val_loss=final_val,
            overall_accuracy=qa["overall_accuracy"] if qa else None,
            task_accuracy=qa["task_accuracy"] if qa else None,
            stratified=qa["stratified"] if qa else None,
            n_eval=qa["n"] if qa else None,
            low_but_useless=bool(final_val is not None and final_val < 2.0
                                 and qa is not None and qa["overall_accuracy"] < 0.5),
            samples=[dict(task="", question="", expected=s, generated="") for s in samples[:6]],
        ))
    return results


def main():
    ap = argparse.ArgumentParser(description="Reusable latent-state benchmark")
    ap.add_argument("--models", default="baseline,baseline_big,latent_ssm,latent_ssm_think,latent_ssm_decoder",
                    help="Comma-separated model keys from the registry")
    ap.add_argument("--quick", action="store_true", help="Tiny CPU defaults")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=768)
    ap.add_argument("--location_max_chars", type=int, default=600)
    ap.add_argument("--inventory_max_chars", type=int, default=600)
    ap.add_argument("--transfer_max_chars", type=int, default=600)
    ap.add_argument("--recall_max_chars", type=int, default=600)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output_dir", default="experiments")
    ap.add_argument("--analyze", action="store_true", help="Report existing dirs only")
    ap.add_argument("--print_every_batches", type=int, default=50,
                    help="Visible [TRAIN] log every N batches. 0 to disable.")
    ap.add_argument("--gen_sample_every", type=int, default=200,
                    help="Visible [GEN] sample every N batches mid-epoch. 0 to disable.")
    ap.add_argument("--answer_loss_weight", type=float, default=1.0,
                    help="Extra loss weight on answer-slot tokens (focused loss)."
                         " 0 = standard uniform CE; 1.0 = doubles answer-slot loss.")
    args = ap.parse_args()

    if args.analyze:
        results = analyze_existing(args.output_dir)
        report = render_report(results)
        print(report)
        Path("bench_report.md").write_text(report)
        print(f"\nSaved bench_report.md ({len(results)} experiments)")
        return

    if args.quick:
        for k, v in QUICK.items():
            setattr(args, k, v)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    print(f"Device: {device} | models: {keys}\n")
    print(f"STAGE: bench start models={keys} epochs={args.epochs} "
          f"n_samples={args.n_samples} device={device}")

    dataset = generate_dataset(
        n_samples=args.n_samples, seed=args.seed,
        location_max_chars=args.location_max_chars,
        inventory_max_chars=args.inventory_max_chars,
        transfer_max_chars=args.transfer_max_chars,
        recall_max_chars=args.recall_max_chars,
    )

    results = []
    for key in keys:
        if key not in MODELS:
            print(f"Unknown model key: {key} (known: {list(MODELS)})")
            continue
        print(f"\n##### {key} #####")
        try:
            results.append(run_one(key, args, dataset, device))
        except Exception as e:
            print(f"FAILED {key}: {e}")

    report = render_report(results)
    print("\n" + report)
    Path("bench_report.md").write_text(report)
    Path("bench_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved bench_report.md and bench_results.json ({len(results)} models)")
    print("STAGE: done")


if __name__ == "__main__":
    main()
