#!/usr/bin/env python3
"""Check the status of experiments, theories, and code at a glance.

Usage:
    python src/_status.py [--all] [--theory] [--experiment] [--code]

Default is `--all` (everything). Each section respects the layout
documented in AGENTS.md:

    theories/        — single-file theory docs, no separate status subfile
    experiments/     — run directories (one per experiment id)
    src/             — code modules, organized by what they serve

The script prints discovered items grouped by theory, with notes on
each experiment's state.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def list_files(dirpath, suffix, hide_gone=True):
    """List files in `dirpath` with `suffix`, sorted alphabetically."""
    p = ROOT / dirpath
    if not p.exists():
        return []
    out = sorted(
        f for f in p.glob(f"*{suffix}")
        if not (hide_gone and f.name.startswith("."))
    )
    return out


def git_hash_short():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=ROOT,
        ).decode().strip()
    except Exception:
        return "no-git"


def git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=ROOT,
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


# ── Theories ────────────────────────────────────────────────────────────

THEORY_BLACKLIST = {"status.md", "proofs.md", "dump.rwkv-rosa-dendrite.txt"}
THEORY_SUBDIRS = (
    "architecture", "memory", "adaptive", "spatial",
    "core", "method", "application", "analysis",
)


def discover_theories():
    """Discover theory files. Top-level files first, then per-subdir."""
    out = []
    # Top-level theories (ultimate, proofs, status, etc.)
    p = ROOT / "theories"
    if p.exists():
        for f in sorted(p.glob("*.md")):
            if f.name in THEORY_BLACKLIST:
                continue
            if f.name.startswith("status") or f.name.startswith("."):
                continue
            blurb = _first_blurb(f)
            out.append((f.name, blurb, f))
    # Subdirectories
    for sub in THEORY_SUBDIRS:
        sp = ROOT / "theories" / sub
        if not sp.exists():
            continue
        out.append(("", f"── {sub}/ ──", None))
        for f in sorted(sp.glob("*.md")):
            blurb = _first_blurb(f)
            out.append((f.name, blurb, f))
    return out


def _first_blurb(path):
    """First non-empty, non-# line of a file, truncated."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped[:120]
    return ""


def print_theories():
    print("== THEORIES ====================================================")
    theories = discover_theories()
    if not theories:
        print("  (none)")
    for name, blurb, _ in theories:
        print(f"  • {name:42s} {blurb}")


# ── Experiments ─────────────────────────────────────────────────────────


def exp_status(exp_dir: Path):
    """Read signals from an experiment's directory and emit a quick verdict."""
    out = []
    cfg = exp_dir / "config.json"
    log = exp_dir / "train.log"
    ckpt = exp_dir / "checkpoint.pt"
    rel = exp_dir / "relationships.json"
    final_loss = None
    final_acc = None
    steps = None
    git_hash_at_run = None
    tag = None
    supports = []
    notes = ""

    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            steps = data.get("steps") or data.get("total_steps")
            git_hash_at_run = data.get("git_hash") or data.get("commit")
            tag = data.get("tag")
        except Exception:
            notes += "[bad-config]"

    if rel.exists():
        try:
            data = json.loads(rel.read_text())
            supports = data.get("supports", [])
        except Exception:
            pass

    if log.exists():
        log_text = log.read_text(errors="replace")
        # Find best loss or last reported value
        for line in log_text.splitlines():
            m = re.search(r"(?i)loss[=:\s]+([\d\.]+)", line)
            if m:
                final_loss = float(m.group(1))
            m = re.search(r"(?i)acc(?:uracy)?[=:\s]+([\d\.]+)", line)
            if m:
                final_acc = float(m.group(1))
            m = re.search(r"step\s+(\d+)/(\d+)", line)
            if m:
                steps = int(m.group(2))

    if ckpt.exists():
        out.append("ckpt")
    if cfg.exists():
        out.append("cfg")
    if log.exists():
        out.append("log")
    else:
        notes += "[no-log] "

    verdict = "?"
    if final_loss is not None and final_acc is not None:
        verdict = f"loss={final_loss:.3f} acc={final_acc:.3f}"
    elif final_loss is not None:
        verdict = f"loss={final_loss:.3f}"
    elif final_acc is not None:
        verdict = f"acc={final_acc:.3f}"
    elif steps:
        verdict = f"steps={steps}"
    else:
        verdict = "no metrics"

    return (
        ", ".join(out) or "empty",
        verdict,
        git_hash_at_run or "no-git-hash",
        tag or "no-tag",
        supports,
        notes.strip(),
    )


def discover_experiments():
    experiments = []
    p = ROOT / "experiments"
    if not p.exists():
        return []
    for f in sorted(p.iterdir()):
        if not f.is_dir():
            continue
        if f.name.startswith("."):
            continue
        # Skip archive-like dirs that have no config
        cfg = f / "config.json"
        log = f / "train.log"
        metric = f / "metrics.json"
        if not (cfg.exists() or log.exists() or metric.exists()):
            continue
        experiments.append(f)
    return experiments


def print_experiments():
    print("\n== EXPERIMENTS ================================================")
    exps = discover_experiments()
    if not exps:
        print("  (none)")
    for e in exps:
        flags, verdict, git_at, tag, supports, notes = exp_status(e)
        prefix = "·"
        if notes:
            prefix = "!"
        print(f"  {prefix} {e.name:32s} [{flags:11s}] {verdict:32s} "
              f"(run@{git_at}) tag={tag} {notes}")
        for s in supports:
            print(f"    └─ supports {s}")


# ── Code ────────────────────────────────────────────────────────────────


CODE_BLACKLIST = {"__init__.py", "_status.py"}


def discover_modules():
    """Group Python modules in src/ by their primary purpose."""
    groups = defaultdict(list)
    p = ROOT / "src"
    if not p.exists():
        return {}
    for f in sorted(p.glob("*.py")):
        if f.name.startswith("_"):
            continue
        if f.name in CODE_BLACKLIST:
            continue
        # Group by training script associated, else by guess
        if "train" in f.name:
            groups["training"].append(f.name)
        elif "model" in f.name:
            groups["models"].append(f.name)
        elif f.name.endswith("_generator.py") or "data" in f.name:
            groups["data"].append(f.name)
        elif "rwkv" in f.name.lower() and "lora" in f.name.lower():
            groups["peft"].append(f.name)
        elif "rwkv" in f.name.lower():
            groups["core"].append(f.name)
        else:
            groups["other"].append(f.name)
    return groups


def print_code():
    print("\n== CODE (src/) =================================================")
    groups = discover_modules()
    if not groups:
        print("  (no modules)")
    for g in sorted(groups):
        files = groups[g]
        if not files:
            continue
        print(f"  [{g}]")
        for name in files:
            print(f"    - {name}")


# ── Provenance ──────────────────────────────────────────────────────────


def print_provenance():
    print("\n== GIT =========================================================")
    h = git_hash_short()
    d = git_dirty()
    state = " (dirty)" if d else ""
    print(f"  HEAD: {h}{state}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Status of theories, experiments, code",
    )
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--code", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-provenance", action="store_true")
    args = ap.parse_args()

    show_all = args.all or (
        not (args.theory or args.experiment or args.code)
    )

    if args.theory or show_all:
        print_theories()

    if args.experiment or show_all:
        print_experiments()

    if args.code or show_all:
        print_code()

    if not args.no_provenance:
        print_provenance()


if __name__ == "__main__":
    main()
