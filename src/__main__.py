#!/usr/bin/env python3
"""Run experiments via `python -m src <command> ...`

Usage:
    python -m src list                                    # list available experiment types
    python -m src run <exp_id> <exp_type> [--gpu] [flags] # run + tag + save everything
    python -m src resume <exp_id> [--gpu]                 # resume from checkpoint
    python -m src redo <exp_id> [--gpu]                   # re-run from config.json

Every `run` produces:
    experiments/<exp_id>/config.json   — everything needed to reproduce
    experiments/<exp_id>/train.log     — per-step loss trace
    experiments/<exp_id>/metrics.json   — start/end/best loss, params, speed
    experiments/<exp_id>/checkpoint.pt  — final weights
    experiments/<exp_id>/relationships.json — claim links (optional)

Every `run` also calls `_tag.py run` (if available & clean tree) to mint
pre/post git tags so the result is traceable.
"""

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
TAG_SCRIPT = ROOT / "src" / "_tag.py"


# ═══════════════════════════════════════════════════════════════════════
# Registry of known experiment types
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENT_TYPES = {
    "byte_loop": {
        "module": "src.byte_loop_model",
        "class": "ByteLoopModel",
        "defaults": {"dim": 64, "n_layers": 2, "min_len_before_trigger": 8, "max_loops": 4},
        "desc": "Byte-level encoder↔decoder loop with trigger gates",
        "task": "arith",
    },
    "patch_loop": {
        "module": "src.patch_loop_model",
        "class": "PatchLoopModel",
        "defaults": {"dim": 64, "n_layers": 2, "patch_size": 8},
        "desc": "Patch-level encoder→sparse decoder (B7)",
        "task": "patch_repeat",
    },
    "adaptive_loop": {
        "module": "src.byte_loop_model",
        "class": "ByteLoopModel",
        "defaults": {"dim": 64, "n_layers": 2, "min_len_before_trigger": 8, "max_loops": 4},
        "desc": "Adaptive-loop byte-state-byte (B5)",
        "task": "arith",
    },
    "rnn_patch": {
        "module": "src.rnn_patch_model",
        "class": "RNNPatchModel",
        "defaults": {"dim": 64, "n_layers": 2, "patch_size": 8},
        "desc": "Step-function RNN patch model",
        "task": "patch_repeat",
    },
    "shared_state": {
        "module": "src.shared_state_model",
        "class": "SharedStateModel",
        "defaults": {"dim": 64, "n_layers": 2, "patch_size": 8},
        "desc": "Per-step RWKV blocks, separate encoder/patch/decoder",
        "task": "patch_repeat",
    },
    "shared_state_v2": {
        "module": "src.shared_state_model_v2",
        "class": "SharedStateModelV2",
        "defaults": {"dim": 64, "n_layers": 2, "patch_size": 8},
        "desc": "Shared-weights unrolled RWKV (v2)",
        "task": "patch_repeat",
    },
    "dendrite": {
        "module": "src.dendrite_model",
        "class": "DendriteModel",
        "defaults": {"dim": 64, "n_layers": 2, "n_dendrites": 4},
        "desc": "Frozen trunk + LoRA dendrite branches",
        "task": "arith",
    },
    "token_byte_head": {
        "module": "src.token_byte_head_model",
        "class": "TokenByteHeadModel",
        "defaults": {"dim": 64, "n_layers": 2, "vocab_size": 258},
        "desc": "Token-level model with byte output head",
        "task": "arith",
    },
    "b3d": {
        "module": "src.b3d_rwkv_model",
        "class": "B3DRWKVModel",
        "defaults": {"dim": 64, "n_layers": 2},
        "desc": "B3D-RWKV spatial model",
        "task": "grid",
    },
    "diffusion_grid": {
        "module": "src.diffusion_grid_model",
        "class": "DiffusionGridModel",
        "defaults": {"dim": 64, "grid_size": 8},
        "desc": "Grid-based diffusion model",
        "task": "grid",
    },
    "movable_grid": {
        "module": "src.movable_grid_model",
        "class": "MovableGridModel",
        "defaults": {"dim": 64, "grid_size": 8},
        "desc": "Pointer-based movable grid model",
        "task": "grid",
    },
    "injection_freq": {
        "module": "src.injection_freq_model",
        "class": "InjectionFreqModel",
        "defaults": {"dim": 64, "n_layers": 2, "injection_freq": 4},
        "desc": "Controlled injection frequency model",
        "task": "arith",
    },
    "viewport_zoom_pan": {
        "module": "src.viewport_zoom_pan_model",
        "class": "ViewportZoomPanModel",
        "defaults": {"dim": 64, "viewport_size": 8},
        "desc": "Screen viewport with zoom/pan",
        "task": "grid",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Synthetic data generators (learnable patterns)
# ═══════════════════════════════════════════════════════════════════════

TASK_GENERATORS = {}


def _task_arith(n=2000, seed=42):
    """Incrementing bytes: predict next byte."""
    import random
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        start = rng.randint(0, 200)
        length = rng.randint(16, 48)
        # 0=PAD, 1=TRIGGER (reserved), 2..257 = bytes 0..255
        seq = [((start + i) % 256) + 2 for i in range(length)]
        data.append(seq)
    return data
TASK_GENERATORS["arith"] = _task_arith


def _task_patch_repeat(n=2000, seed=42):
    """Patches that repeat (predict same patch repeats)."""
    import random
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        ps = rng.choice([4, 8])
        np_ = rng.randint(3, 6)
        patch = [rng.randint(2, 257) for _ in range(ps)]
        data.append((patch * np_)[:])
    return data
TASK_GENERATORS["patch_repeat"] = _task_patch_repeat


def _task_grid(n=500, seed=42):
    """2D spatial pattern flattened to 1D."""
    import random
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        gs = rng.choice([4, 8])
        flat = [(rng.randint(2, 10)) + 2 for _ in range(gs * gs)]
        data.append(flat)
    return data
TASK_GENERATORS["grid"] = _task_grid


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def _git_tag(name, msg="", cwd=None):
    cwd = cwd or ROOT
    cmd = ["git", "-C", str(cwd), "tag"]
    if msg:
        cmd += ["-m", msg]
    cmd += [name, "HEAD"]
    subprocess.run(cmd, cwd=cwd, capture_output=True, check=False)


def _clean_tree():
    out = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    return not any(l.strip() for l in out.stdout.splitlines() if l.strip())


def _import_model(exp_type):
    info = EXPERIMENT_TYPES[exp_type]
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    return cls, info


def _make_config(exp_type, overrides):
    """Build config dict: defaults + overrides + inferred params.

    Only includes keys the constructor accepts (plus 'vocab_size' if needed).
    """
    cls, info = _import_model(exp_type)
    config = dict(info["defaults"])

    import inspect
    try:
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys())
    except Exception:
        valid_params = set(config.keys())

    # Only pass through overrides that the constructor accepts
    for k, v in overrides.items():
        if k in valid_params:
            config[k] = v

    # Infer vocab_size if constructor expects it
    if "vocab_size" in valid_params and "vocab_size" not in config:
        config["vocab_size"] = 258

    return config


def _instantiate(exp_type, config):
    cls, _ = _import_model(exp_type)
    return cls(**config)


def _count_params(model):
    import torch
    return sum(p.numel() for p in model.parameters())


def _device(gpu):
    import torch
    if gpu and torch.cuda.is_available():
        d = "cuda"
        print(f"  device: {d} ({torch.cuda.get_device_name(0)})")
    else:
        d = "cpu"
        if gpu:
            print("  WARNING: --gpu but CUDA unavailable, falling back to cpu")
        else:
            print(f"  device: {d}")
    return d


def _pad_batch(batch):
    """Pad a list of sequences to same length. Returns (x_list, y_list)."""
    max_len = max(len(s) for s in batch)
    x_list, y_list = [], []
    for seq in batch:
        x = seq[:-1]
        y = seq[1:]
        pad = (max_len - 1) - len(x)
        if pad > 0:
            x = x + [0] * pad
            y = y + [0] * pad
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train(model, data, config, device, log_f):
    """Run training, write to log_f, return metrics dict."""
    import torch
    import torch.nn.functional as F

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    steps = config["steps"]
    batch_size = config["batch_size"]
    log_every = config.get("log_every", 10)
    save_every = config.get("save_every", steps)

    best_loss = float("inf")
    start_loss = None
    start_time = time.time()
    step_times = []

    for step in range(steps):
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            x_list, y_list = _pad_batch(batch)
            x = torch.tensor(x_list, dtype=torch.long, device=device)
            y = torch.tensor(y_list, dtype=torch.long, device=device)

            logits, info = model(x)
            if not isinstance(info, dict):
                logits, info = info, {}

            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        dt = time.time() - t0
        step_times.append(dt)

        if avg_loss < best_loss:
            best_loss = avg_loss
        if start_loss is None:
            start_loss = avg_loss

        elapsed = time.time() - start_time
        line = f"step {step:4d} | loss {avg_loss:.4f} | best {best_loss:.4f} | {dt:.2f}s | {elapsed:.1f}s"
        log_f.write(line + "\n")
        if step % log_every == 0 or step < 3:
            print(line)

        # Save checkpoint periodically
        if save_every > 0 and step > 0 and step % save_every == 0:
            ckpt_path = Path(log_f.name).parent / f"step_{step:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            log_f.write(f"  ckpt: {ckpt_path.name}\n")

    total_time = time.time() - start_time
    metrics = {
        "start_loss": round(start_loss, 4),
        "best_loss": round(best_loss, 4),
        "final_loss": round(avg_loss, 4),
        "steps": steps,
        "total_time_s": round(total_time, 1),
        "avg_step_s": round(sum(step_times) / len(step_times), 3),
        "params": _count_params(model),
    }

    # Save final checkpoint
    torch.save(model.state_dict(), Path(log_f.name).parent / "checkpoint.pt")

    line = f"\nDone. {steps} steps | start {start_loss:.4f} → best {best_loss:.4f} | {total_time:.1f}s"
    log_f.write(line + "\n")
    print(line)

    # Verdict
    if best_loss < start_loss * 0.3:
        verdict = "LEARNED"
    elif best_loss < start_loss * 0.8:
        verdict = "PARTIAL"
    else:
        verdict = "FLAT"
    metrics["verdict"] = verdict
    print(f"Verdict: {verdict}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════

def cmd_list(args):
    """List available experiment types."""
    print(f"{'Name':20s} {'Task':14s} {'Description'}")
    print("-" * 60)
    for name in sorted(EXPERIMENT_TYPES):
        info = EXPERIMENT_TYPES[name]
        print(f"{name:20s} {info['task']:14s} {info['desc']}")
    print()
    print(f"Total: {len(EXPERIMENT_TYPES)} types")


def cmd_run(args):
    """Run + tag + save. The single entry point for all experiments."""
    exp_id = args.id
    exp_type = args.type

    if exp_type not in EXPERIMENT_TYPES:
        print(f"Unknown type '{exp_type}'. Use `python -m src list`")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_id}  ({exp_type})")
    print(f"{'='*60}")

    # Create dir
    exp_dir = EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    overrides = {}
    for k in ("dim", "n_layers", "patch_size", "max_loops", "lr", "steps", "batch_size", "seed", "n_samples", "log_every", "save_every"):
        v = getattr(args, k, None)
        if v is not None:
            overrides[k] = v
    overrides["task"] = args.task if args.task else EXPERIMENT_TYPES[exp_type]["task"]

    model_kwargs = _make_config(exp_type, overrides)

    config = {
        "exp_id": exp_id,
        "type": exp_type,
        "description": args.desc or f"{exp_type} run",
        "git_hash": _git_hash(),
        "steps": overrides.get("steps", 500),
        "batch_size": overrides.get("batch_size", 32),
        "lr": overrides.get("lr", 3e-4),
        "seed": overrides.get("seed", 42),
        "n_samples": overrides.get("n_samples", 2000),
        "log_every": overrides.get("log_every", 10),
        "save_every": overrides.get("save_every", 0),
        "task": overrides.get("task", EXPERIMENT_TYPES[exp_type]["task"]),
        **model_kwargs,
    }

    # Write config
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  config: {exp_dir / 'config.json'}")

    # Model
    model = _instantiate(exp_type, model_kwargs)
    n_params = _count_params(model)
    config["params"] = n_params
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    # Update model_kwargs back into config dict keys the model constructor can use
    for k in model_kwargs:
        config[k] = model_kwargs[k]
    print(f"  params: {n_params:,}")

    # Device
    device = _device(args.gpu)

    # Data
    task_gen = TASK_GENERATORS.get(config["task"])
    if task_gen is None:
        print(f"Unknown task '{config['task']}'")
        sys.exit(1)
    data = task_gen(n=config["n_samples"], seed=config["seed"])
    print(f"  data:   {len(data)} samples, task={config['task']}")

    # Pre-tag if clean tree
    tagged = False
    if _clean_tree() and TAG_SCRIPT.exists():
        try:
            tag_result = subprocess.run(
                ["python", str(TAG_SCRIPT), "run", "--id", exp_id, "--topic", exp_type, "--pre-only",
                 "--dry-run"], cwd=ROOT, capture_output=True, text=True, check=False)
            # We don't actually use _tag.py's run (it's complex), just do lightweight tags ourselves
            _git_tag(f"exp/{exp_type}/{exp_id}-pre", f"pre-run: {exp_id}", cwd=ROOT)
            tagged = True
            print(f"  tag:    exp/{exp_type}/{exp_id}-pre")
        except Exception:
            pass

    # Train
    log_path = exp_dir / "train.log"
    with open(log_path, "w") as log_f:
        log_f.write(f"Experiment: {exp_id} ({exp_type}) @ {config['git_hash']}\n")
        log_f.write(f"Config: {json.dumps(config)}\n\n")
        metrics = train(model, data, config, device, log_f)

    # Write metrics.json
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  metrics: {exp_dir / 'metrics.json'}")

    # Update config with final metrics
    config.update(metrics)
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Pre-tag if clean
    if tagged:
        try:
            subprocess.run(["git", "-C", str(ROOT), "add", "--force", f"experiments/{exp_id}"], check=False)
            subprocess.run(
                ["git", "-C", str(ROOT), "commit", "-m", f"experiment {exp_id}: {config['description']}"],
                check=False, capture_output=True,
            )
            _git_tag(f"exp/{exp_type}/{exp_id}-post", f"post-run: {exp_id}", cwd=ROOT)
            print(f"  tag:    exp/{exp_type}/{exp_id}-post")
        except Exception:
            pass

    print(f"{'='*60}\n")


def cmd_resume(args):
    """Resume an experiment from its checkpoint."""
    exp_id = args.id
    exp_dir = EXPERIMENTS_DIR / exp_id
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_id}")
        sys.exit(1)

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"No config.json in {exp_dir}")
        sys.exit(1)

    config = json.loads(config_path.read_text())
    exp_type = config.get("type", "byte_loop")
    ckpt_path = exp_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint.pt in {exp_dir} — cannot resume")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"RESUME: {exp_id}  ({exp_type})")
    print(f"{'='*60}")

    import torch
    model = _instantiate(exp_type, config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    n_params = _count_params(model)
    print(f"  loaded checkpoint: {ckpt_path}")
    print(f"  params: {n_params:,}")

    # Extend config
    config["steps"] = (config.get("steps", 0) or 0) + getattr(args, "extra_steps", 500)
    config["git_hash"] = _git_hash()

    device = _device(args.gpu)

    task_gen = TASK_GENERATORS.get(config.get("task", "arith"))
    data = task_gen(n=config.get("n_samples", 2000), seed=config.get("seed", 42))
    print(f"  data: {len(data)} samples")

    log_path = exp_dir / "train.log"
    with open(log_path, "a") as log_f:
        log_f.write(f"\n--- RESUME @ {config['git_hash']} ---\n")
        metrics = train(model, data, config, device, log_f)

    config.update(metrics)
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{'='*60}\n")


def cmd_redo(args):
    """Re-run an experiment from its config.json under a new id."""
    src_id = args.id
    src_dir = EXPERIMENTS_DIR / src_id
    if not src_dir.exists():
        print(f"Experiment not found: {src_id}")
        sys.exit(1)

    config_path = src_dir / "config.json"
    if not config_path.exists():
        print(f"No config.json in {src_dir}")
        sys.exit(1)

    config = json.loads(config_path.read_text())
    exp_type = config.get("type", "byte_loop")
    new_id = f"{exp_type}_rerun_{int(time.time())}"
    if getattr(args, "new_id", None):
        new_id = args.new_id

    print(f"Re-running {src_id} as {new_id}")

    # Override config with fresh params
    config["git_hash"] = _git_hash()
    config["description"] = f"re-run of {src_id}"

    # Create fresh args and call cmd_run
    class FakeArgs:
        pass
    fa = FakeArgs()
    fa.id = new_id
    fa.type = exp_type
    fa.gpu = args.gpu
    fa.desc = config.get("description", f"re-run {src_id}")
    fa.task = config.get("task", EXPERIMENT_TYPES[exp_type]["task"])
    for k in ("dim", "n_layers", "patch_size", "max_loops", "lr", "steps", "batch_size", "seed", "n_samples", "log_every", "save_every"):
        setattr(fa, k, config.get(k, None))

    cmd_run(fa)


def cmd_prove(args):
    """Record a proof in proofs.md and link an experiment to a claim."""
    claim = args.claim_id
    exp_id = args.exp_id
    exp_dir = EXPERIMENTS_DIR / exp_id
    if not exp_dir.exists():
        print(f"No such experiment: {exp_id}")
        sys.exit(1)

    metrics_path = exp_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    config_path = exp_dir / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    verdict = metrics.get("verdict", "UNKNOWN")
    best_loss = metrics.get("best_loss", "?")
    n_params = metrics.get("params", "?")

    proofs_path = ROOT / "theories" / "proofs.md"
    if args.file:
        proofs_path = Path(args.file)

    # Build proof line
    claim_short = claim.replace("theo/", "")
    gh = config.get("git_hash", _git_hash())[:7]
    line = f"- `{exp_id}` ({gh}): {args.text or f'claim {claim}'}. Loss {best_loss}, {n_params} params. Verdict: {verdict}.\n"

    # Append to proofs.md
    if proofs_path.exists():
        content = proofs_path.read_text()
        content += line
        proofs_path.write_text(content)
        print(f"Appended to {proofs_path}")
    else:
        print(f"(no proofs.md at {proofs_path})")
        print(f"Proof line: {line.strip()}")

    # Write relationships.json
    rel_path = exp_dir / "relationships.json"
    rel_data = {}
    if rel_path.exists():
        rel_data = json.loads(rel_path.read_text())
    supports = rel_data.get("supports", [])
    if claim not in supports:
        supports.append(claim)
    rel_data["supports"] = supports
    rel_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rel_path.write_text(json.dumps(rel_data, indent=2))
    print(f"Linked {exp_id} → {claim} in {rel_path}")

    # Try git tag
    try:
        _git_tag(claim, msg=f"proof: {args.text or claim}", cwd=ROOT)
        print(f"Tagged {claim}")
        subprocess.run(["git", "-C", str(ROOT), "add", str(proofs_path), str(rel_path)], check=False)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(
        description="Run, resume, redo experiments. Every run saves everything.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m src list
  python -m src run byte_loop_003 byte_loop --gpu --steps 500
  python -m src run dendrite_002 dendrite --gpu --steps 200 --dim 32
  python -m src resume byte_loop_003 --gpu --extra-steps 200
  python -m src redo byte_loop_001 --gpu
  python -m src prove theo/token_surgery/T1 token_surgery_full "core transfers through surgery"
""",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ── list ──
    p_list = sub.add_parser("list", help="list experiment types")
    p_list.set_defaults(func=cmd_list)

    # ── run ──
    p_run = sub.add_parser("run", help="run + tag + save")
    p_run.add_argument("id", help="experiment id (e.g. byte_loop_003)")
    p_run.add_argument("type", choices=sorted(EXPERIMENT_TYPES), help="experiment type")
    p_run.add_argument("--gpu", action="store_true", help="use GPU")
    p_run.add_argument("--desc", help="description")
    p_run.add_argument("--steps", type=int, help="training steps")
    p_run.add_argument("--batch-size", type=int, help="batch size")
    p_run.add_argument("--lr", type=float, help="learning rate")
    p_run.add_argument("--n-samples", type=int, help="samples")
    p_run.add_argument("--seed", type=int, help="random seed")
    p_run.add_argument("--task", choices=sorted(TASK_GENERATORS), help="override task")
    p_run.add_argument("--dim", type=int, help="model dim")
    p_run.add_argument("--n-layers", type=int, help="layers")
    p_run.add_argument("--patch-size", type=int, help="patch size")
    p_run.add_argument("--max-loops", type=int, help="max loops")
    p_run.add_argument("--log-every", type=int, default=10, help="log interval")
    p_run.add_argument("--save-every", type=int, default=0, help="checkpoint interval (0 = end only)")
    p_run.set_defaults(func=cmd_run)

    # ── resume ──
    p_res = sub.add_parser("resume", help="resume from checkpoint")
    p_res.add_argument("id", help="experiment id")
    p_res.add_argument("--gpu", action="store_true", help="use GPU")
    p_res.add_argument("--extra-steps", type=int, default=500, help="additional steps")
    p_res.set_defaults(func=cmd_resume)

    # ── redo ──
    p_re = sub.add_parser("redo", help="re-run from config.json under new id")
    p_re.add_argument("id", help="experiment id to redo")
    p_re.add_argument("--new-id", help="new experiment id (default: <type>_rerun_<ts>)")
    p_re.add_argument("--gpu", action="store_true", help="use GPU")
    p_re.set_defaults(func=cmd_redo)

    # ── prove ──
    p_pr = sub.add_parser("prove", help="record a proof in proofs.md + link + tag")
    p_pr.add_argument("claim_id", help="theo/<topic>/<CID> (e.g. theo/token_surgery/T1)")
    p_pr.add_argument("exp_id", help="experiment that proved it")
    p_pr.add_argument("text", nargs="?", default="", help="one-line proof description")
    p_pr.add_argument("--file", help="proofs.md path (default: theories/proofs.md)")
    p_pr.set_defaults(func=cmd_prove)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
