#!/usr/bin/env python3
"""Run artifacts — the primitive every training script reimplemented.

One Run = one self-contained directory holding everything a run produced:
config.json (seed, git hash), train.log (per-step lines, also echoed),
metrics.json + metrics.jsonl (key numbers), checkpoints.

[meta]
status: active
[/meta]
"""

import datetime
import json
import subprocess
import time
from pathlib import Path


def git_hash() -> str:
    """Short hash of the commit this run was produced with."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


class Run:
    """Handle for one experiment's artifact directory.

    Usage:
        with Run("threads/x/experiments/exp_001", config) as run:
            run.log("step 0 | loss 5.7")
            run.record(best_loss=0.42)
            run.save_checkpoint(model)
    """

    def __init__(self, dir, config: dict, resume: bool = True):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        resume = resume and (self.dir / "config.json").exists()
        self.config = dict(config)
        self.config.setdefault("seed", 42)
        self.config.setdefault("git_hash", git_hash())
        if resume:
            old = json.loads((self.dir / "config.json").read_text())
            old.update(self.config)
            self.config = old
        self.config.setdefault("created", _now())
        self._write(self.dir / "config.json", self.config)
        self.log_path = self.dir / "train.log"
        self.t0 = time.time()

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        self._log_f = self.log_path.open("a")
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self.config["elapsed_s"] = round(time.time() - self.t0, 1)
        self._write(self.dir / "config.json", self.config)
        if getattr(self, "_log_f", None):
            self._log_f.close()

    # -- artifacts ---------------------------------------------------------
    def log(self, line: str, echo: bool = True):
        """One line to train.log and stdout. Raw per-step, easy to grep."""
        if echo:
            print(line)
        self._log_f.write(line + "\n")
        self._log_f.flush()

    def record(self, **metrics):
        """key=value into metrics.json (merged) and metrics.jsonl (appended)."""
        row = {"ts": _now(), **metrics}
        with (self.dir / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        m_path = self.dir / "metrics.json"
        m = json.loads(m_path.read_text()) if m_path.exists() else {}
        m.update(metrics)
        m["updated"] = row["ts"]
        self._write(m_path, m)

    def save_checkpoint(self, model, name: str = "checkpoint.pt"):
        import torch
        path = self.dir / name
        torch.save(model.state_dict(), path)
        return path

    @staticmethod
    def _write(path, data):
        path.write_text(json.dumps(data, indent=2) + "\n")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")
