#!/usr/bin/env python3
"""
Single reusable control script for all Kaggle interactions.

Wraps the raw `kaggle kernels ...` CLI calls (which are easy to get wrong) into
one tool with subcommands:

  python kaggle_ctl.py status               # kernel status
  python kaggle_ctl.py logs                 # fetch + show STAGE: lines
  python kaggle_ctl.py download             # pull outputs -> kaggle_output/
  python kaggle_ctl.py report               # print modules_report.json / bench_report.md
  python kaggle_ctl.py run [--max_wait N]   # pre-flight + push + monitor + download + report
  python kaggle_ctl.py watch [--max_wait N] # monitor an already-pushed run, then download+report

The notebook is a self-contained wrapper (embeds src/ + the training script via
%%writefile), so `run` just pushes notebook.ipynb and lets Kaggle execute it.
Monitoring reads STAGE: lines and fails fast on a traceback.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

KERNEL = "kitastro/hybrid-latent-state-language-model"
OUT_DIR = "kaggle_output"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Local pre-flight self-check (fast, CPU, tiny)
# ---------------------------------------------------------------------------
def pre_flight():
    print(">> local pre-flight self-check (imports + tiny forward)...")
    try:
        from src.dataset import generate_dataset, format_for_training, build_prompt
        from src.tokenizer import CharTokenizer
        from src.modules import TokenEncoder, StateDecoder, StateTransform, AnswerComposer
        import torch
        ds = generate_dataset(n_samples=10, seed=0)
        tok = CharTokenizer([format_for_training(s) for s in ds], max_vocab=256)
        enc = TokenEncoder(tok.vocab_size, d_state=32)
        s = enc.state_of(torch.zeros(1, 8, dtype=torch.long))
        assert s.dim() == 2, s.shape
        print("SELFCHECK_OK (local)")
        return True
    except Exception as e:
        print("SELFCHECK_FAIL (local):", repr(e))
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Kaggle helpers
# ---------------------------------------------------------------------------
def push():
    print(">> pushing notebook + src/ to Kaggle (triggers run)...")
    r = sh("kaggle kernels push -p .")
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("PUSH FAILED:\n", r.stderr[-1500:])
        sys.exit(1)


def status():
    r = sh(f"kaggle kernels status {KERNEL}")
    line = [l for l in r.stdout.splitlines() if "status" in l.lower()]
    return line[0] if line else r.stdout.strip()


def logs(tail_n=40):
    r = sh(f"kaggle kernels logs {KERNEL}")
    txt = r.stdout
    stages = []
    for line in txt.splitlines():
        if "STAGE:" in line:
            try:
                import json
                o = json.loads(line)
                stages.append(o.get("data", ""))
            except Exception:
                stages.append(line)
    print(f">> {len(stages)} STAGE lines:")
    for s in stages[-25:]:
        print("  ", s)
    print(f">> last {tail_n} raw log lines:")
    print("\n".join(txt.splitlines()[-tail_n:]))


def download():
    print(f">> downloading outputs to {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    sh(f"kaggle kernels output {KERNEL} -p {OUT_DIR}")


def report():
    rep = Path(OUT_DIR) / "modules_report.json"
    if rep.exists():
        print(rep.read_text())
        return
    rep = Path(OUT_DIR) / "bench_report.md"
    if rep.exists():
        print(rep.read_text())
        return
    print("no report found in", OUT_DIR, "(run `download` first)")


def latest_stage(logtext):
    stages = [l.strip() for l in logtext.splitlines() if "STAGE:" in l]
    return stages[-1] if stages else "(unknown)"


def monitor(max_wait):
    print(f">> monitoring (up to {max_wait}s) -- reporting status + stage...")
    t0 = time.time()
    last_stage = "(none)"
    while time.time() - t0 < max_wait:
        L = logs_text()
        s = status()
        st = latest_stage(L)
        if st != last_stage:
            last_stage = st
        if "Traceback (most recent call last)" in L:
            print("!! TRACEBACK in logs -- aborting. Last logs:")
            print(L[-3000:])
            return False
        if "SELFCHECK_FAIL" in L:
            print("!! SELFCHECK_FAIL -- aborting.")
            return False
        if "COMPLETE" in s:
            print(f">> run COMPLETE. final stage: {last_stage}")
            return True
        if "ERROR" in s or "CANCELLED" in s:
            print(f"!! run ended with status: {s}")
            return False
        print(f"  status={s}  stage={last_stage}")
        time.sleep(30)
    print(f">> still running after wait window. last stage: {last_stage}")
    return None


def logs_text():
    return sh(f"kaggle kernels logs {KERNEL}").stdout


def download_and_analyze():
    download()
    report()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Reusable Kaggle control for the latent-state experiments")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("status").set_defaults(func=lambda a: print(status()))
    p_logs = sub.add_parser("logs")
    p_logs.add_argument("--tail", type=int, default=40)
    p_logs.set_defaults(func=lambda a: logs(a.tail))
    sub.add_parser("download").set_defaults(func=lambda a: download())
    sub.add_parser("report").set_defaults(func=lambda a: report())

    p_run = sub.add_parser("run")
    p_run.add_argument("--max_wait", type=int, default=7200)
    p_run.set_defaults(func=lambda a: _run(a.max_wait))

    p_watch = sub.add_parser("watch")
    p_watch.add_argument("--max_wait", type=int, default=7200)
    p_watch.set_defaults(func=lambda a: _watch(a.max_wait))

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    args.func(args)


def _run(max_wait):
    if not pre_flight():
        print("ABORT: local pre-flight failed. Fix the code before pushing.")
        sys.exit(1)
    push()
    result = monitor(max_wait)
    if result is True:
        download_and_analyze()
    elif result is False:
        print("ABORT: run failed on Kaggle (see logs above).")
        sys.exit(1)
    else:
        print("Run still in progress on Kaggle. Re-run `python kaggle_ctl.py watch` later.")


def _watch(max_wait):
    result = monitor(max_wait)
    if result is True:
        download_and_analyze()
    elif result is False:
        print("ABORT: run failed on Kaggle (see logs above).")
        sys.exit(1)
    else:
        print("Run still in progress on Kaggle. Re-run `python kaggle_ctl.py watch` later.")


if __name__ == "__main__":
    main()
