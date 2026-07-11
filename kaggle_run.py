#!/usr/bin/env python3
"""
Reusable Kaggle runner + monitor + analyzer for the latent-state experiments.

This is the SINGLE local entry point. It:
  1. Runs a FAST LOCAL pre-flight self-check (imports src/, builds a tiny
     dataset + tokenizer, runs a forward pass on every model) so code errors
     are caught on the local machine BEFORE we waste a Kaggle GPU run.
  2. Pushes notebook.ipynb (+ src/) to Kaggle and triggers the run.
  3. Polls Kaggle, printing the notebook STATUS and which STAGE it is at
     (dataset / train exp001 ep N/30 / eval / save / done), and aborts early
     if a traceback or failure appears (fail-fast, no wasted GPU hours).
  4. On completion, downloads outputs and runs `bench.py --analyze`.

Usage:
    python kaggle_run.py                       # pre-flight + push + monitor + report
    python kaggle_run.py --max_wait 7200       # wait up to 2h for completion
    python kaggle_run.py --no-push             # monitor an already-pushed run
    python kaggle_run.py --download-only       # just download latest outputs + analyze
"""
import argparse
import os
import subprocess
import sys
import time

KERNEL = "kitastro/hybrid-latent-state-language-model"
OUT_DIR = "kaggle_output"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 1) Local pre-flight self-check (fast, CPU, tiny)
# ---------------------------------------------------------------------------
def pre_flight():
    print(">> local pre-flight self-check (imports + tiny forward)...")
    try:
        from src.dataset import generate_dataset, format_for_training, build_prompt
        from src.tokenizer import CharTokenizer
        from src.models import BaselineTransformer, LatentSSM, LatentSSMDecoder
        import torch

        ds = generate_dataset(n_samples=10, seed=0)
        tok = CharTokenizer([format_for_training(s) for s in ds], max_vocab=256)
        specs = {
            "baseline": lambda: BaselineTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=2, nhead=8),
            "latent_ssm": lambda: LatentSSM(vocab_size=tok.vocab_size, d_state=32, d_model=32, num_ssm_layers=2, latent_steps=2, think_every=2),
            "latent_ssm_decoder": lambda: LatentSSMDecoder(vocab_size=tok.vocab_size, d_state=32, d_model=32, num_ssm_layers=2, latent_steps=2, tokens_per_step=4, think_every=2),
        }
        for name, mk in specs.items():
            m = mk()
            lg = m(torch.zeros(1, 8, dtype=torch.long))
            assert lg.dim() == 3 and lg.size(2) == tok.vocab_size, lg.shape
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
    for line in r.stdout.splitlines():
        if "status" in line.lower():
            return line.strip()
    return r.stdout.strip()


def logs():
    return sh(f"kaggle kernels logs {KERNEL}").stdout


def latest_stage(logtext):
    stages = [l.strip() for l in logtext.splitlines() if "STAGE:" in l]
    return stages[-1] if stages else "(unknown)"


# ---------------------------------------------------------------------------
# 3) Monitor (status + stage, fail-fast)
# ---------------------------------------------------------------------------
def monitor(max_wait):
    print(f">> monitoring (up to {max_wait}s) -- reporting status + stage...")
    t0 = time.time()
    last_stage = "(none)"
    while time.time() - t0 < max_wait:
        L = logs()
        s = status()
        st = latest_stage(L)
        if st != last_stage:
            last_stage = st
        # fail-fast on errors
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
    return None  # still running


# ---------------------------------------------------------------------------
# 4) Download + analyze
# ---------------------------------------------------------------------------
def download_and_analyze():
    print(f">> downloading outputs to {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    sh(f"kaggle kernels output {KERNEL} -p {OUT_DIR}")
    # The notebook writes modules_report.json (modular run) and/or
    # bench_report.md (monolithic run). Show whichever exists.
    rep = Path(OUT_DIR) / "modules_report.json"
    if rep.exists():
        print(">> modules_report.json:")
        print(rep.read_text())
    else:
        print(">> analyzing with bench.py --analyze ...")
        r = sh(f"{sys.executable} bench.py --analyze --output_dir {OUT_DIR}/experiments")
        print(r.stdout[-4000:])
        if r.returncode != 0:
            print("analyze stderr:\n", r.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_wait", type=int, default=7200, help="seconds to wait for COMPLETE")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-preflight", action="store_true")
    args = ap.parse_args()

    if args.download_only:
        download_and_analyze()
        return

    if not args.no_preflight:
        if not pre_flight():
            print("ABORT: local pre-flight failed. Fix the code before pushing.")
            sys.exit(1)

    if not args.no_push:
        push()

    result = monitor(args.max_wait)
    if result is True:
        download_and_analyze()
    elif result is False:
        print("ABORT: run failed on Kaggle (see logs above).")
        sys.exit(1)
    else:
        print("Run still in progress on Kaggle. Re-run `python kaggle_run.py --no-push` "
              "later to download + analyze.")


if __name__ == "__main__":
    main()
