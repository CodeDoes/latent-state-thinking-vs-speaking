#!/usr/bin/env python3
"""Generate notebook.ipynb as a thin wrapper around the reusable bench.py.

Design (per AGENTS.md rule 7): the notebook must NOT duplicate model code.
It imports `src/` (uploaded alongside it) and runs `bench.py`, which is the
single entry point for train + strict exact-match QA eval + report. Improving
`src/` or `bench.py` therefore automatically improves the Kaggle run.

bench.py emits `STAGE:` lines every epoch so `kaggle_run.py` can monitor live
progress and fail-fast on tracebacks.
"""
import json


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


COMPAT = [
    "import subprocess, sys, os, json, time, random, math\n",
    "from pathlib import Path\n",
    "\n",
    "# IMPORTANT: do NOT import torch here. Training runs in a SEPARATE\n",
    "# subprocess (bench.py), so the notebook process only needs the right\n",
    "# torch installed in the environment. Importing torch in this process and\n",
    "# then reloading it after a P100 downgrade causes the\n",
    "# 'Only a single TORCH_LIBRARY can be used to register triton' error.\n",
    "# Detect the GPU via nvidia-smi and, if it's a P100 (sm_60), install a\n",
    "# compatible torch into the env; the bench.py subprocess will load it.\n",
    "\n",
    "def gpu_capability():\n",
    "    try:\n",
    "        out = subprocess.check_output(\n",
    "            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],\n",
    "            stderr=subprocess.DEVNULL).decode().strip().splitlines()\n",
    "        if out:\n",
    "            return float(out[0].strip())\n",
    "    except Exception:\n",
    "        pass\n",
    "    return None\n",
    "\n",
    "cap = gpu_capability()\n",
    "print('GPU compute capability:', cap)\n",
    "if cap is not None and cap < 7.0:\n",
    "    print('P100 (sm_60) detected -> installing torch 2.3.1+cu118...')\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n",
    "        '--index-url', 'https://download.pytorch.org/whl/cu118',\n",
    "        'torch==2.3.1', 'torchvision==0.18.1'])\n",
    "else:\n",
    "    print('Using Kaggle-default torch (T4 / newer GPU).')\n",
    "print('bench.py will report the torch version it loads.')",
]

RUN = [
    "# Run the reusable benchmark. It imports src/ and does train + strict\n",
    "# exact-match QA eval + report. Output streams here so kaggle_run.py can\n",
    "# watch STAGE: lines live. Outputs land in CWD (/kaggle/working) and are\n",
    "# preserved as notebook outputs: experiments/<exp>/..., bench_report.md.\n",
    "import sys, os\n",
    "\n",
    "# Matched-parameter battery for the 'first night' win condition:\n",
    "#   baseline      ~2M   (efficient AR reference)\n",
    "#   baseline_big  ~33M  (AR baseline at the SAME size as the latent models)\n",
    "#   latent_ssm       34M, thinking=0   (isolates the 'thinking' effect)\n",
    "#   latent_ssm_think 34M, think every 4 (main hypothesis)\n",
    "#   latent_ssm_decoder 34M, multi-token decoder\n",
    "MODELS = 'baseline,baseline_big,latent_ssm,latent_ssm_think,latent_ssm_decoder'\n",
    "cmd = (f\"{sys.executable} bench.py --models {MODELS} \"\n",
    "       f\"--epochs 20 --n_samples 5000 --device cuda --d_model 256 --eval_every 1\")\n",
    "print('STAGE: launch', cmd)\n",
    "os.system(cmd)\n",
    "print('NOTEBOOK DONE')",
]

SUMMARY = [
    "# Show the generated report (also saved as bench_report.md).\n",
    "report = Path('bench_report.md')\n",
    "if report.exists():\n",
    "    print(report.read_text())\n",
    "else:\n",
    "    print('bench_report.md not found (run may have failed).')\n",
]

nb = {
    "cells": [
        md([
            "# Hybrid Latent-State Language Model — Benchmark\n",
            "\n",
            "This notebook is a thin wrapper around `bench.py`, the reusable entry point\n",
            "for training, strict exact-match QA evaluation, and reporting. It imports\n",
            "the shared `src/` code (uploaded with this notebook), so improving `src/`\n",
            "or `bench.py` automatically improves this run.\n",
            "\n",
            "**Hypothesis:** `latent_state_update() × N` + `decode_token() × M` beats\n",
            "token-by-token next-token prediction on long-horizon reasoning, at equal\n",
            "parameter count.\n",
        ]),
        code(COMPAT),
        code(RUN),
        code(SUMMARY),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"notebook.ipynb generated — {len(nb['cells'])} cells")
print("Wrapper around bench.py (imports src/, strict QA eval, STAGE: monitoring)")
