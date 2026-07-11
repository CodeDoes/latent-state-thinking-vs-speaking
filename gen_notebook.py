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
    "from collections import Counter\n",
    "from dataclasses import dataclass, field\n",
    "from typing import List, Dict, Optional, Tuple\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from tqdm import tqdm\n",
    "\n",
    "# Kaggle ships torch 2.10+cu128 which dropped P100 (sm_60) support.\n",
    "# Install torch 2.3.1+cu118 which still works on P100.\n",
    "if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:\n",
    "    print(f'P100 detected, installing torch 2.3.1+cu118...')\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n",
    "        '--index-url', 'https://download.pytorch.org/whl/cu118',\n",
    "        'torch==2.3.1', 'torchvision==0.18.1'])\n",
    "    import importlib; importlib.reload(torch)\n",
    "\n",
    "print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')\n",
    "if torch.cuda.is_available():\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}, cap: {torch.cuda.get_device_capability()}')\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Device: {device}')",
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
