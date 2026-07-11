#!/usr/bin/env python3
"""Generate notebook.ipynb as a self-contained wrapper around the reusable bench.py.

Kaggle's `kaggle kernels push` uploads ONLY the notebook file (it does not
upload the rest of the working directory). So we cannot rely on bench.py /
src/ being present on Kaggle. Instead, this generator embeds the real
src/*.py and bench.py files as `%%writefile` cells, so the notebook
*recreates* them on disk at runtime, then runs bench.py as a subprocess.

Benefits:
  * Single source of truth: editing src/ or bench.py and re-running this
    generator updates the embedded copies (AGENTS.md rule 7: improve src/,
    regenerate notebook, don't hand-patch cells).
  * Training runs in a separate `bench.py` subprocess, so the notebook
    process never imports torch -> no P100 reload / 'triton' double-registration
    error.
  * bench.py emits STAGE: lines every epoch so kaggle_run.py can monitor live.

The COMPAT cell detects the GPU via nvidia-smi (no torch import) and, only if
it is a P100 (sm_60), installs a compatible torch into the env. The bench.py
subprocess then loads whatever torch is present.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# Files to embed (real source of truth). Order matters for package imports.
# Note: no src/__init__.py needed (namespace package on sys.path); we create an
# empty one in the SETUP cell for safety. Empty %%writefile cells error out.
EMBED = [
    ("src/dataset.py", (HERE / "src/dataset.py").read_text()),
    ("src/tokenizer.py", (HERE / "src/tokenizer.py").read_text()),
    ("src/models.py", (HERE / "src/models.py").read_text()),
    ("src/trainer.py", (HERE / "src/trainer.py").read_text()),
    ("bench.py", (HERE / "bench.py").read_text()),
]


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def writefile_cell(rel_path: str, content: str):
    src = [f"%%writefile {rel_path}\n", content]
    # Ensure the file ends with a newline for clean writing.
    if content and not content.endswith("\n"):
        src.append("\n")
    return code(src)


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
    "print('bench.py will report the torch version it loads.')\n",
]

SETUP = [
    "# Recreate the reusable src/ package and bench.py from the embedded cells.\n",
    "import os\n",
    "os.makedirs('src', exist_ok=True)\n",
    "open('src/__init__.py', 'w').close()  # optional, enables `import src...`\n",
    "print('workspace files ready:', sorted(os.listdir('.')))\n",
]

RUN = [
    "# Run the reusable benchmark (now on disk). It imports src/ and does train\n",
    "# + strict exact-match QA eval + report. Output streams here so kaggle_run.py\n",
    "# can watch STAGE: lines live. Outputs land in CWD (/kaggle/working) and are\n",
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
    "print('NOTEBOOK DONE')\n",
]

SUMMARY = [
    "# Show the generated report (also saved as bench_report.md).\n",
    "report = Path('bench_report.md')\n",
    "if report.exists():\n",
    "    print(report.read_text())\n",
    "else:\n",
    "    print('bench_report.md not found (run may have failed).')\n",
]

cells = [
    md([
        "# Hybrid Latent-State Language Model — Benchmark\n",
        "\n",
        "This notebook is a self-contained wrapper around `bench.py`, the reusable\n",
        "entry point for training, strict exact-match QA evaluation, and reporting.\n",
        "It recreates `src/` + `bench.py` from embedded cells (so improving `src/`\n",
        "or `bench.py` and regenerating this notebook keeps them in sync), then runs\n",
        "bench.py as a subprocess.\n",
        "\n",
        "**Hypothesis:** `latent_state_update() × N` + `decode_token() × M` beats\n",
        "token-by-token next-token prediction on long-horizon reasoning, at equal\n",
        "parameter count.\n",
    ]),
    code(COMPAT),
    code(SETUP),
]

for rel_path, content in EMBED:
    cells.append(writefile_cell(rel_path, content))

cells.append(code(RUN))
cells.append(code(SUMMARY))

nb = {
    "cells": cells,
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
print("Self-contained wrapper: embeds src/ + bench.py via %%writefile, runs bench.py")
