#!/usr/bin/env python3
"""Generate notebook.ipynb as a self-contained wrapper around train_modules.py.

Kaggle's `kaggle kernels push` uploads ONLY the notebook file (it does not
upload the rest of the working directory). So we cannot rely on bench.py /
src/ being present on Kaggle. Instead, this generator embeds the real
src/*.py and train_modules.py files as `%%writefile` cells, so the notebook
*recreates* them on disk at runtime, then runs train_modules.py as a subprocess.

train_modules.py trains the SEPARABLE latent-state pieces, each with its OWN
objective, in a curriculum:
  Phase 0  token<->state autoencoder           ("output sane words")
  Phase 1  latent algebra, each piece separate:
           make_B(A)->B, make_A(B)->A, continue(A)->A2, continue(B)->B2,
           Answer_in_format_D(A,B,C)->D
The notebook process never imports torch (training is in the subprocess), so
there is no P100 reload / 'triton' double-registration error.
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
    ("src/diagnostics.py", (HERE / "src/diagnostics.py").read_text()),
    ("src/modules.py", (HERE / "src/modules.py").read_text()),
    ("bench.py", (HERE / "bench.py").read_text()),
    ("train_modules.py", (HERE / "train_modules.py").read_text()),
    ("src/latent.py", (HERE / "src/latent.py").read_text()),
    ("train_converged.py", (HERE / "train_converged.py").read_text()),
]


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def writefile_cell(rel_path: str, content: str):
    src = [f"%%writefile {rel_path}\n", content]
    if content and not content.endswith("\n"):
        src.append("\n")
    return code(src)


COMPAT = [
    "import subprocess, sys, os, json, time, random, math\n",
    "from pathlib import Path\n",
    "\n",
    "# IMPORTANT: do NOT import torch here. Training runs in a SEPARATE\n",
    "# subprocess (train_modules.py), so the notebook process only needs the\n",
    "# right torch installed in the environment. Importing torch in this process\n",
    "# and then reloading it after a P100 downgrade causes the\n",
    "# 'Only a single TORCH_LIBRARY can be used to register triton' error.\n",
    "# Detect the GPU via nvidia-smi and, if it's a P100 (sm_60), install a\n",
    "# compatible torch into the env; the train_modules.py subprocess loads it.\n",
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
    "print('Running train_converged.py on CPU (--device cpu): using Kaggle-default torch on CPU; no CUDA, so no P100 torch-downgrade needed.')\n",

    "print('train_modules.py will report the torch version it loads.')\n",
]

SETUP = [
    "# Recreate the reusable src/ package and train_modules.py from the embedded\n",
    "# cells.\n",
    "import os\n",
    "os.makedirs('src', exist_ok=True)\n",
    "open('src/__init__.py', 'w').close()  # optional, enables `import src...`\n",
    "print('workspace files ready:', sorted(os.listdir('.')))\n",
]

RUN = [
    "# Run the modular training (now on disk). Each piece (token<->state, and the\n",
    "# latent algebra make_B/make_A/continue/Answer_in_format_D) is trained with\n",
    "# its OWN objective, in a curriculum: first the autoencoder ('output sane\n",
    "# words'), then the latent ops. Output streams here so kaggle_run.py can\n",
    "# watch STAGE: lines live. Outputs (modules_report.json) land in CWD.\n",
    "import sys, os\n",
    "cmd = (f\"{sys.executable} -u train_converged.py --device cpu \"\n",
    "       f\"--n_samples 5000 --d_state 0 --epochs 20 --max_facts 4\")\n",
    "print('STAGE: launch', cmd)\n",
    "os.system(cmd)\n",
    "print('NOTEBOOK DONE')\n",
]

SUMMARY = [
    "# Show the generated report (also saved as modules_report.json).\n",
    "report = Path('modules_report.json')\n",
    "if report.exists():\n",
    "    print(report.read_text())\n",
    "else:\n",
    "    print('modules_report.json not found (run may have failed).')\n",
]

cells = [
    md([
        "# Hybrid Latent-State Language Model - Modular Training\n",
        "\n",
        "This notebook is a self-contained wrapper around `train_modules.py`, which\n",
        "trains the latent-state pieces SEPARATELY, each with its own objective\n",
        "(curriculum: autoencoder 'output sane words' first, then the latent\n",
        "algebra). It recreates `src/` + `train_modules.py` from embedded cells, then\n",
        "runs train_modules.py as a subprocess.\n",
        "\n",
        "**Hypothesis:** decomposing into trainable input->output maps (make_B(A)->B,\n",
        "continue(A)->A2, Answer_in_format_D(A,B,C)->D, plus token<->state) lets the\n",
        "latent state actually *contain the correct thing* instead of just learning\n",
        "to spell tokens.\n",
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
        "accelerator": "none",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"notebook.ipynb generated - {len(nb['cells'])} cells")
print("Self-contained wrapper: embeds src/ + train_modules.py, runs train_modules.py")
