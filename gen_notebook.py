#!/usr/bin/env python3
"""Generate notebook.ipynb as a self-contained wrapper around train_converged.py.

Kaggle's `kaggle kernels push` uploads ONLY the notebook file (it does not
upload the rest of the working directory). So we cannot rely on bench.py /
src/ being present on Kaggle. Instead, this generator embeds the real
src/*.py and train_converged.py files as `%%writefile` cells, so the notebook
*recreates* them on disk at runtime, then runs train_converged.py as a
subprocess.

train_converged.py trains BOTH the latent model (sequential SSM think +
FFN speak) and an equal-capacity token-by-token baseline on the same
long-horizon reasoning task, then reports val exact-match per query type
for each -- the actual "latent thinking vs tokens" test.

GPU strategy (robust, no network hang):
  * We do NOT pip-install a P100 torch (the pytorch-index cu118
    download HUNG on Kaggle's network).
  * Instead we detect the GPU capability and pick the device:
      - T4 / sm>=7.0 : the default cu121 torch works -> train on CUDA.
      - P100 / sm<7.0 : cu121 torch would crash at runtime, so we
        fall back to CPU (reliable, no install needed).
  * The notebook process never imports torch (training is a subprocess), so
    there is no triton double-registration crash.
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
    "# subprocess (train_converged.py), so the notebook process only needs the\n",
    "# right torch installed. We do NOT downgrade torch (the P100 cu118\n",
    "# pip install from the pytorch index HUNG on Kaggle's network).\n",
    "# Instead we pick the device from the GPU capability:\n",
    "#   - T4 / sm>=7.0 : default cu121 torch works -> train on CUDA\n",
    "#   - P100 / sm<7.0 : cu121 torch would crash -> fall back to CPU (no install)\n",
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
    "DEVICE = 'cuda' if (cap is not None and cap >= 7.0) else 'cpu'\n",
    "print('GPU compute capability:', cap, '-> training device:', DEVICE)\n",
]

SETUP = [
    "# Recreate the reusable src/ package and train_converged.py from the embedded\n",
    "# cells.\n",
    "import os\n",
    "os.makedirs('src', exist_ok=True)\n",
    "open('src/__init__.py', 'w').close()  # optional, enables `import src...`\n",
    "print('workspace files ready:', sorted(os.listdir('.')))\n",
]

RUN = [
    "# Run the converged training (now on disk). It trains BOTH the latent\n",
    "# model (sequential SSM think + FFN speak) and an equal-capacity\n",
    "# token-by-token baseline on the same long-horizon reasoning task, then\n",
    "# reports val exact-match per query type (WHERE / AT / SAME) for each.\n",
    "# The latent model builds its state ONCE from the source (think once,\n",
    "# speak many); the baseline re-encodes the full source for every query.\n",
    "# DEVICE is chosen in the COMPAT cell (cuda on T4, cpu on P100).\n",
    "import sys, os\n",
    "cmd = (f\"{sys.executable} -u train_converged.py --device {DEVICE} \"\n",
    "       f\"--n_samples 2500 --epochs 14 --K 2 \"\n",
    "       f\"--d_emb 128 --d_hidden 256 --d_state 48 --max_events 16\")\n",
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
        "# Hybrid Latent-State Language Model - Converged Design\n",
        "\n",
        "Self-contained wrapper around `train_converged.py`, which trains BOTH the\n",
        "latent model (sequential SSM think + FFN speak) and an equal-capacity\n",
        "token-by-token baseline on the same **long-horizon** reasoning task, then\n",
        "reports val exact-match per query type (WHERE / AT / SAME) for each --\n",
        "the 'latent thinking vs tokens' test. The latent model builds its state\n",
        "ONCE from the source (think once, speak many); the baseline re-encodes\n",
        "the full source for every question.\n",
        "\n",
        "Device: CUDA on T4 (sm>=7.0, default cu121 torch); CPU on P100\n",
        "(sm<7.0) where cu121 would crash -- no torch downgrade (the pytorch\n",
        "index cu118 install HUNG on Kaggle's network).\n",
        "\n",
        "**Hypothesis:** separating thinking (latent state updates) from speaking\n",
        "(token generation) lets the model answer many queries from one compressed\n",
        "state, beating an equal-size autoregressive model on long-horizon reasoning.\n",
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
        "accelerator": "gpu",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("notebooks/notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"notebooks/notebook.ipynb generated - {len(nb['cells'])} cells")
print("Self-contained wrapper: embeds src/ + train_converged.py, runs train_converged.py")
