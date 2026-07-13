# Theory 08 — Gradual novelty + local-GPU fast iteration

## Claim
Novel architecture components should be introduced **gradually** with fast
local iteration loops (not an unproven all-at-once structure pushed to Kaggle),
and the local machine **has a usable GPU** (RTX 2050, 4 GB) that makes this
viable.

## Rationale
- Pushing every incremental experiment to Kaggle (push/wait/monitor/download)
  is slow and discourages iteration.
- The project's real struggle was **over-engineering an unproven structure on
  an unguarded script** (`train_converged.py`: no answer-focusing loss, no
  sample dumps, no stratified eval) rather than using the guarded end-to-end
  `bench.py` (`BaselineTransformer` + `answer_loss_weight` + stratified eval +
  `low_but_useless` flag).

## Supporting evidence (2026-07-13)
- Local GPU discovered: **NVIDIA GeForce RTX 2050**, driver CUDA 13.2, torch
  2.5.1+cu121. `cuda_available` was `False` only because `LD_LIBRARY_PATH` did
  not include `/usr/lib/x86_64-linux-gnu` (the driver's `libcuda.so`); fixed in
  `devenv.nix` (mirrors `../roco_ai/devenv.nix`).
- Measured throughput (batch 16, seq 256):
  - d_model 256 (bench default): **63,894 tok/s**, 0.40 GB VRAM
  - d_model 768 / 33.5M params (`baseline_big`, the 20M-scale reference):
    **10,749 tok/s**, 1.67 GB VRAM
- Extrapolated: a 5k-sample × 20-epoch `bench.py` run ≈ **15–20 min locally**;
  a 20M-param run ≈ **1–1.5 hr**. RTX 2050 ≈ T4-class for these sizes.

## Predictions (testable)
- With local GPU, incremental novelty experiments (Model A→D path) validate in
  minutes.
- The guarded `bench.py` baseline should be established first; novelty
  (`latent_ssm`, `latent_ssm_think`) compared fairly on the same benchmark.

## Proposed experiments
- Local: `bench.py --models baseline --device cuda` (establish clean baseline).
- Then `latent_ssm` / `latent_ssm_think` on the same benchmark for fair
  comparison.
- Reserve Kaggle for the largest / long-horizon win-condition runs and parallel
  sweeps.

## Status
✅ **Confirmed** (GPU available + throughput measured); workflow in adoption.
Overrides the stale AGENTS.md "no local GPU" assumption — local GPU is now a
legitimate fast-iteration surface for ≤20M-param experiments.

## Related
T01 (core hypothesis), T02 (ssm-vs-tape split).
