# Agent Instructions: Hybrid Latent-State Language Model

## Project Overview

This is an autonomous research project exploring whether a neural architecture can separate **thinking** from **speaking** — maintaining a private latent computational space while using language only as an output device.

**Core hypothesis:** A model with `latent_state_update() × N` + `decode_token() × M` can outperform an equivalent token-by-token model on long-horizon reasoning.

**First night win condition:** A 20M parameter latent-state model with 4 recurrent thinking steps achieves better long-horizon recall than a same-size autoregressive model.

---

## File Guide

| File | Purpose |
|---|---|
| `CONVO.md` | Full conversation history (~2000 lines) with ChatGPT tracing the idea's evolution |
| `USER.md` | Refined architecture specification (hybrid cognitive architecture with SSM + Tape + Context + Decoder) |
| `USER_base.md` | Autonomous research loop specification and agent behavior rules |
| `PROGRESS.md` | Current project status, research log, and experiment tracking |
| `PLAN.md` | Detailed phased research plan, research ladder, and deliverables |
| `devenv.nix` / `devenv.yaml` | Nix development environment configuration |

---

## Architecture

```
              INPUT TOKENS
                   |
                   v
          Context Manager
                   |
      +------------+-------------+
      |                          |
      v                          v
Prefix Tape Memory          SSM State
  exact recall              semantic state
  token patterns            planning
  spelling                  logic
  names                     world model
      |                          |
      +------------+-------------+
                   |
                   v
          Latent Processor
        (SSM + FFN loop)
                   |
                   v
        State -> Token Decoder
                   |
                   v
                OUTPUT
```

**Component roles:**
- **SSM:** "What does this mean?" — logic, planning, world model
- **Tape:** "What exactly was written?" — exact token recall, spelling
- **Context:** "What am I currently saying?" — attention management
- **Decoder:** "How do I express the state?" — rendering state to tokens

**Key insight:** A token is a terrible clock cycle for reasoning. The latent state becomes the scratchpad.

---

## Research Ladder

Each level must be proven before advancing:

| Level | Question |
|---|---|
| 0 | Does latent state work at all? |
| 1 | Does latent thinking beat tokens? |
| 2 | Does latent state survive context removal? |
| 3 | Can latent state generate multiple tokens? |
| 4 | Can latent state continue a story after interruption? |
| 5 | Can state computation become independent of token generation? |

---

## Agent Behavior

You are an autonomous ML researcher. Continue working until compute budget is exhausted.

### Rules
1. Build baseline → run experiments → record results → keep improvements → generate hypotheses → repeat
2. Create a unique experiment ID for every experiment
3. Record: hypothesis, code changes, hyperparameters, training loss, evaluation scores, generation samples, conclusions
4. **Never overwrite previous results**
5. Save checkpoint, metrics, and sample outputs after each experiment
6. **RUN EVERY EXPERIMENT ON KAGGLE — NEVER ON LOCAL CPU.** Local hardware here has no usable GPU (CUDA is unavailable) and is far too slow for real training, so any experiment executed locally is wasted compute and produces misleading "too slow / runs forever" results. The workflow is:
   - Develop and validate code locally with **fast CPU syntax/import/sanity checks only** (tiny `n_samples`, 1 epoch, small `d_model`). Do NOT treat local runs as real experiments.
   - Push to Kaggle and execute there: `python kaggle_push.py --run` (pushes `notebook.ipynb` + the `src/` directory and runs it on a free GPU). For a scripted battery, push and run `bench.py` the same way (Kaggle runs `.py` files in the pushed directory).
   - Monitor with `python kaggle_push.py --monitor` and download with `python kaggle_push.py --download` into `kaggle_output/`.
   - The notebook already writes model files + `results.json` + `samples.json` + `loss_curves.png` into `/kaggle/working/`, which Kaggle preserves as Output; pull those down to analyze.
   - Analyze downloaded results with `python bench.py --analyze` (or `python analyze_experiments.py`) — do not re-train locally to "check".
7. **Reusable entry points (do not hand-patch the notebook JSON):** training/benchmarking goes through `bench.py` (single entry point for train + strict eval + report; `--quick` for local sanity, full battery on Kaggle with `--device cuda`). The Kaggle notebook imports the same `src/` code, so improving `src/` automatically improves the Kaggle run. If the notebook must change, edit `src/` and re-push — do not edit notebook cells by hand.

### Experiment Structure
```
experiments/
    exp001/
        config.json
        metrics.json
        samples.txt
        model.pt
    exp002/
    ...

results.json
best_model.pt
research_log.md
```

### Kaggle Constraints
- **ALL experiments run on Kaggle GPUs, never locally.** Local runs are only for code sanity checks.
- Design around checkpointing — free accelerators are limited, sessions can stop
- Save models, metrics, and samples frequently (notebook writes to `/kaggle/working/`)
- Commit results after each experiment
- Push/run via `python kaggle_push.py --run`; monitor via `--monitor`; download via `--download`

---

## Training Objectives

Standard next-token prediction encourages shortcutting. Implement:
1. **Latent consistency loss** — state after thinking ≈ state after observing answer
2. **Token reconstruction loss** — can decoder recover tokens from state?
3. **State evolution loss** — state should remain predictive over time (JEPA-style)
4. **Specialization pressure** — punish SSM for memorizing names/passwords, reward for reasoning

---

## Phases

See `PLAN.md` for full details. Summary:

1. **Phase 1: Baseline** — Tiny transformer LM + recurrent latent model
2. **Phase 2: Synthetic Tasks** — Toy world generator, memory/recall/story datasets
3. **Phase 3: Experiments** — Model A (SSM only) through Model D (full architecture)
4. **Phase 4: Improvements** — Gated updates, separate vectors, consistency losses

---

## Research Questions

1. Does more latent computation improve reasoning?
2. Can token generation be cheaper than state computation?
3. Can latent state preserve information after context removal?
4. Can the model resume generation from latent_state alone?

---

## Reference

[JEPA-Reasoner: Decoupling Latent Reasoning from Token Generation](https://arxiv.org/abs/2512.19171)
