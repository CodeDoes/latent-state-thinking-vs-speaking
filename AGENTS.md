# Agent Instructions: Hybrid Latent-State Language Model

## Project Overview

This is an autonomous research project exploring whether a neural architecture can separate **thinking** from **speaking** — maintaining a private latent computational space while using language only as an output device.

**Core hypothesis:** A model with `latent_state_update() × N` + `decode_token() × M` can outperform an equivalent token-by-token model on long-horizon reasoning.

**First night win condition:** A 20M parameter latent-state model with 4 recurrent thinking steps achieves better long-horizon recall than a same-size autoregressive model.

---

## File Guide

| File | Purpose |
|---|---|
| `CONVO.md` | (removed — content consolidated into AGENTS.md / PROGRESS.md) |
| `USER.md` | (removed — architecture spec consolidated into AGENTS.md) |
| `USER_base.md` | (removed — agent-behavior rules consolidated into AGENTS.md) |
| `PROGRESS.md` | Current project status, research log, and experiment tracking |
| `docs/PLAN.md` | Detailed phased research plan, research ladder, and deliverables |
| `devenv.nix` / `devenv.yaml` | Nix development environment configuration |
| `src/modules.py` | Separable latent-state pieces (TokenEncoder/Decoder, StateTransform for make_B/make_A/continue, AnswerComposer, ReasoningStep, ContextManager, Tape) |
| `train_modules.py` | Modular training: each piece trained separately, curriculum (autoencoder to latent algebra) |
| `bench.py` | Monolithic baseline comparison: train + strict exact-match QA eval + report |
| `gen_notebook.py` | Regenerates `notebook.ipynb` (embeds `src/*.py` + training script via `%%writefile`) |
| `kaggle_ctl.py` | **Single Kaggle control script**: `status`/`logs`/`download`/`report`/`run`/`watch` |
| `kaggle_run.py` | (legacy) older push/monitor script; superseded by `kaggle_ctl.py` |
| `modules_foundation.pt` | Cached autoencoder (Phase 0); `train_modules.py` reloads it so the foundation is not retrained each run |

---

## Tooling: What to Use & Maintain

This repo has accumulated several overlapping scripts. To avoid confusion (and
accidentally extending a dead script), here is the canonical split.

### ✅ Canonical — use these and keep them maintained

| Tool | Role | When to use |
|---|---|---|
| `src/` | **Source of truth.** `modules.py` (separable latent algebra pieces), `models.py` (monolithic baselines: `BaselineTransformer`, `LatentSSM`, `LatentSSMDecoder`), `trainer.py`, `dataset.py` (toy-world tasks), `tokenizer.py`. | Every change to model/training logic goes here. Both `train_modules.py` and `bench.py` import from `src/`, and the Kaggle notebook embeds these files via `%%writefile`. |
| `bench.py` | The single benchmark: train + **strict exact-match** QA eval (per task & by difficulty bucket) + rich report. | Comparing model variants (baseline vs latent). **`--quick` ONLY on local** (confirm STAGE: prints + imports work); `--models ... --device cuda` for any real battery (run on Kaggle via `kaggle_ctl.py run`); `--analyze` (report downloaded experiment dirs). **Never invoke `bench.py` without `--quick` locally** — see Guardrails. |
| `train_modules.py` | Trains the separable latent-state pieces, each with its OWN objective (Phase 0 autoencoder → Phase 1 latent algebra). | Embeds into the Kaggle notebook. Do not execute locally — push via `gen_notebook.py` + `kaggle_ctl.py run`. |
| `gen_notebook.py` | Regenerates `notebook.ipynb` by embedding `src/*.py` + `train_modules.py` as `%%writefile` cells. | **Always run this before pushing** whenever `src/` or `train_modules.py` changes — the notebook is the only thing Kaggle receives. Do NOT hand-edit notebook cells. |
| `notebook.ipynb` | Self-contained Kaggle wrapper; recreates `src/` on disk then runs `train_modules.py` as a subprocess. Writes `results.json`/`samples.json`/`loss_curves.png`/`best_model.pt` to `/kaggle/working/`. | Install/run target on Kaggle. Treat as generated output (rebuild via `gen_notebook.py`). |
| `kaggle_ctl.py` | **Single Kaggle control script.** `status` / `logs` / `download` / `report` / `run` (pre-flight + push + monitor + download + report) / `watch` (monitor an already-pushed run). | The one Kaggle interaction tool to use. Fails fast on `STAGE:` tracebacks. |

**Golden workflow (canonical):**
1. Edit `src/` (and/or `train_modules.py`).
2. `python bench.py --quick` — **fast local CPU sanity ONLY** (tiny `n_samples`, 1–2 epochs, small `d_model`). Purpose: catch import errors and confirm STAGE: prints show up. **NOT a real experiment — never run `bench.py` without `--quick` locally.**
3. `python gen_notebook.py` — rebuild `notebook.ipynb` from current source.
4. `python kaggle_ctl.py run --max_wait N` — push + monitor + download + report on a Kaggle GPU. **All real training happens here.**
5. Pull `/kaggle/working/` outputs into `kaggle_output/`; analyze with `bench.py --analyze`.

### 🧊 Legacy — frozen, do NOT extend or rely on

These exist for historical continuity only. They are superseded by the
canonical tools above. If you need their behavior, reuse the canonical script
instead of reviving these.

| File | Why frozen |
|---|---|
| `kaggle_run.py` | Older push/monitor; superseded by `kaggle_ctl.py`. Its header still claims to be "the SINGLE entry point" — that is stale; `kaggle_ctl.py` is authoritative. |
| `kaggle_push.py` | Legacy push/monitor for experiments. Use `kaggle_ctl.py run` instead. |
| `run_experiment.py` | Older local experiment runner (pre-`src/` refactor). Use `bench.py` / `train_modules.py`. |
| `run_complete_workflow.py` | Older end-to-end workflow wrapper. Use `kaggle_ctl.py run`. |
| `analyze_results.py` | Older analyzer. Use `bench.py --analyze`. |
| `download_kaggle_results.py` | Older downloader. Use `kaggle_ctl.py download`. |
| `kaggle_notebook.py` | Pre-`gen_notebook.py` notebook generator. Use `gen_notebook.py`. |
| `auto_download.sh`, `monitor_kaggle.sh`, `wait_and_download.sh` | Shell glue around the old flow. Use `kaggle_ctl.py`. |

### Maintenance rules
- **Single source of truth:** all model/training logic lives in `src/`; never fork logic into a standalone script.
- **Notebook is generated:** never hand-edit `notebook.ipynb`; change `src/` then `python gen_notebook.py`.
- **One Kaggle tool:** all Kaggle interaction through `kaggle_ctl.py`.
- **Never delete legacy files outright** (they document the project's evolution) — just stop using/maintaining them.
- **Local = sanity only.** Real experiments run on Kaggle GPUs (see Agent Behavior).

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
6. **RUN EVERY EXPERIMENT ON KAGGLE — NEVER ON LOCAL CPU.** Local hardware here has no usable GPU (CUDA is unavailable) and is far too slow for real training, so any experiment executed locally is wasted compute and produces misleading "too slow / runs forever" results. Concrete rules:
   - The ONLY command you may run on local CPU is `python bench.py --quick` (or `${PYTHON} ... --quick`) and ONLY to (a) catch import errors and (b) confirm the trainer stages print (DATA / TOKEN / INIT / TRAIN / EVAL / GEN). Local is a debug surface, not a measurement surface.
   - **You must NEVER invoke `python bench.py` without `--quick` on the local machine,** even "just to check." Even `--quick` is for sanity only. Any battery with realistic `--n_samples`, `--epochs`, or `--d_model` comparable to a real experiment **must go to Kaggle.**
   - Develop and validate code locally with **fast CPU syntax/import/sanity checks only** (tiny `n_samples`, 1–2 epochs, small `d_model`). Do NOT treat local runs as real experiments.
   - **If you ever feel tempted to inspect a non-`--quick` bench run or retrain anything here locally, stop. Push to Kaggle first** via `python kaggle_ctl.py run --max_wait N` (the canonical tool).
   - **Visualize training:** `bench.py` now emits `STAGE:` (DATA / TOKEN / INIT / TRAIN / EVAL / GEN) and mid-epoch `[TRAIN]` lines (loss_full + loss_answer) so you can SEE learning happen. The local `--quick` run is enough to confirm those prints show up before pushing to Kaggle — no need to run anything heavier here.
   - Push to Kaggle and execute there: `python kaggle_ctl.py run --max_wait N` (pushes `notebook.ipynb` + the `src/` directory and runs it on a free GPU). For a scripted battery, push and run `bench.py` the same way. **Use `kaggle_ctl.py`, not `kaggle_push.py` / `kaggle_run.py` (legacy).**
   - Monitor with `python kaggle_ctl.py watch` and download with `python kaggle_ctl.py download` into `kaggle_output/`.
   - The notebook already writes model files + `results.json` + `samples.json` + `loss_curves.png` into `/kaggle/working/`, which Kaggle preserves as Output; pull those down to analyze.
   - Analyze downloaded results with `python bench.py --analyze` (or `python analyze_experiments.py`) — do not re-train locally to "check".
7. **Reusable entry points (do not hand-patch the notebook JSON):** training/benchmarking goes through `bench.py` (single entry point for train + strict eval + report; `--quick` for local sanity, full battery on Kaggle with `--device cuda`). The Kaggle notebook imports the same `src/` code, so improving `src/` automatically improves the Kaggle run. If the notebook must change, edit `src/` and re-push — do not edit notebook cells by hand.

### Guardrails — non-negotiable

These are explicit, mechanical guards the agent must respect:

- **No local real training.** Any invocation of `bench.py` without `--quick` on the local machine is forbidden (waste of compute, produces misleading results, demoralizing). If you find yourself wanting to "just run one more epoch," push to Kaggle instead.
- **No hand-editing `notebook.ipynb`.** Just edit `src/` and run `gen_notebook.py`. The notebook is generated output.
- **No silently changing evaluation.** Cross-entropy going down does **not** mean the model is correct. Always report (a) strict exact-match QA accuracy and (b) a sample of expected vs generated text per model. If accuracy is 0 while loss is low, **flag it explicitly in PROGRESS.md** as "low loss but useless output" and diagnose (DIAGNOSTICS section in `bench.py`/modules_report) before "improving" anything.
- **Every Kaggle run must produce** `results.json` + `samples.json` + `metrics.json` + `loss_curves.png` in `/kaggle/working/`. If any of these is missing, the run is incomplete and the output should not be analyzed.
- **Never delete experiments.** Append-only `experiments/expNNN/`. If a run failed, keep the failure record; do not amputate history.
- **Latent-only reality check.** The user's first question (“why is training invisible?”) and second (“how can accuracy be exactly 0 if I have a baseline?”) both come from the same root cause: the trainer used to log only one number per epoch and the dataset was too hard for the model's capacity at that scale. The `--quick` runs now print STAGE: DATA/TOKEN/INIT/TRAIN/EVAL/GEN and mid-epoch `[TRAIN]` lines so you can watch learning live — use those.

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
- **ALL experiments run on Kaggle GPUs, never locally.** Local runs are only for code sanity checks (see Agent Behavior rule 6 above). The only legal local invocation is `python bench.py --quick` to confirm STAGE: prints and catch import errors.
- Design around checkpointing — free accelerators are limited, sessions can stop
- Save models, metrics, and samples frequently (notebook writes to `/kaggle/working/`)
- Commit results after each experiment
- Push/run via `python kaggle_ctl.py run --max_wait N`; monitor via `kaggle_ctl.py watch`; download via `kaggle_ctl.py download`

---

## Training Objectives

Standard next-token prediction encourages shortcutting. Implement:
1. **Latent consistency loss** — state after thinking ≈ state after observing answer
2. **Token reconstruction loss** — can decoder recover tokens from state?
3. **State evolution loss** — state should remain predictive over time (JEPA-style)
4. **Specialization pressure** — punish SSM for memorizing names/passwords, reward for reasoning

---

## Phases

See `docs/PLAN.md` for full details. Summary:

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
