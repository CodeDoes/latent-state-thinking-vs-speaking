# Working Method

How this project gets run. These rules are derived from the user's verbatim instructions and from what actually went wrong when they were violated.

## Core operating principles

### Observable by default
Every training run logs loss, accuracy, and steps-per-second to stdout and to a file. If you cannot see the curve you cannot diagnose. The user complained once that an experiment "gave no feedback" because loss was invisible. Logs are a deliverable, not a debugging afterthought.

### Resumable
Every training run can be restarted from a checkpoint with `--resume`. A long run is allowed to be interrupted without losing ground.

### Continually improvable
Snapshots are saved and one-day-later runs can build on them. Not a "train once, freeze forever" project.

### Git-bound
Every experiment records `git rev-parse HEAD` in its config.json. The hash is a traceable artifact: "this result was produced with code at commit X." If someone later asks "where did run-007 come from," the answer is in the experiment dir.

### Easy to clear
`experiments/<id>/` is a self-contained run directory. Deleting it removes everything that run produced. No global state scattered across the repo. No surprise side-effects from a half-deleted experiment.

### Single-variable ablations
One variable per experiment. Match every other parameter. Know what changed. If you cannot name the variable you changed in an experiment, you did not do an experiment, you did a stew.

### Theory / experiment / code separation
- `theories/`: prose. What we want, why. No executable code.
- `experiments/`: run results. Logs, configs, checkpoints, samples. Each dir owns itself.
- `src/`: code that runs experiments. Reusable across experiments.
- `reports/`: polished documented summaries written for consumption outside the repo.

### Self-directed research
The AI maintains a proof ledger (`proofs.md`) and proposes the next experiment based on it. The user does not have to drive each step.

### Backend failures are real failures
If an experiment goes bad, fix the code or pivot the theory — don't sweep it under the rug. If the data disproves a claim, archive the theory and replace it. Git history preserves the lost attempt.

### Brainstorming is welcome
The user's verbatim thinking lives in `theories/<name>.md` even when it is messy. The work of making sense of it is mine — interpretation goes in the same file alongside the verbatim where useful.

## Project infrastructure rules

### Single dependency manager
`devenv.nix` and only `devenv.nix`. README.md, requirements.txt have been deleted in favor of this.

### Experiment dirs are git-tracked
Their contents are small (configs, logs, samples). Checkpoints `*.pt` are ignored separately. Wipe a single dir to wipe a single run.

### Use Kaggle when GPU matters; use local when iterations matter
Kaggle for real GPU runs that have to ship. Local devenv-shell for everything CPU. Both have been in use; both have been dropped for stretches. Most small experiments happen on CPU.

### Use safetensors when it is fine
For model weights that are not pickle-needed, `.safetensors` is preferred over `.pt`. Already adopted in `dendrite_rwkv`.

### Smoke tests are mandatory for any new model
A 1-minute CPU run with synthetic data that has a learnable pattern. See [`smoke_test_methodology.md`](smoke_test_methodology.md). If your smoke test does not learn, do not run a longer experiment.

## Workflow for new code

1. Read the theory file your work is supposed to prove.
2. Find or create the experiment id `<theory_short>_<NNN>` (e.g., `dendrite_rwkv_001`).
3. Write a training script in `src/` or `train_*.py` that logs to `experiments/<id>/train.log` and writes `experiments/<id>/config.json` with git hash.
4. Set a CLI budget. For smoke tests, 60-second CPU time is the cap.)
5. Run.
6. Inspect logs. If loss is flat for the whole run, fix or pivot before scaling.
7. Move on. Archive what got superseded.

## What this does not allow

- Multi-change experiments that conflate causes.
- Re-running the same experiment hoping for a different result without first changing the seed and recording why.
- Pretraining-anything larger than ~10M params on CPU.
- Any experiment without a written hypothesis.

## Definition of done for an experiment

- `config.json` present, includes git hash and exact seed.
- `train.log` present, contains at least 3 valid updates showing the loss curve.
- Either a checkpoint, or a documented reason none was saved.
- An honest verdict in the commit message ("proved", "falsified", "no signal") and reference to the relevant theory.

Any experiment missing any of these is not done.
