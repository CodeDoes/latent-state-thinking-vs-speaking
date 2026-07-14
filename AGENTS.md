# AGENTS.md

Clean local-experimentation scaffold for researching architectures that
separate **thinking** (latent-state computation) from **speaking**
(token generation).

## Current state
This is a minimal base for a *new* architecture. The full prior
implementation, datasets, docs, and design notes are preserved in git history
at commit `028f2a3` ("Pre-experiment snapshot"). Recover any old file with
`git show 028f2a3:<path>` or `git checkout 028f2a3 -- <path>` — nothing was lost.

## Layout
```
src/             # your models, datasets, training loops (Python package)
experiments/     # results (gitignored); one dir per run, e.g. experiments/exp001/
requirements.txt # python deps (torch, numpy, tqdm, matplotlib)
devenv.nix       # provides the Python + torch environment
.gitignore       # ignores experiments/, *.pt, __pycache__, etc.
```

## Local workflow
1. Develop in `src/` — keep model, dataset, and training loop as separate modules.
2. Sanity-check on CPU with tiny runs before any heavier training.
3. Log every experiment to `experiments/<exp_id>/` (config + metrics + samples).
4. Commit progress frequently; never overwrite prior results.

## Environment
Managed by devenv (`devenv up`, or `direnv allow`). The interpreter lives at
`.devenv/state/venv/bin/python`. Scale up (GPU / Kaggle) only after a local
run is sane.
