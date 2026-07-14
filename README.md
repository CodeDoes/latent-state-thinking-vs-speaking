# Latent-State / Thinking-vs-Speaking Experiments

Clean scaffold for researching architectures that separate latent-state
("thinking") computation from token generation ("speaking").

## Setup
```bash
devenv up          # or: direnv allow   (provides Python + torch)
```
Verify the environment:
```bash
.devenv/state/venv/bin/python -c "import torch; print(torch.__version__)"
```

## Layout
- `src/` — your models, datasets, and training loops (a Python package).
- `experiments/` — results; one directory per experiment (gitignored).
- `requirements.txt` / `devenv.nix` — dependency definitions.

## Prior work
The complete earlier implementation, datasets, docs, and design notes are
preserved in git at commit `028f2a3`. This folder was reset to a minimal base
so a new architecture can be dropped in cleanly — nothing was deleted from
history.

## Conventions
- Put each experiment's code under `src/`.
- Write `config.json`, `metrics.json`, and `samples.txt` into
  `experiments/<exp_id>/`.
- Keep local runs small; scale up only once a run is sane.
