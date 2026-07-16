# AGENTS.md

Scaffold for architectures separating **thinking** (latent state) from **speaking** (token generation). See `theories/ultimate.md` for philosophy.

## Rules
- Prove one thing at a time — single-variable ablations, matched params.
- No claimed capability without a named mechanism and a controlled disablement that breaks it.
- If an experiment doesn't isolate one property, it's not an experiment.

## Status
`theories/status.md` — live project state.

## Layout
```
src/             # models, datasets, training loops
theories/        # design rationales, experimental proposals
experiments/     # results (gitignored), one dir per run
devenv.nix       # Python + torch deps (the only dependency manager)
```

## Theories
- `<topic>.md` — your write-up (verbatim words from you).
- `<topic>.infer.md` — my interpretation: fills in latent assumptions, highlights what `.md` leaves out. Never repeats `.md` content.
- Read order: `ultimate.md` → `rwkv.md` → `status.md`.

## Workflow
- Models, datasets, training loops stay as separate modules in `src/`.
- Training must be observable (log loss/accuracy/sps to stdout + file), resumable, and record the git commit hash.
- Delete `experiments/<exp_id>/` to fully clear an experiment — no state leaks.
