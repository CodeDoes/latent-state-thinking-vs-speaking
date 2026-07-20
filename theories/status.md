# status

Live project state is generated, not hand-maintained:

- `python src/_status.py --all` — theories, experiments, code, git state
- `experiments/INDEX.md` — every run, its thread, its verdict
- `src/CODEMAP.md` — every module, scraped from `[meta]` docstrings
- `proofs.md` — the claim ledger (Supported / Refuted)

## Governance (what still lives here)

- [`ultimate_thesis.md`](ultimate_thesis.md) — the project's one-paragraph
  framing. (`ultimate.md` was merged in 2026-07-20.)
- [`proofs.md`](proofs.md) — the ledger. Entries need claim, theory, runs,
  baseline, numbers, prior-art Δ, commit.
- [`method/`](method/) — operating methods (smoke tests, tagging, loss
  design, and `working_method.md`, the history of why the rules exist).
  The enforced contract is `AGENTS.md`.
- [`archive/`](archive/) — cross-thread dead ends. Thread-local dead ends
  live in `threads/<slug>/archive/`.

## Where questions live now

Theories moved **into their threads** (2026-07-20 reorganization):
`threads/<slug>/` holds each question's spec docs, code, and runs together.
See [`threads/README.md`](../threads/README.md) for the tier model and the
thread index.

Read order for a newcomer: `AGENTS.md` → `ultimate_thesis.md` →
`threads/README.md` → one thread's docs → its `experiments/`.
