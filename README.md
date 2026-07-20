# Latent-State / Thinking-vs-Speaking Experiments

Research into architectures that separate latent-state ("thinking")
computation from token generation ("speaking") — currently: a byte-level
front-end for frozen RWKV-7 g1g 2.9B that replaces the tokenizer, embedding,
and layer 0 with tiny learned models reading raw bytes.

**How work happens here** → [`AGENTS.md`](AGENTS.md). Theory first, then a
baseline, then a single-variable experiment, then a verdict against prior
art. No throwaway code.

## Setup
```bash
devenv up          # or: direnv allow   (provides Python + torch)
```
Verify the environment:
```bash
.devenv/state/venv/bin/python -c "import torch; print(torch.__version__)"
```

## Layout — organized by question, plus shared codebases

- **`threads/<slug>/`** — one directory per research question: its theory
  docs, its code, its `experiments/`. See [`threads/README.md`](threads/README.md).
- **`kit/`** — tier-0 primitives every thread imports: THE training loop,
  run-artifact writer, synthetic tasks, shared nn blocks.
- **`domains/`** — tier-1 project codebases: `byte/`, `rwkv/`, `g1g/`.
- **`src/`** — repo tooling (experiment lifecycle CLI, runner, status, tags).
- **`theories/`** — cross-thread governance: `proofs.md` (claim ledger),
  `status.md`, methods, archive.
- **`research/`** — prior art; start at `research/PRIOR_ART.md`.
- **`tests/`** — pytest suite. `reports/` — outside-facing write-ups.

## The experiment loop (10-second version)
```bash
python src/_status.py --all                          # 1. orient
# 2. check research/PRIOR_ART.md + research/*.md for previous attempts
python -m src.experiment theory my-question --thread my_thread   # 3. theory BEFORE code
python -m src.experiment new my_exp_001 --type byte_loop \
    --theory threads/my_thread/my-question.md \
    --variable n_loops --baseline byte_loop_001      # 4. scaffold (lands in the thread)
python -m src run my_exp_001 byte_loop               # 5. smoke, 6. run
python -m src.experiment record my_exp_001 best_loss=0.42   # 6b. metrics
python -m src.experiment compare byte_loop_001 my_exp_001   # 7. compare
python -m src.experiment verdict my_exp_001 supports --note "..."  # 8. verdict
python -m src.experiment index && python -m src.experiment audit
# code metadata lives in module docstrings ([meta] blocks):
python -m src.experiment harvest   # regenerates src/CODEMAP.md + src/meta.json
```

Every module declares what it tests inside its own docstring — theory, the
one variable it varies, its baseline, its verdict — and `harvest` scrapes
those comments into the registry. Comments are formal; nothing is maintained
twice.

## Prior work
Archived threads and dead ends live in `threads/*/archive/` and
`theories/archive/` — consult them before re-proposing an idea. Earlier
repo-wide states are in git history (see `experiments/ATTIC.md` for what the
2026-07-20 dedup removed and how to restore it).
