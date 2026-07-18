# AGENTS.md

Project: byte-level RWKV with adaptive state and modular memory. Thesis in [`theories/ultimate.md`](theories/ultimate.md).

## Layout

```
src/                models, datasets, training, analysis (Python)
theories/           prose: design, hypotheses, results (one file per topic)
experiments/        per-run artifacts: config.json, train.log, metrics.json
                    (one dir per run; delete the dir to clear)
reports/            summaries written for outside readers
```

Each `experiments/<id>/config.json` records the git hash it was run on. Each `theories/<file>.md` is one self-contained document — no separate `.infer.md` or `.status.md`.

## How to work

1. **Read first**: `theories/ultimate.md` → `theories/ultimate_thesis.md` → `theories/proofs.md` → `python src/_status.py`. The status script is a one-screen dump of theories, experiments, code, and git state.

2. **Theory claims are tagged.** Stable IDs (`exp/<topic>/<NNN>`, `theo/<topic>/<CID>`) are git tags, not on-disk names. Directories get renamed; tags don't. See `src/_tag.py` for `exp`, `theo`, `link`, `list`, `show` subcommands.

3. **One variable per experiment.** If you can't name the single thing you changed, you didn't run an experiment. Match every other parameter.

4. **Loss must move.** If it's flat from step 1, the data has no learnable pattern (or the code is broken) — fix that *first*. No "needs more steps" reflex.

5. **Smoke test before scale.** Every new model runs a 60-second CPU pass with a pattern-bearing synthetic generator. See `theories/method/smoke_test_methodology.md`. If the smoke test doesn't learn, a longer run won't either.

## What not to do

- Multi-cause experiments.
- Scale-up debugging of RWKV without a named hypothesis.
- Re-opening retired threads in `theories/archive/`.
- Reading or rewriting older esses' milk — most theories are already wrong at the edges; we mark them superseded in the prose instead of deleting.
