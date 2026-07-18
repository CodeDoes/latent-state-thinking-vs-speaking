# AGENTS.md

Scaffold for architectures separating **thinking** (latent state) from **speaking** (token generation). See [`theories/ultimate.md`](theories/ultimate.md) for philosophy, [`theories/proofs.md`](theories/proofs.md) for what is already proven.

## Rules
- Prove one thing at a time — single-variable ablations, matched params.
- No claimed capability without a named mechanism and a controlled disablement that breaks it.
- If an experiment doesn't isolate one property, it's not an experiment.

## Layout

```
src/                 # models, datasets, training, analysis scripts
theories/            # prose, hypotheses, design rationales (grouped by theme)
experiments/         # one dir per run; tracked in git (configs + logs)
reports/             # polished summaries written for outside readers
devenv.nix           # sole dependency manager
```

### `experiments/<id>/` contains
- `config.json` (git hash, tag, hyperparams, seed)
- `train.log` (stdout/stderr from the run)
- `metrics.json` or `samples.json` (final numbers + per-checkpoint samples)
- `checkpoint.pt` if saved (gitignored — large)
- `relationships.json` (which theory claims this run supports)

Delete the whole dir to fully clear a run — no state leaks.

### `theories/` is organised by theme

```
theories/
├── architecture/       — main scaffold threads
├── adaptive/           — adaptive compute, ablations on interface
├── memory/             — state-based memory variants
├── spatial/            — 2D / pointer / diffusion variants
├── core/               — RWKV layer mechanics
├── method/             — operating principles, tagging, proof rules
├── application/        — target use-cases
├── analysis/           — research reading notes
├── archive/            — retired drafts
├── proofs.md           — proven claims (one line each, newest first)
├── status.md           — index of theories
├── ultimate.md         — single-line thesis
└── ultimate_thesis.md  — multi-paragraph overview
```

Each theory file is a single self-contained document. No `.infer.md`/`.status.md` triple.

## Tagging

Stable IDs are minted as `git tag`s via [`src/_tag.py`](src/_tag.py). The tag name *is* the canonical reference; renaming a directory doesn't break the reference.

```
exp/<topic>/<NNN>     — experiment run (e.g. exp/byte_state_byte/004)
theo/<topic>/<CID>    — theory claim   (e.g. theo/dendrite_growth/G1a)
```

`<topic>` is a snake_case slug matched to the theory file's stem. `<NNN>` is the next available 3-digit sequence under that topic. `<CID>` is the claim id from the prose (`B5`, `G1a`, `H2`, …).

Tag commands:
```bash
python src/_tag.py exp    <exp_id>           --topic <topic> --note <text> --apply
python src/_tag.py theo   <topic> [claim_id] --note <text> --apply
python src/_tag.py link   exp/.../NNN theo/.../CID     # records supports relationship
python src/_tag.py list [--exp|--theo]
python src/_tag.py show   <tag>
```

Without `--apply`, the scripts print the `git tag` command instead of running it (so you can review the message before sealing).

`theo/` mints also append a proof entry to [`theories/proofs.md`](theories/proofs.md). `link` commands write `experiments/<id>/relationships.json`. `python src/_status.py` shows tags and `supports` lines.

## Proof rules

- A claim in a `<topic>.md` is **open** until an experiment supports or refutes it. Both end states live in the proof ledger, both pointing to the same git-tagged claim.
- An experiment is *done* when `config.json` exists with git hash, `train.log` shows a curve (not just one line), and either a `metrics.json` or a `samples.json` is present.
- A claim is **proven** when an experiment with matching conditions supports it. A claim is **falsified** when an experiment contradicts the named mechanism.

## Operating principles

See [`theories/method/working_method.md`](theories/method/working_method.md). In one paragraph:

Observable by default (loss/acc/sps logged), resumable (`--resume`), git-bound (`git_hash` recorded in `config.json`), easy to clear (delete the dir, no global state), single-variable ablations, self-directed research. Burn-in failure modes: a flat loss curve is *not* a "needs more steps" — first fix the synthetic data so it has a learnable pattern, then re-run.

## Smoke-test rule

See [`theories/method/smoke_test_methodology.md`](theories/method/smoke_test_methodology.md). Every new model gets a 60-second CPU run with a learnable-pattern synthetic generator before any longer experiment is committed.

## Reader path

1. [`theories/ultimate.md`](theories/ultimate.md) — the thesis.
2. [`theories/ultimate_thesis.md`](theories/ultimate_thesis.md) — overview of threads.
3. Any individual theory [`theories/<theme>/<topic>.md`](theories/) you care about.
4. [`theories/proofs.md`](theories/proofs.md) for state of evidence.
5. [`python src/_status.py`](src/_status.py) for a one-screen dump.
