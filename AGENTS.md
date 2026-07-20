# AGENTS.md

Project: byte-level front-end for frozen RWKV-7 g1g 2.9B. Replace tokenizer,
embedding, and layer 0 with tiny learned models that read raw bytes directly,
plus the surrounding latent-state ("thinking vs speaking") architecture
research.

Core insight: the g1g byte-interface model (`byte_embed` + 32 RWKV blocks)
already accepts raw bytes. Our job is to compress/accelerate the front-end —
replacing expensive components with tiny learned alternatives that produce
equivalent hidden states — and to answer open questions about latent-state
computation with small, decisive experiments.

---

## The engineering mindset

You are a **research engineer**, act like one:

1. **Specifications before implementation.** The theory doc is the spec.
   If the requirement isn't written down, you don't know what you're
   building, and neither does anyone reviewing you.
2. **Measurements over opinions.** "It seems better" is not a result.
   A number, measured against a baseline, at a matched budget, is a result.
3. **Code is a liability; knowledge is the asset.** Every line you write is
   a line someone must maintain. The deliverable of an experiment is the
   verdict, not the script.
4. **Negative results are results.** A refuted hypothesis recorded honestly
   saves everyone from running it again. Deleting or hiding them is the
   only real failure.
5. **Everything is reversible or it doesn't happen.** Git hash in every
   config, tags around runs, self-contained experiment dirs.
6. **Small and proven beats big and hopeful.** Narrowest test that can
   decide the question. Scale only what's already proven.

---

## The Loop (mandatory, in order)

Run experiments through this lifecycle. Skipping steps is how the repo
roted last time; `python -m src.experiment audit` enforces the loop on
every commit.

### 1. Orient

```bash
python src/_status.py --all        # theories, experiments, code, git state
```
Read `theories/proofs.md` (what is proven/refuted), the theory doc for your
thread, and `experiments/INDEX.md` (what was already tried — do not re-run
the past).

### 2. Prior art first

**No method is new until you've named what came before.** Before designing:

- Check `research/PRIOR_ART.md` and the topic files in `research/`.
- Query arXiv: `python src/arxiv_query.py -n 10 "all:<query>"` (save
  `-o research/<topic>.json` when you learn something durable).

Your theory doc's *Prior art* section must name **at least two previous
attempts** at the same problem: what they did, why the question is still
open, and which one you copy design decisions from. If a published method
already does it, your theory is a *reimplementation, simplification, budget
cut, or falsification* of theirs — say which, and theirs becomes your
baseline.

### 3. Theory before code

**No theory doc, no code.** Write or extend the theory doc **inside its
thread** first:

```bash
python -m src.experiment theory <kebab-slug> --thread <thread-slug>
```

Fill `templates/THEORY.md` completely: falsifiable hypothesis, a prediction
with numbers, the ONE variable, the in-repo baseline, the success criterion
(metric + threshold, pass/fail). Writing predictions *after* seeing results
is fraud; git timestamps call it out.

### 4. Scaffold the experiment

```bash
python -m src.experiment new <exp_id> --type <type> \
    --theory threads/<thread>/<slug>.md \
    --variable <what-changes> --baseline <exp_id> \
    --hypothesis "<one-line falsifiable claim>"
```

`new` refuses to run without an existing theory doc and an existing baseline,
and puts the run in its thread (`threads/<thread>/experiments/<id>`).
Naming: `<theory_short>_<NNN>` (e.g. `dendrite_rwkv_001`); increment, never
reuse, never overwrite a run dir.

### 5. Smoke test first

≤60s CPU run on synthetic data with a learnable pattern
(`theories/method/smoke_test_methodology.md`). **If the smoke test doesn't
learn, fix the code or the theory — do not scale up and do not debug by
longer training.**

### 6. Run — observable or it didn't happen

- Log loss/metrics at fixed intervals (`LOG_EVERY`), to stdout *and*
  `experiments/<id>/train.log`; curves are deliverables, not debugging aids.
- Record final numbers: `python -m src.experiment record <id> best_loss=<v> acc=<v> steps_per_sec=<v>`
- Save a checkpoint (`*.safetensors` preferred), or write down why not.
- `config.json` carries seed, git hash, theory, variable, baseline — no
  untracked edits mid-run; change code → new run id.

### 7. Compare (the verdict needs a denominator)

An experiment is evidence only **relative to something**:

```bash
python -m src.experiment compare <baseline_id> <exp_id>
```

Fill the Comparison table in `RESULT.md`: baseline (matched params and
budget) and closest prior art (published numbers where available). Exactly
one config field should differ between baseline and experiment — the tool
warns when you stews up.

### 8. Verdict, ledger, commit

```bash
python -m src.experiment verdict <exp_id> supports|refutes|inconclusive --note "<measured vs predicted>"
python -m src.experiment index      # regenerate experiments/INDEX.md
```

Then: add the claim to `theories/proofs.md` (Supported or Refuted section,
with commit hash + key metrics), and commit with an honest message —
"X: refuted at Δloss +0.03", not "improved model". Update the theory doc's
*Verdict* section and status. Archive superseded theories to
`theories/archive/`.

---

## No throwaway code (hard rules)

The goal is a codebase where every file earns its keep:

- **Modules, not scripts.** All code imports without side effects and runs
  via `python -m threads.<slug>.<name>` (importable + `if __name__ ==
  "__main__"` guard). No entry-point scripts at repo root.
- **Config at the top** of each training module, plus `THEORY_PATH =
  "threads/<slug>/..."` stating what it tests and `EXPERIMENT_ID` what it
  fills. Training loops come from `kit.train`; run artifacts from
  `kit.runlog.Run` — no home-grown loops.
- **Promote up, never copy sideways.** Shared behavior belongs in `kit/`
  (universal) or `domains/` (project-wide). Before adding a block, check
  `kit/nn.py` and the domains package — and if you write the second copy of
  something, stop and promote instead.
- **No `python -c` beyond a 2-line check.** Scratch goes in `/tmp`, never
  the repo.
- **Parameterize, don't copy.** `train_x_v2_final2.py` is banned. Vary
  behavior through config/CLI flags. Fix bugs in the module, not in a fork
  of it.
- **Every file cites its reason to exist**: module docstring links theory
  doc + experiment id. Files that prove nothing get deleted or moved to
  `theories/archive/`, and the ledger explains why.
- **Tests guard behavior.** If a util was wrong once, there is a test in
  `tests/` now.
- If it's worth running, it's worth keeping reproducible. If it's not worth
  keeping, don't run it in the repo.

## Module metadata — comments are the registry

You already comment your code; make the comments *formalizable*. Every
`src/*.py` module carries a `[meta]` block at the end of its docstring:

```python
[meta]
status: active | triage-needed | superseded-by:src/x.py | archived
theory: theories/<dir>/<slug>.md      # the spec this code tests
experiment: <exp_id>[, ...]           # run(s) it produces
variable: <the ONE knob it varies>
baseline: <exp_id>                    # what it is compared against
verdict: supports | refutes | inconclusive | pending
[/meta]
```

- `python -m src.experiment harvest` scrapes all blocks into
  `src/meta.json` (machine) and `src/CODEMAP.md` (human). Both are
  **generated — never edited by hand**; the docstring is the source of
  truth, so metadata lives exactly once, inside the file it describes.
- `audit` fails on *invalid* meta: dangling theory paths or experiment ids,
  wrong enums. References are checked, so meta can't lie about what exists.
- A module without `[meta]` is **undocumented** and lands in the
  CODEMAP triage queue. New modules: start from `templates/MODULE.py` —
  claim status before you write the class.
- At Loop step 8, stamp the run's verdict into the module's `[meta]`
  `verdict:` line as well as the ledger, then re-harvest.

## Comparison culture (hard rules)

- "Relative to WHAT?" — always answerable. Default: same-budget plain
  variant of the new component (ablation), else the best prior-art method.
- **One variable per experiment.** If you can't name what changed, it was
  a stew, and the run's evidence is zero.
- **Claims need comparison rows.** proofs.md entries without a baseline
  number (or a clear "no baseline exists, produced one at …") are not
  accepted.
- **Budgets matched**: params, steps, data, optimizer held constant across
  baseline and treatment.
- Re-running the same config hoping for a different loss is noise, not
  science — change a variable or change the seed and say why.

## Repository map

The repo is organized **by question, not by artifact type** — plus tiered
shared codebases:

```
kit/                 TIER 0 — universal primitives (train loop, run artifacts,
                     synthetic tasks, shared nn blocks). Works in any project.
domains/             TIER 1 — project-shared codebases:
                     byte/ (vocab, tokenizers), rwkv/ (nano core, LoRA, vocab),
                     g1g/ (frozen 2.9B plumbing: auto_tokenizer, NF4, conversion)
threads/<slug>/      TIER 2 — ONE question each, self-contained:
                     <slug>.md theory docs + *.py code + experiments/ + archive/
theories/            cross-thread governance ONLY: proofs.md (claim ledger),
                     status.md, ultimate_thesis.md, method/, archive/
research/            prior-art summaries per topic + PRIOR_ART.md (the map)
src/                 the repo's TOOLING package: experiment lifecycle CLI,
                     runner (`python -m src`), _status, _tag, arxiv_query
templates/           THEORY.md & EXPERIMENT.md & MODULE.py — loop scaffolds
experiments/         generated INDEX.md + ATTIC.md (registry of all runs)
tests/               pytest suite
reports/             write-ups for outside readers (derived, never primary)
```

**Tier rule:** if two threads need the same code, it moves UP a tier with
its tests — it never gets copied sideways. Fix the primitive, not the
copies. A block lands in `kit/` only after proving itself in a thread.

## Active components

- **`domains/byte/hybrid_tokenizer.py`** — production tokenizer (TRIE
  boundaries, XOR-hash IDs). 96% story-level match vs real tokenizer.
- **`threads/g1g_frontend/train_byte_ae.py`** — byte auto-encoder; latent at
  trigger position is the compact representation for g1g.
- **`threads/g1g_frontend/train_loopy_timemix.py`** — predicts layer-0
  time-mix state from bytes; cos 0.77 vs real time-mix. Replaces
  byte_embed + layer 0.
- **`domains/g1g/auto_tokenizer.py`** — ByteG1GInference: quantized g1g
  byte-by-byte. NF4 cache at
  `~/Documents/models/rwkv7-g1g-byte-iface/nf4_cache/`.
- **`kit/`** — primitives every run uses: `train`, `runlog`, `tasks`, `nn`.
- **`src/__main__.py`** — registered model types:
  `python -m src list | run | resume | redo` (writes into the run's thread).
- **`src/experiment.py`** — the lifecycle CLI used throughout The Loop.

## Budgets & environment

- CPU smoke ≤ 60s; CPU pretraining ≤ ~10M params; real GPU runs go to Kaggle.
- `devenv.nix` is the single dependency definition. `requirements.txt` mirrors
  it for non-nix hosts only.
- Everything reproducible from `experiments/<id>/` alone.

## What not to do

- Write code before its theory doc has Prior art, Hypothesis, Prediction,
  and Success criterion filled in.
- Launch an experiment with no baseline named ("let's see what happens").
- Tune by vibes: change code → new experiment id, never rerun-and-hope.
- Inline scratch in the repo (`python -c`, `Untitled*.py`, `_v2` copies).
- Claim improvements without the comparison row (baseline + prior art).
- Sweep away failures. Record the refuted verdict and what it taught us.
- Debug by adding steps or params. Fix the data or the architecture.
