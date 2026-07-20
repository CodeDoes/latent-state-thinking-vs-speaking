# Experiment: {{EXP_ID}}

> **Theory:** {{THEORY}}
> **Hypothesis:** {{HYPOTHESIS}}
> **One variable:** {{VARIABLE}}
> **Baseline:** {{BASELINE}}
> **Created:** {{DATE}} | **Git:** {{GIT_HASH}} | **Verdict:** _pending_

<!--
Copy of the contract (see templates/THEORY.md): this file is the lab-book
page for one run. Fill the Comparison table — an experiment without a
comparison is not done.
-->

## Setup
| Key | Value |
|-----|-------|
| Type | <experiment type / model class> |
| Seed | 42 |
| Steps | <n> |
| Params | <model parameter count> |
| Task / data | <generator or dataset + cache path> |
| Time / device | <wall clock, cpu/gpu> |

Config: see `config.json` (authoritative; this table is a summary).

## Results
<What the metrics were. Paste the metric lines, not adjectives.>

## Comparison
| Method | Key metric | Params | Budget | Source |
|--------|-----------|--------|--------|--------|
| Baseline (`{{BASELINE}}`) |  |  |  |  |
| **This run (`{{EXP_ID}}`)** |  |  |  |  |
| Closest prior art (<name>) |  |  |  | <paper / research/*.md> |

Δ vs baseline: <number, in the metric the hypothesis named>
Δ vs prediction in theory doc: <number>

## Samples / evidence
<Qualitative outputs, plots, or failure examples. Link files in this dir.>

## Verdict
**<supports | refutes | inconclusive>** — <one paragraph: measured vs
predicted; what belief changed; the next single-variable follow-up, if any.>

## Reproduce
```bash
python -m src run {{EXP_ID}} <type>   # or the exact command used
# then: python -m src.experiment compare {{BASELINE}} {{EXP_ID}}
```
