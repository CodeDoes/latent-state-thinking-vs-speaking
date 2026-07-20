# Theory: <short name>

> **Status:** proposed   <!-- proposed → testing → supported | refuted | superseded -->
> **Slug:** {{SLUG}}
> **Created:** {{DATE}}
> **Author:** <who/what wrote this>

<!--
ENGINEERING CONTRACT (AGENTS.md): fill every section below BEFORE writing
any code that tests this theory. A theory with an empty "Prior art" or
"Success criterion" section does not authorize code. Delete this comment
when the doc is complete.
-->

## One-line claim
<What is true, in one sentence, if this theory holds.>

## Problem
<What limitation, cost, or behavior motivates this. Cite the measurement
(benchmark, run, or proof entry) that shows the problem exists.>

## Prior art (external)
<Named previous attempts at the SAME problem, found before designing ours.
Minimum two. Use research/*.md and `python src/arxiv_query.py` to find them.
If a published method already does this, ours must be framed as a
reimplementation, simplification, or falsification of it — say which.>

| Method | Venue / arXiv | What they did | Why it doesn't close the question here |
|--------|---------------|---------------|------------------------------------------|
|        |               |               |                                          |
|        |               |               |                                          |

Closest prior art to copy design decisions from: <name + what we borrow>.

## Baseline (internal)
<The in-repo reference this is measured against: an experiment id, or a
plain/unmodified version of the new component at matched parameter count and
training budget. "No baseline" means the theory is not yet testable.>

- Baseline run: `experiments/<id>` — key metric: <value>
- If none exists: <plan to produce one, with config deltas vs the new method>

## Hypothesis (falsifiable)
<If X is changed to Y, then metric M will move from current ≥ Δ in direction D.
Must be phrased so a run can prove it WRONG. "It might help" is not a
hypothesis.>

## Prediction (measurable)
<Specific numbers before running, e.g. "val loss Δ ≥ 0.05 vs baseline at
200 steps", "cosine ≥ 0.9 vs real layer-0 output", "≤ 2× step time". Write
them down before the run — updating them afterwards is a new hypothesis.>

## Design: the one variable
<The single change vs the baseline. Everything else — data, seed, budget,
optimizer, params — listed as "held constant". If you cannot name the one
variable, this is a stew, not an experiment.>

**Variable:** <what changes>
**Held constant:** <what is fixed, incl. seed and parameter budget>

## Success criterion
<Pass/fail written as: metric, threshold, comparison ("vs baseline", "vs
prior art"). E.g. "refutes if |Δloss| < 0.02", "supports if acc ≥ baseline
− 0.5% with ≤ 20% of the params".>

## Risks / failure modes
<What could make the run lie to us (leakage, degenerate task, unmatched
budgets, wrong normalization). How each is controlled for.>

## Verdict
<filled after the experiment: **supports | refutes | inconclusive**, the
run id(s), the measured numbers vs the prediction, and what changed in our
beliefs. Link the proofs.md entry and the git tag.>

## References
<arXiv ids, URLs, and in-repo files cited above.>
