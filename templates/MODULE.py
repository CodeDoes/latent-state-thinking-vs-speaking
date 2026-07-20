#!/usr/bin/env python3
"""<One line: what this module does and why it exists.>

<2-4 sentences: the mechanism, and the claim it tests. If you cannot write
this, the module is not ready to exist.>

[meta]
status: triage-needed          # active | triage-needed | superseded-by:src/x.py | archived
theory: theories/<dir>/<slug>.md
experiment: <exp_id>           # run(s) this module produces, comma-list ok
variable: <the ONE knob this module varies>
baseline: <exp_id>             # same-budget run this is compared against
verdict: pending               # supported | refuted | inconclusive | pending
params: <param count once known>
[/meta]

Starter template for src/ modules. The [meta] block is machine-read by
`python -m src.experiment harvest` into src/CODEMAP.md; audit fails on
dangling refs. Keep it truthful — a reviewer reads this before the code.
"""

# ── config (everything tunable lives here, nowhere else) ─────────────────
THEORY_PATH = "theories/<dir>/<slug>.md"
EXPERIMENT_ID = "<exp_id>"

STEPS = 200
BATCH_SIZE = 16
LR = 1e-3
DIM = 64
SEED = 42
LOG_EVERY = 10
BUDGET_SECONDS = 60          # smoke-test cap (see AGENTS.md, Loop step 5)


def build_model():
    """Construct the thing under test."""
    raise NotImplementedError


def main():
    # smoke test first; only scale after the curve moves
    raise NotImplementedError


if __name__ == "__main__":
    main()
