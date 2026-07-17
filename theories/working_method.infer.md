# Working Method — Inferred Interpretation

> **Source**: `theories/working_method.md` (verbatim)  
> **Date**: 2025-07-17

---

## The Operating Principles (Extracted from Your Words)

| Principle | Verbatim | Where it applies |
|---|---|---|
| **Observable training** | *"i want the training to be observable"* | Every training script must log loss/accuracy/sps to stdout + file |
| **Resumable** | *"i prefer to have resumable training"* | `--resume` flag, save checkpoints |
| **Continually improving** | *"i prefer to constantly improve the trained model"* | No "frozen forever" experiments; save model snapshots |
| **Git-bound** | *"experiments should attach their code to a git commit hash"* | Every `experiments/<id>/` records `git rev-parse HEAD` |
| **Easy to clear** | *"trained models should be easy to clear"* | `Delete experiments/<exp_id>/` to fully clear; no state leaks |
| **Single-variable ablations** | *"isolate things that can be issolated. and those that can't try to introduce them sequentially"* | One variable at a time, matched params |
| **Self-directed research** | *"continue with your self-directed research"* | AI proposes next experiment from the proof ledger |
| **Error recovery** | *"either fix your error. or if the experiment proves the theory is invalid ... create a different theory that might unlock a different advantage"* | Failed experiment ≠ dead end; pivot if data demands |
| **Theory/paper lifecycle** | *"theories can get simplified and archived in git history instead"* | Don't keep stale theories; archive old versions, snapshot "X proves Y" |
| **Source verbiage** | *"theories/ is for my crazy verbatim utterings! theories/*.infer is what you made sense of"* | `.md` = your words verbatim; `.infer.md` = my interpretation |
| **Trash included** | *"you should extract even the trash ! its not like it matters. im not talking to a person. im brain storming"* | Brainstorming content lives in `.md` even if messy |
| **Progressive phased training** | *"a small pure BLT model -> working. a small BLT encoder and decoder with a small RWKV -> working. etc... in that order. would be better"* | Incremental model complexity; prove each stage before next |

---

## The Folder Contract (Your Words, My Mapping)

| Folder | Your Definition |
|---|---|
| `theories/` | *"my crazy verbatim utterings!"* — your raw thinking |
| `theories/*.infer` | *"what you made sense of"* — my interpretation of latent assumptions |
| `theories/*.state`* | *"how the experiments are going"* — proof ledger |
| `experiments/` | *"experiments for all the underlying assumptions and the hypothesis and claims in theories/"* |
| `reports/` | *"outside consumption on what has been discovered so far"* — polished summaries |
| `src/` | (implied) code: models, datasets, training loops |
| `devenv.nix` | (implied) the only dependency manager |

---

## Training & Experiment Lifecycle (From Your Words)

1. **Plan**: theory in `theories/*.md`, hypotheses numbered (H1, H2, etc.)
2. **Status**: `theories/*.status.md` tracks claim status (open / running / proven / invalidated)
3. **Code**: `src/` holds model + data + training scripts (not in `experiments/`)
4. **Run**: `experiments/<id>/` holds logs + checkpoints for one run
5. **Commit**: every experiment records `git rev-parse HEAD`
6. **Clear**: delete `experiments/<id>/` to fully clear — no global state

---

## What Changed Over Time

| Phase | What you wanted | Notes |
|---|---|---|
| **Phase 1** (Jul 10–13) | Use **Kaggle** for experiments; pit API setup | Here's the concurrency: i want you to use a kaggle notebook input. save the pytorch notebook as a space or something" |
| **Phase 2** (Jul 13–14) | Clean up repo; **local only** with devenv | *"im not training massive models"* |
| **Phase 3** (Jul 14–15) | **Observable, resumable** training; small-scale proof before scale | *"i want the training to be observable"* |
| **Phase 4** (Jul 15–17) | **Self-directed research**: AI proposes next experiment | *"continue with your self-directed research"* |
| **Phase 5** (Jul 17) | **Brainstorming mode**: capture verbatim, not polish | *"extract even the trash"* |

---

## Latent Assumptions (Infer Notes)

- You assume **observable ≠ noisy**: logs should be *readable*, not exhaustive.
- "Easy to clear" implies no hidden checkpoints outside `experiments/<id>/`.
- "Git-bound" implies *also* "git-restorable": checkpoint must be loadable from any commit.
- "Stale theories" means: when a theory is proven wrong, its `.md` and `.infer.md` get archived but not deleted.
- The Kaggle → Local pivot (Jul 14) suggests you're optimizing for **CPU rapid iteration** over GPU scale. RWKV-nano fits this.
- The byte-level-RWKV architecture (Jul 16) was a "small-step concrete goal": start with plain BLT, gradually introduce RWKV core.
- You treat dendrite theories as a *final destination*, not an immediate one ("only after we have a solid foundation do i want to do a training run").
- The trash-included signal means *raw think-tap content* is the substrate. My job is to **make sense of it**, not filter it.

---

## Open Follow-Ups (For Working Method)

1. Verify `git rev-parse HEAD` is consistently logged in every experiment's metadata
2. Check `experiments/<id>/` clean-deletability (no scattered checkpoints in `src/`)
3. Make `theories/*.status.md` updates *automated* from experiment runs (not just manual)
4. Ensure each theory has both a `.md` and `.infer.md` paired; refuse to let them diverge silently
