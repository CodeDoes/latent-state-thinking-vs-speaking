# Attic

> **2026-07-20 reorganization note:** surviving experiment dirs moved into
> their threads (`threads/<slug>/experiments/<id>`). `experiments/` now only
> holds the generated `INDEX.md` and this file. Restoration commands below
> use the pre-move layout at git ref `d851b64`.

Rolled-up record of experiment dirs removed in the 2026-07-20 cleanup.
Nothing was lost: every removed dir is recoverable from git. To restore any
entry below:

```bash
git checkout d851b64 -- experiments/<name>
```

Removed because it was **exact duplication** (identical config to a kept run,
sometimes identical metrics too):

| Removed dir | Kept canonical | Evidence |
|---|---|---|
| `exp_better1` | `exp_baseline1` | byte-identical config, same metrics (acc 0.047) |
| `exp_diag3` | `exp_baseline1` | byte-identical config, same metrics |
| `exp_simple1` | `exp_baseline1` | byte-identical config, same metrics |
| `exp_diag2` | `exp_diag1` | byte-identical config; acc 0.0 vs kept run |
| `ent_0.0_check` | `ent_smoke_0.0` | entropy_weight=0.0 already covered by the kept run |
| `ent_smoke_0.0_long` | `ent_smoke_0.0` | config identical despite the name ("long") |
| `rwkv7_surgery_distil` | `rwkv7_surgery_001` | identical metrics; 001 has full artifacts |
| `patch_loop_002`, `patch_loop_003` | `patch_loop_001` | orphan metrics.jsonl, no config → unreproducible orphans |

Removed because it was **throwaway scratch** (test/trial names, trivial or
aborted runs — the exact antipattern AGENTS.md now bans):

`exp_test`, `exp_test2`, `exp_test3`, `exp_test4`, `prog_exp_test_check` (0.7 s
trial, 50 steps)

Removed because it was **dead** (config.json only; never produced a metric,
log, or sample — intent documented nowhere, recoverable from git if wanted):

`niiah_wc`, `niiah_wc_100`, `niiah_wc_den`, `niiah_wc_den2`, `niiah_wc_fast`,
`niiah_wc_final`, `niiah_wc_std`, `niiah_wc_test` (8 dirs of `_v2/_final` name
drift), `final_compare`, `rnn_patch_001`, `rnn_patch_002`, `cl_smoke_001`,
`ml_train_001`, `byte_iface_001`, `shunt_001`

Removed because it was **regenerable cache/checkpoint litter** (recreate by
re-running the owning script): `streaming_tokenizer` (3 pickles),
`loopy_tokenizer` (model.pt + model.step)

Reorganised (kept, better location):

- `self-directed-experiments/rwkv_state_passing_001` → `experiments/rwkv_state_passing_001`
- `self-directed-experiments/shared_state_unrolled_feedback_001` → `experiments/shared_state_unrolled_feedback_001`

Result: 95 → 68 experiment dirs; every remaining dir holds at least one
piece of evidence or an active role. The `exp_*` survivors (`exp_baseline1`,
`exp_diag1`, `exp_scale1`) should be renamed to theory-keyed ids as their
threads get proper docs.
