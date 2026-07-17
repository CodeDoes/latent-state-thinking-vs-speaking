# smoke_test_methodology.status

Status of the small-scale proof methodology.

## Claims

- **S1** — *1-minute smoke tests on synth data with learnable patterns inform architecture decisions.*
  Status: **partially proven** — pattern-elimination used Jul 17 to reject "no-shape generators."

- **S2** — *Observable logs (loss/acc/samples) make run failure diagnoses possible.*
  Status: **proven** — most runs have logs.

- **S3** — *Deliberate overfitting (capacity probe) is a useful first goal.*
  Status: **partially used** — `byte_loop_001` and similar look for overfit, but don't formally classify.

- **S4** — *Simplest → Complex progression (BLT → BLT+RWKV → Transform+RWKV → RNN-only) is the right order.*
  Status: **proven** — matchable to `byte-state-byte.md` and `progressive-expansion.md`.

## Mechanism Gaps

- No formal "shape check" for synth generators (manual inspection)
- No auto-flag for vacuous-true eval (e.g. truncation ellipsing the answer mask)
- 1-minute budget not enforced by tooling — relies on self-discipline

## Follow-Ups

1. `smoke_001`: Make smoke test runner enforce ≤60s CPU budget
2. `smoke_002`: Build `SynthEvaluator` that tests generator for learnable-pattern signal
3. `smoke_003`: Fix known bugs (collation in `dendrite_rwkv_001`, max_len misconfig)
4. `smoke_004`: Per-experiment dashboard
