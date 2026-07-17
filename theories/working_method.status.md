# working_method.status

Status of the operating principles themselves — not the experiments they govern.

## Claims

- **W1** — *Observable, resumable, git-bound training protocol keeps experiments reproducible.*
  Status: **partially proven** — most experiments record commit hash; some record grad/step rate.

- **W2** — *Theory/status/infer triple keeps authorship traceable.*
  Status: **partially met** — folders agreed on Jul 17; older theories missing `.infer.md` or `.status.md`.

- **W3** — *Local (devenv.nix) iteration beats Kaggle for small experiments.*
  Status: **proven** by user's pivot Jul 14.

- **W4** — *Self-directed AI research loop is sustainable.*
  Status: **open** — relies on AI's ability to read the proof ledger and choose the next hypothesis.

- **W5** — *Brainstorming verbatim (incl. mess) is preserved as .md, not filtered.*
  Status: **met** (as of Jul 17).

## Mechanism Gaps

- Need automated commit-hash injection into every experiment's config
- Need to track which `.md` and `.infer.md` are *out-of-sync*
- Need to enforce: each new theory gets all three files (`.md`, `.infer.md`, `.status.md`)

## Follow-Ups

1. Audit all theories for `.md`-`.infer.md`-`.status.md` triple — add missing pairs
2. Verify `experiments/<id>/` are fully self-contained (no external checkpoints)
3. Add `--assert_hashes` flag to training scripts (warn if checkpoint's recorded hash ≠ current)
4. Build a `theories_lint.py` tool: flag any `theories/*.md` without matching `*.infer.md`
