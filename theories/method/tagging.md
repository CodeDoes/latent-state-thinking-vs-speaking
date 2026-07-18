# tagging

How experiments and theory claims get stable IDs.

## Why

The repo is small enough that "git grep" usually finds anything. The reason this is a theory anyway:

- A run on disk (`experiments/<id>/`) is identified by the *directory name* it lives in. Directory names collide and get renamed retroactively (e.g. `adapt_ent_smoke_test` became `adaptive_loop_001` once we figured out what it was).
- A theory claim in a markdown file ("**G1** — branches over a frozen trunk …") doesn't have a stable reference target either. Saying "see the G1 claim" is a soft pointer.
- We want a *canonical* identifier for both, so:
  - a run can be referred to before, during, after it ran.
  - a claim stays traceable even when we delete / supersede / rename the prose.
  - a git host gives us the guarantee that "tag `exp/adaptive_loop_001`" means something even if the on-disk dir is renamed.

## The scheme

Two tag namespaces.

### Experiment tags

```
exp/<topic>/<NNN>
```

- `topic` is a slug for the theory the run belongs to (e.g. `adaptive_compute`, `byte_state_byte`, `dendrite_growth`).
- `NNN` is the next available 3-digit sequence number under that topic in `git tag`.
- The tag's body points to the commit in which the experiment was run.

Examples: `exp/adaptive_compute/001`, `exp/byte_state_byte/004`, `exp/dendrite_growth/001`.

### Theory claim tags

```
theo/<topic>/<CID>
```

- `topic` matches the corresponding experiment topic when one exists.
- `CID` is a **claim id** chosen by the author — usually matching whatever the prose already uses (`B5`, `G1a`, `H2`). Lowercase or uppercase; the namespace treats them as distinct.

Examples: `theo/byte_state_byte/B5`, `theo/dendrite_growth/G1a`, `theo/realtime_ai/R1`.

## Minting tags

The utility is `src/_tag.py`. It writes tags into git only when you pass `--apply`; otherwise it prints the `git tag` command for you to run.

```bash
# 1. Mint an experiment tag from a current run
python ./src/_tag.py exp adaptive_loop_001 \
    --topic adaptive_compute \
    --note "B5 proof: 228K params, loss 5.74 -> 0.47" \
    --apply                # creates the env tag if also committed
```

```bash
# 2. Mint a claim tag for a theory
python ./src/_tag.py theo dendrite_growth G1a --apply
# (no claim_id → next H<N> is suggested)
```

```bash
# 3. List tags
python ./src/_tag.py list              # all
python ./src/_tag.py list --exp        # only exp/
python ./src/_tag.py list --theo       # only theo/

# 4. Inspect
python ./src/_tag.py show theo/byte_state_byte/B5

# 5. Link an experiment to a claim it supports
python ./src/_tag.py link exp/adaptive_compute/001 theo/byte_state_byte/B5
# writes experiments/.../relationships.json
```

## What gets written where

| Action | File change |
|---|---|
| Mints `exp/...` | `experiments/<id>/config.json` gets `tag`, `tag_topic`, `tag_seq`, `tag_note` fields |
| Mints `theo/...` | git only, and an entry appended into `theories/proofs.md` (the ledger) |
| `link` | `experiments/<id>/relationships.json` gets `{"supports": [<theo-tag>, …]}` |

## Why `lightweight` tags

`git tag <name> <commit>` creates a *lightweight* tag (just a pointer to a commit). No message, no signature, no special commit. That's what we want:

- Cheap — no friction to mint.
- Visible to plain `git log --decorate` and `git describe`.
- Scriptable — `python src/_tag.py list` walks them.
- Diffable — `git diff <tag1>..<tag2>` shows what changed.
- Survives mail/PR — push the tag and someone else can `python src/_tag.py show <tag>` on a fresh clone.

When we want to *prove* a claim (record its evidence permanently), `git tag -a` is one flag away. For now, lightweight + `relationships.json` is enough.

## Failure modes

- Tag clashes. `theo/dendrite_growth/G1a` already exists → pick `G1b` or refuse. The CLI refuses, prints the conflict.
- Topic drift. Topics can collide if you rename theories later (`byte_state_byte` → `bsb`). The fix is: pick one canonical topic per theory and stick to it.
- Tag rename mid-flight. We don't support it; if a topic name is wrong, retire the old tag and mint new ones. Old tag stays pointing at the old commit.

## Names that match the on-disk layout

`topic` should match the slug of the theory file's name stripped of suffix and lowercased. So:

| theory file | topic |
|---|---|
| `adaptive/adaptive-exit-entropy.md` | `adaptive_exit_entropy` |
| `memory/dendrite_growth.md` | `dendrite_growth` |
| `architecture/byte-state-byte.md` | `byte_state_byte` |

This isn't enforced. The CLI normalises to snake_case. If you pick a topic that doesn't match, lint will see it later. Don't try too hard.

## When to mint an experiment tag

Mint an experiment tag as soon as the experiment `config.json` exists. Then the experiment is "named" with the tag, even before its `train.log` is written. Mint *again* if you re-train — the tag points to the second commit. The first tag stays. They are different runs even on the same machine, and both should be findable.

## When to close a theory claim tag

Once an experiment supports or refutes the claim:

```bash
python ./src/_tag.py link exp/<topic>/<NN> theo/<topic>/<claim>
```

This is the formal "the result of run NN is recorded against claim X" event. The proof ledger `proofs.md` is updated automatically as a side effect of `theo` mints; but `link` does NOT update proofs.md yet — that is your job, after you read the result. Reading the link and reading the run is what produces a one-line proof entry.

## Status

- Design phase.
- `src/_tag.py` exists; tested end-to-end with one example run (later reverted).
- `_status.py` shows experiment tags and "supports" lines for each tagged experiment.
- Need to mint tags for all proofs already in `proofs.md`.

---

# Running an experiment with bracketed snapshots (`_tag.py run`)

The `run` subcommand in `src/_tag.py` automates the bookkeeping that, before,
was done by hand: snapshot the codebase *before* the run, then commit and
snapshot *after*. Two lightweight tags (`-pre`, `-post`) replace what would
otherwise be a worktree.

## Why a worktree isn't used

The original idea was to make a git worktree per experiment so the run
"lives" in a separate copy of the repo. Reasons not to:

- The repo is small (~370 tracked files, ~2 MB on disk). A worktree copy
  is cheap, but the *file count grows with codebase size* and so does the
  risk of an experiment polluting or partially committing state.
- A worktree is a separate working directory. That means a separate Python
  venv / `PYTHONPATH` / imports, separate shell boot, separate relative
  paths (`./src/foo` vs `<worktree>/src/foo`). All friction for no
  provenance benefit.
- The whole reason a worktree helps is to make the *code as it looked at
  run time* findable later. A **git tag** pointing at a commit does the
  same thing in ~200 bytes.
- The repo already ignores `.devenv/`, `__pycache__/`, and `*.pt` via
  `.gitignore`. A worktree doesn't snapshot those regardless.

So: the snapshot is two tags plus one commit. Nothing else.

## What `run` does

```
python ./src/_tag.py run \
    --id byte_loop_002 \
    --topic byte_state_byte \
    --note "smoke: encoder-decoder with 1L each, 117K params" \
    --config /path/to/my_params.json \
    --command python ./src/train_byte_rwkv.py
```

Steps, in order:

1. Verify the working tree is clean (so the -pre snapshot is a real
   recorded state, not "state plus my in-progress edits").
2. Compute the next sequence number (`NNN`) under `exp/<topic>/`.
3. Write `experiments/<id>/config.json` with `tag`, `tag_topic`,
   `tag_seq`, `tag_note` fields *and* any fields merged from
   `--config`.
4. `git tag exp/<topic>/<NNN>-pre HEAD` — the **pre** snapshot of the
   code as it looked right before the run.
5. Run the command via subprocess; capture stdout+stderr into
   `experiments/<id>/train.log`.
6. `git add experiments/<id>/` (plus, with `--include-edits`, any
   tracked-file edits the run produced).
7. `git commit` and `git tag exp/<topic>/<NNN>-post <commit>` —
   the **post** snapshot with the run's outputs in tree.

After a `run` you have exactly two git tags. Their diff tells you
exactly what the run produced. The previous "-pre" commit is unchanged.

## Options

| flag | meaning |
|---|---|
| `--id NAME` | experiment directory under `experiments/` (required) |
| `--topic SLUG` | topic slug for the tag namespace (required) |
| `--config FILE` | optional JSON file with extra fields merged into config.json |
| `--note TEXT` | recorded into config.json and both tag messages |
| `--command A B C ...` | passthrough to subprocess; everything that follows is the command |
| `--include-edits` | commit tracked-file edits in the -post commit too (default: only `experiments/<id>/`) |
| `--dry-run` | print the plan, do nothing |
| `--pre-only` | mint the -pre tag and exit (e.g. you want a "starting state" record before a long interactive experimentation phase) |
| `--post-only` | skip the -pre tag and the run; commit + tag an already-finished experiment |

In `--post-only` mode the -pre tag is *not* minted. We rely on it being
explicit that the experiment pre-state was whatever HEAD said at the
time the experiment *finished*. Use this when you've already run code
without the wrapper and want to bring the bookkeeping in line.

## Verifying a snapshot pair

```
git diff exp/byte_state_byte/002-pre exp/byte_state_byte/002-post
git show    exp/byte_state_byte/002-post --stat
```

That diff is the canonical answer to "what changed from start to end of
that experiment." Reproducing it on another machine: checkout the -pre
commit, run the captured command from `train.log`, and you should land
on the -post commit.

## Failure modes

- **Dirty tree**. `run` refuses to start. Commit or stash first. This is
  not just bookkeeping: an unsigned -pre tag of a dirty tree is not a
  real snapshot — your `-` and `M` lines won't be in HEAD.
- **Command exits non-zero**. The -post snapshot still happens. We want
  failed runs preserved, not erased. The exit code prints to the
  runner's stdout; re-read `train.log` for the captured version.
- **No --command supplied**. We skip the run and go straight to
  post-snapshot. Useful for "the experiment dir already exists with
  outputs from elsewhere; please lock it in git history."
- **Tag clash**. Two runs of the same topic with the same seq number
  collide. `_tag.py` advances the seq from existing tags, so two
  parallel runners can each mint `NNN+1` and then the second commit
  produces a different commit with the same tag name. We don't
  auto-force overwrites; if you see a duplicate, re-tag with
  `_tag.py exp … --apply --target <commit>`.
