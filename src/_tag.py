#!/usr/bin/env python3
"""Tag and link experiments and theory claims.

Each experiment run gets a stable ID (e.g. `exp/byte_state_byte/004`). Each
theory claim gets a stable ID (e.g. `theo/dendrite_growth/G1a`). Both can
be registered as `git tag`s so they survive without external state.

Usage:
    # 1. Mint a new tag for the *current* experiment (writes the config first)
    python ./src/_tag.py exp <exp_id> [--link <claim>] [--note <text>]
        → computes the next sequence number based on existing tags
        → updates experiments/<exp_id>/config.json with the tag
        → prints "git tag <name>" command you should run

    # 2. Mint a tag for a *theory claim*
    python ./src/_tag.py theo <theory> <claim_id> [--note <text>]
        → checks the claim hasn't been tagged before
        → prints "git tag <name>" command you should run

    # 3. List tags
    python ./src/_tag.py list [--exp|--theo|all]
        → walks git tags and groups by prefix

    # 4. Find an experiment/theory by tag, get a summary
    python ./src/_tag.py show <tag>
        → prints the experiment's last commits/note and links

    # 5. Link experiments and claims
    # When you finish an experiment, link the result to the claim:
    python ./src/_tag.py link exp/byte_state_byte/004 theo/dendrite_growth/G1a
        → records in experiments/<exp_id>/relationships.json that this
          experiment supports the claim

    # 6. Run an experiment with bracketed -pre / -post git-tag snapshots
    python ./src/_tag.py run --id <exp_id> --topic <topic> \
            [--note <text>] [--config <cfg.json>] \
            --command <command + args...>
        → ensures a clean working tree
        → mints exp/<topic>/<NNN>-pre at HEAD (lightweight tag)
        → writes experiment config (and merges --config into it)
        → runs the command, capturing stdout+stderr into train.log
        → commits experiments/<exp_id>/ (and tracked edits if
          --include-edits) and mints exp/<topic>/<NNN>-post at the
          new commit
    See theories/method/tagging.md for the full rationale.

A tag like `exp/byte_state_byte/004` is human-meaningful. Registered
with git as a *lightweight tag*:
    $ git tag exp/byte_state_byte/004 <commit>

The tag name is the canonical ID. Body lives in the file the tag points
to (commit blob). So if you push to a separate box, the tag still
identifies the claim.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# ── Naming ────────────────────────────────────────────────────────────────


TAG_PREFIX_EXP = "exp/"
TAG_PREFIX_THEO = "theo/"

EXP_RE = re.compile(r"^exp/([a-z0-9_]+)/(\d+)$")
THEO_RE = re.compile(r"^theo/([a-z0-9_]+)/([A-Za-z0-9_]+)$")


def sanitize_topic(name):
    """Convert 'byte-state-byte' → 'byte_state_byte'."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")


def parse_tag(tag):
    if tag.startswith(TAG_PREFIX_EXP):
        m = EXP_RE.match(tag)
        if m:
            return ("exp", m.group(1), m.group(2), None)
    if tag.startswith(TAG_PREFIX_THEO):
        m = THEO_RE.match(tag)
        if m:
            return ("theo", m.group(1), m.group(2), None)
    return None


def slug_for(exp_id):
    """'byte-state-byte.md' → 'byte_state_byte' or 'architecture/byte-state-byte.md' → 'byte_state_byte'."""
    name = Path(exp_id).stem
    return sanitize_topic(name)


def normalize_topic(topic):
    return re.sub(r"[^a-z0-9_]", "_", topic.lower()).strip("_")


# ── Experimental Numbering ───────────────────────────────────────────────


def next_exp_number(repo_root, topic):
    """Find the highest existing exp/<topic>/NNN tag, return next NNN."""
    out = subprocess.run(
        ["git", "tag", "--sort=v:refname"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    n = 0
    pat = re.compile(rf"^{re.escape(TAG_PREFIX_EXP)}{re.escape(topic)}/(\d+)$")
    for line in out.stdout.splitlines():
        m = pat.match(line)
        if m:
            n = max(n, int(m.group(1)))
    return n + 1


def next_theo_claim_id(repo_root, topic, taken=None):
    """Find used claim IDs for this theory, return next suggestion.

    The claim IDs are usually `H1`, `G1`, `B5`, etc. - non-numeric.
    Suggest `H<count>` when no claim IDs exist.
    """
    out = subprocess.run(
        ["git", "tag", "--sort=v:refname"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    nums = set()
    pat = re.compile(rf"^{re.escape(TAG_PREFIX_THEO)}{re.escape(topic)}/([A-Za-z0-9_]+)$")
    for line in out.stdout.splitlines():
        m = pat.match(line)
        if m:
            cid = m.group(1)
            # Track numeric suffixes as H1, H2, ...
            cm = re.match(r"^[Hh](\d+)$", cid)
            if cm:
                nums.add(int(cm.group(1)))
    if taken:
        nums.update(int(t) for t in taken)
    if nums:
        return f"H{max(nums) + 1}"
    return "H1"


# ── Project Layout ────────────────────────────────────────────────────────


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"


def find_experiment(exp_id):
    """Locate the experiment directory; return Path or None."""
    candidates = [EXPERIMENTS_DIR / exp_id]
    # Allow full or partial
    if not candidates[0].exists():
        # Try matching prefix
        for sub in EXPERIMENTS_DIR.iterdir():
            if sub.is_dir() and sub.name == exp_id:
                candidates[0] = sub
                break
    return candidates[0] if candidates[0].exists() else None


def find_config(exp_id):
    d = find_experiment(exp_id)
    if not d:
        return None, None
    cfg = d / "config.json"
    if not cfg.exists():
        return d, None
    return d, json.loads(cfg.read_text())


def update_config(exp_id, **updates):
    d, cfg = find_config(exp_id)
    if not d:
        sys.exit(f"experiment {exp_id} not found")
    if not cfg:
        cfg = {}
    cfg.update(updates)
    cfg_path = d / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"updated {cfg_path}")


def git_tag(name, target="HEAD", msg=None, force=False):
    cmd = ["git", "tag"]
    if force:
        cmd.append("-f")
    if msg:
        cmd += ["-m", msg]
    cmd += [name, target]
    subprocess.run(cmd, cwd=ROOT, check=True)


def git_show_tag(name):
    out = subprocess.run(
        ["git", "show", name, "--no-patch", "--format=%(objecttype) %(refname:short) %(taggerdate:short) %(subject)"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else None


def list_git_tags(prefix=None):
    out = subprocess.run(
        ["git", "tag", "--sort=v:refname"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tags = out.stdout.splitlines()
    if prefix:
        tags = [t for t in tags if t.startswith(prefix)]
    return tags


# ── Commands ─────────────────────────────────────────────────────────────


def cmd_exp(args):
    """Mint a tag for an experiment and stamp it into config.json."""
    raw_id = args.exp_id
    topic = normalize_topic(args.topic)
    n = next_exp_number(ROOT, topic)
    tag = f"{TAG_PREFIX_EXP}{topic}/{n:03d}"
    note = args.note or ""
    update_config(raw_id, tag=tag, tag_topic=topic, tag_seq=n, tag_note=note)

    if args.apply:
        git_tag(tag, target=args.target, msg=note or f"exp: {tag}")
        print(f"tagged {tag} → {args.target}")
    else:
        print()
        print(f"# Run this to register the tag (lightweight, points to {args.target}):")
        print(f"git tag {tag} {args.target}")
        if note:
            print(f"# Note: {note}")


def cmd_theo(args):
    """Mint a tag for a theory claim; check it isn't already taken."""
    topic = normalize_topic(args.topic)
    claim = args.claim_id[0].upper() + args.claim_id[1:] if args.claim_id else ""
    if not claim:
        claim = next_theo_claim_id(ROOT, topic)

    tag = f"{TAG_PREFIX_THEO}{topic}/{claim}"

    existing = list_git_tags()
    if tag in existing:
        sys.exit(f"tag {tag} already exists; pick another claim id")

    note = args.note or ""
    if args.apply:
        target = getattr(args, "target", "HEAD")
        git_tag(tag, target=target, msg=note or f"theo: {tag}")
        print(f"tagged {tag} → {target}")
        # Also append to proofs.md
        proofs_path = ROOT / "theories" / "proofs.md"
        if proofs_path.exists():
            proofs = proofs_path.read_text()
            entry = f"- `{tag}` ({target[:7]}): {note or 'claim opened'}\n"
            if tag not in proofs:
                proofs_path.write_text(proofs + entry)
                print(f"appended entry to {proofs_path}")
    else:
        print()
        print(f"# Theory claim tag candidate:")
        print(f"git tag '{tag}' HEAD")
        if note:
            print(f"# Note: {note}")
        print()
        print(f"# To remove if needed: git tag -d '{tag}'")


def cmd_list(args):
    tags = list_git_tags()
    if not tags:
        print("(no git tags)")
        return

    if args.exp:
        prefixes = (TAG_PREFIX_EXP,)
    elif args.theo:
        prefixes = (TAG_PREFIX_THEO,)
    else:
        prefixes = (TAG_PREFIX_EXP, TAG_PREFIX_THEO)

    shown = [t for t in tags if any(t.startswith(p) for p in prefixes)]
    if not shown:
        print(f"(no tags matching {prefixes})")
        return

    print(f"Found {len(shown)} tag(s):")
    for t in shown:
        info = git_show_tag(t)
        print(f"  {t}")
        if info:
            print(f"    {info}")


def cmd_show(args):
    tag = args.tag
    info = parse_tag(tag)
    if not info:
        sys.exit(f"{tag} is not a recognised exp/ or theo/ tag")

    kind, topic, seq, claim = info
    print(f"Tag      {tag}")
    print(f"Kind     {kind}")
    print(f"Topic    {topic}")
    print(f"Seq/ID   {seq}")

    info_str = git_show_tag(tag)
    if info_str:
        print(f"Git:     {info_str}")

    if kind == "exp":
        # Locate the experiment directory via tag_topic config field
        # We're reverse-mapping: try each experiments/* dir looking for matching tag in config
        for d in EXPERIMENTS_DIR.iterdir():
            cfg = d / "config.json"
            if not cfg.exists():
                continue
            try:
                data = json.loads(cfg.read_text())
            except Exception:
                continue
            if data.get("tag") == tag:
                print()
                print(f"Experiment: {d.name}  ({d})")
                print(f"  config.json path: {cfg}")
                # Print vcs commit
                gh = data.get("git_hash")
                if gh:
                    print(f"  commit:   {gh}")
                # Print relationships
                rel = d / "relationships.json"
                if rel.exists():
                    rel_data = json.loads(rel.read_text())
                    if "supports" in rel_data:
                        for claim_id in rel_data["supports"]:
                            print(f"  supports: {claim_id}")


def cmd_link(args):
    """Record that an experiment supports theory claims."""
    exp_tag = args.exp_tag
    theo_tag = args.theo_tag

    # Find experiment by tag
    info = parse_tag(exp_tag)
    if not info or info[0] != "exp":
        sys.exit(f"{exp_tag} is not an exp/ tag")

    exp_dir = None
    for d in EXPERIMENTS_DIR.iterdir():
        cfg = d / "config.json"
        if not cfg.exists():
            continue
        try:
            data = json.loads(cfg.read_text())
        except Exception:
            continue
        if data.get("tag") == exp_tag:
            exp_dir = d
            break
    if not exp_dir:
        sys.exit(f"no experiment has tag {exp_tag}")

    rel_path = exp_dir / "relationships.json"
    if rel_path.exists():
        rel_data = json.loads(rel_path.read_text())
    else:
        rel_data = {}

    supports = rel_data.get("supports", [])
    if theo_tag not in supports:
        supports.append(theo_tag)
    rel_data["supports"] = supports

    # Also reverse-link: experiment indexed by claim
    rel_data["updated_at"] = _now()

    rel_path.write_text(json.dumps(rel_data, indent=2))
    print(f"linked {exp_tag} → {theo_tag} in {rel_path}")

    # Suggest a git tag to mark the link
    print()
    print("# Run if you want to lock this link in git history (optional):")
    link_tag = f"link/{exp_tag}/{theo_tag}".replace("/", "__").replace("__", "_")
    print(f"# git tag '{link_tag}' HEAD")


# ── run: pre/post snapshots around a command ─────────────────────────────


def _require_clean_tree(repo_root):
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if lines:
        sys.exit(
            "working tree is not clean. Commit or stash first:\n  "
            + "\n  ".join(lines[:10])
            + ("\n  ..." if len(lines) > 10 else "")
        )


def _current_commit(repo_root):
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _stage_experiment(repo_root, exp_id, include_edits=False):
    """Stage `experiments/<exp_id>/` plus optional tracked edits."""
    subprocess.run(
        ["git", "add", "--force", f"experiments/{exp_id}"],
        cwd=repo_root,
        check=True,
    )
    if include_edits:
        # also pick up edits to any tracked file the run produced
        subprocess.run(
            ["git", "add", "-u"],
            cwd=repo_root,
            check=True,
        )


def _make_exp_dir(exp_id):
    d = EXPERIMENTS_DIR / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_config(exp_id, topic, n, note, user_config_path):
    d = _make_exp_dir(exp_id)
    cfg_path = d / "config.json"
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}
    cfg.update(
        {
            "tag_topic": topic,
            "tag_seq": n,
            "tag_note": note or "",
        }
    )
    if user_config_path:
        user_cfg = json.loads(Path(user_config_path).read_text())
        cfg.update(user_cfg)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg_path


def _run_command(cmd, log_path):
    """Invoke command via subprocess; stream to terminal + log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def cmd_run(args):
    """Run a command bracketed by -pre/-post git tags.

    Lightweight snapshotting (no worktree, no copy):
      1. Verify the working tree is clean.
      2. Create experiments/<id>/config.json with the upcoming tag scheme.
      3. Mint a -pre lightweight tag at HEAD.
      4. Run the command, capturing stdout+stderr into train.log.
      5. Stage just experiments/<id>/ (and tracked edits if asked), commit,
         mint a -post lightweight tag at the new commit.
    """
    repo_root = ROOT
    if not args.pre_only:
        _require_clean_tree(repo_root)

    exp_id = args.exp_id
    topic = normalize_topic(args.topic)

    # If --post-only is set and a matching -pre tag exists, reuse its seq;
    # otherwise mint a fresh seq.
    note = args.note or ""
    pre_tag_existing = None
    if args.post_only:
        out = subprocess.run(
            ["git", "tag", "--sort=v:refname", "-l", f"{TAG_PREFIX_EXP}{topic}/*-pre"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        tags = out.stdout.splitlines()
        if tags:
            last = tags[-1]
            m = re.match(
                rf"^{re.escape(TAG_PREFIX_EXP)}{re.escape(topic)}/(\d+)-pre$", last
            )
            if m:
                n = int(m.group(1))
                pre_tag_existing = last
        if pre_tag_existing is None:
            sys.exit(
                f"--post-only: no existing exp/{topic}/NNN-pre tag found; nothing to anchor to"
            )
    else:
        n = next_exp_number(repo_root, topic)
    pre_tag = f"{TAG_PREFIX_EXP}{topic}/{n:03d}-pre"
    post_tag = f"{TAG_PREFIX_EXP}{topic}/{n:03d}-post"

    # Always (re)write config.json with the upcoming tag, even on --post-only.
    cfg_path_before = None
    if not args.dry_run:
        cfg_path_before = _write_config(exp_id, topic, n, note, args.config)
        # include config.json in the pre-tag repo state too
        if not args.pre_only:
            # we re-stage after writing, but if we're going to commit later
            # this file must exist before the run (so its state can be part
            # of the -pre snapshot's *next* commit recipients).
            pass

    print(f"# Experiment: experiments/{exp_id}/")
    print(f"#   pre  tag: {pre_tag}")
    print(f"#   post tag: {post_tag}")
    print(f"#   topic:    {topic}")
    print(f"#   seq:      {n:03d}")
    if note:
        print(f"#   note:     {note}")

    if args.dry_run:
        print("# (dry-run: nothing will be tagged, committed, or executed)")
        return

    # 3) -pre tag at HEAD
    if not args.post_only:
        git_tag(pre_tag, target="HEAD", msg=f"pre-run snapshot for experiments/{exp_id}")
        print(f"# tagged {pre_tag} → HEAD")

    if args.pre_only:
        print("# --pre-only set; stopping before the run")
        return

    # 4) Run the command
    if args.command:
        log_path = EXPERIMENTS_DIR / exp_id / "train.log"
        print(f"# running: {' '.join(args.command)}  →  {log_path}")
        rc = _run_command(args.command, log_path)
        print(f"# exit code: {rc}")
        if rc != 0:
            print(
                f"# WARNING: command exited non-zero ({rc}); continuing to post-snapshot anyway."
            )
    else:
        print("# no --command supplied; skipping run, going straight to post-snapshot")

    # 5) Post-snapshot
    _stage_experiment(repo_root, exp_id, include_edits=args.include_edits)

    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    if not staged:
        print("# (nothing staged for the post-snapshot; nothing to commit)")
        git_tag(post_tag, target="HEAD", msg=f"post-run snapshot for experiments/{exp_id} (no diff)")
        print(f"# tagged {post_tag} → HEAD (no commit)")
        return

    msg = f"experiment {exp_id}: post-run snapshot (seq {n:03d}, topic {topic})"
    if note:
        msg += f"\n\n{note}"
    subprocess.run(["git", "commit", "-m", msg], cwd=repo_root, check=True)
    new_commit = _current_commit(repo_root)
    git_tag(post_tag, target="HEAD", msg=f"post-run snapshot for experiments/{exp_id}")
    print(f"# tagged {post_tag} → {new_commit[:7]}")
    # Help link command:
    print(f"# next: link to a claim:  python ./src/_tag.py link {pre_tag[:-4]} theo/<topic>/<CID>")


def _now():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Mint and inspect tags for experiments and theory claims."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # exp — mint a tag for a current experiment
    s_exp = sub.add_parser("exp", help="mint an experiment tag from experiments/<id>/")
    s_exp.add_argument("exp_id", help="experiment id (directory name under experiments/)")
    s_exp.add_argument("--topic", required=True, help="topic slug (e.g. byte_state_byte)")
    s_exp.add_argument("--note", help="note to embed into config.json and the tag message")
    s_exp.add_argument("--apply", action="store_true", help="actually run `git tag`")
    s_exp.add_argument("--target", default="HEAD", help="what commit the tag points to")
    s_exp.set_defaults(func=cmd_exp)

    # theo — mint a tag for a theory claim
    s_theo = sub.add_parser("theo", help="mint a claim tag for a theory")
    s_theo.add_argument("topic", help="theory topic (e.g. dendrite_growth)")
    s_theo.add_argument("claim_id", nargs="?", help="claim id (e.g. G1a). default: next H<N>")
    s_theo.add_argument("--note", help="note")
    s_theo.add_argument("--apply", action="store_true", help="actually run `git tag`")
    s_theo.add_argument("--target", default="HEAD", help="what commit the tag points to")
    s_theo.set_defaults(func=cmd_theo)

    # list
    s_list = sub.add_parser("list", help="list git tags by kind")
    s_list.add_argument("--exp", action="store_true", help="only exp/ tags")
    s_list.add_argument("--theo", action="store_true", help="only theo/ tags")
    s_list.set_defaults(func=cmd_list)

    # show
    s_show = sub.add_parser("show", help="print info about a tag")
    s_show.add_argument("tag", help="the tag to inspect")
    s_show.set_defaults(func=cmd_show)

    # link
    s_link = sub.add_parser("link", help="link an experiment tag to a theory claim tag")
    s_link.add_argument("exp_tag", help="exp/<topic>/<NNN>")
    s_link.add_argument("theo_tag", help="theo/<topic>/<claim>")
    s_link.set_defaults(func=cmd_link)

    # run — execute a command with pre/post git-tag snapshots
    s_run = sub.add_parser(
        "run",
        help="run a command, snapshotted by exp/-pre and exp/-post git tags",
    )
    s_run.add_argument(
        "--id",
        dest="exp_id",
        required=True,
        help="experiment id (directory name under experiments/, e.g. byte_loop_002)",
    )
    s_run.add_argument("--topic", required=True, help="topic slug for the exp tag (e.g. byte_state_byte)")
    s_run.add_argument(
        "--config",
        help="optional path to a JSON file with config.json fields; copied into experiments/<id>/config.json",
    )
    s_run.add_argument(
        "--note",
        help="note for both the -pre and -post tags (recorded in config too)",
    )
    s_run.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="command + args to run after `--`. Captured into experiments/<id>/train.log",
    )
    s_run.add_argument(
        "--include-edits",
        action="store_true",
        help="if the run modifies tracked files, also commit those edits into the -post commit (default: only commit experiment/<id>/)",
    )
    s_run.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would happen without creating tags, commits, or running the command",
    )
    s_run.add_argument(
        "--pre-only",
        action="store_true",
        help="only mint the -pre tag and exit (do not run, do not post-snapshot)",
    )
    s_run.add_argument(
        "--post-only",
        action="store_true",
        help="skip the -pre tag and the run; assume the experiment is already done, just commit + tag it",
    )
    s_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
