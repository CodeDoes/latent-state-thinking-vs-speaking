#!/usr/bin/env python3
"""Experiment lifecycle tool — the bookkeeping half of the engineering loop.

`python -m src` *runs* experiments. This tool manages everything around the
run so that no experiment exists without a theory, a hypothesis, a baseline,
and a verdict. See AGENTS.md ("The Loop") for the mandated workflow.

Usage:
    python -m src.experiment theory  <slug> [--dir adaptive]
    python -m src.experiment new     <exp_id> --type T --theory PATH
                                     --variable V --baseline EXP_ID
                                     [--hypothesis H] [--set k=v]...
    python -m src.experiment record  <exp_id> key=value [key=value]...
    python -m src.experiment verdict <exp_id> supports|refutes|inconclusive
                                     --note "..."
    python -m src.experiment compare <exp_id_a> <exp_id_b>
    python -m src.experiment index        # regenerate experiments/INDEX.md
    python -m src.experiment audit        # report experiments missing
                                          # required artifacts / fields

Design rules enforced here:
  * `new` refuses to scaffold an experiment without a theory doc that exists.
  * `new` refuses an existing directory (one dir per run, never overwritten).
  * `verdict` only accepts the three canonical verdicts.
  * `audit` is the "definition of done" checker; run it before committing.

Template placeholders use {{NAME}} syntax (see templates/).
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
TEMPLATES = ROOT / "templates"

VERDICTS = ("supports", "refutes", "inconclusive")

# keys in config.json the loop requires; audit checks these
REQUIRED_CONFIG_FIELDS = ("seed", "git_hash", "theory", "variable", "baseline")

# preferred order when picking a headline metric out of metrics.json
METRIC_PREFERENCE = (
    "best_loss", "final_loss", "val_loss", "loss",
    "cosine_similarity", "cosine", "accuracy", "acc", "val_mse",
)

# ── [meta] docstring contract ─────────────────────────────────────────────
# Agents habitually comment their code; the comments ARE the registry.
# Every src/*.py module carries a [meta] block in its docstring:
#
#     [meta]
#     status: active | triage-needed | superseded-by:src/x.py | archived
#     theory: theories/<dir>/<slug>.md      (the spec it tests)
#     experiment: <exp_id>[, <exp_id>...]   (runs it produces)
#     variable: <the one knob this module varies>
#     baseline: <exp_id>                    (what it is compared against)
#     verdict: supported | refuted | inconclusive | pending
#     params: <parameter count>             (optional)
#     [/meta]
#
# `harvest` parses all blocks into src/meta.json + src/CODEMAP.md; `audit`
# fails on INVALID meta (dangling refs, wrong enums). A module without meta
# is by definition undocumented — the bucket reviewers attack first.

META_STATUS = ("active", "triage-needed", "archived")
META_VERDICTS = VERDICTS + ("pending",)

CODE_DIR = ROOT / "src"
META_JSON = CODE_DIR / "meta.json"
CODEMAP_MD = CODE_DIR / "CODEMAP.md"


# ── small helpers ────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return datetime.date.today().isoformat()


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _coerce(value: str):
    """'3.5' -> 3.5, 'true' -> True, else str."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_kv(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            sys.exit(f"error: expected key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def _all_exp_roots():
    """Experiment homes: per-thread dirs, then the legacy top-level dir."""
    roots = sorted(ROOT.glob("threads/*/experiments"))
    if EXP_DIR.exists():
        roots.append(EXP_DIR)
    return roots


def _thread_of(path: Path) -> str:
    try:
        rel = path.relative_to(ROOT)
        if rel.parts[0] == "threads":
            return rel.parts[1]
    except Exception:
        pass
    return ""


def _exp_path(exp_id: str) -> Path:
    """Locate an experiment dir by id across all threads."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-+]+", exp_id):
        sys.exit(f"error: bad experiment id {exp_id!r} "
                 "(letters, digits, _ . - + only)")
    for root in _all_exp_roots():
        cand = root / exp_id
        if cand.exists():
            return cand
    return EXP_DIR / exp_id  # default (may not exist yet)


def _fill_template(name: str, mapping: dict) -> str:
    text = (TEMPLATES / name).read_text()
    for key, val in mapping.items():
        text = text.replace("{{" + key + "}}", str(val))
    leftover = re.findall(r"\{\{(\w+)\}\}", text)
    for key in set(leftover):
        text = text.replace("{{" + key + "}}", f"<{key.lower()}>")
    return text


def _headline_metric(metrics: dict):
    """Pick (key, value) that best summarizes a metrics.json dict."""
    for key in METRIC_PREFERENCE:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key, metrics[key]
    return None, None


def _final_metric(exp: Path):
    """Best-effort key metric from metrics.json, else last metrics.jsonl."""
    m = _read_json(exp / "metrics.json")
    key, val = _headline_metric(m)
    if val is not None:
        return key, val
    jl = exp / "metrics.jsonl"
    if jl.exists():
        for line in reversed(jl.read_text(errors="replace").splitlines()):
            try:
                row = json.loads(line)
            except Exception:
                continue
            key, val = _headline_metric(row)
            if val is not None:
                return key, val
    return None, None


# ── [meta] parsing & harvest ─────────────────────────────────────────────


def _docstring(path: Path) -> str:
    import ast
    try:
        tree = ast.parse(path.read_text(errors="replace"))
        return ast.get_docstring(tree) or ""
    except Exception:
        return ""


def _parse_meta(doc: str) -> dict:
    """Extract the [meta] ... ([/meta]) block from a docstring."""
    if "[meta]" not in doc:
        return {}
    block = doc.split("[meta]", 1)[1]
    block = block.split("[/meta]", 1)[0]
    out = {}
    for line in block.strip().splitlines():
        line = line.split("#", 1)[0].strip()       # allow inline comments
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[key.strip()] = val.strip()
    return out


def _split_ids(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


CODE_ROOTS = ("kit", "domains", "threads", "src")


def harvest_modules():
    """Walk kit/domains/threads/src; return {relpath: meta-dict}."""
    out = {}
    for base in CODE_ROOTS:
        base_p = ROOT / base
        if not base_p.exists():
            continue
        for f in sorted(base_p.rglob("*.py")):
            if f.name == "__init__.py":
                continue
            key = str(f.relative_to(ROOT))
            out[key] = _parse_meta(_docstring(f))
    return out


def validate_meta(metas):
    """Return list of (module, problem) for invalid blocks."""
    problems = []
    for name, meta in metas.items():
        if not meta:
            continue
        status = meta.get("status", "")
        if not status:
            problems.append((name, "meta missing required 'status'"))
        elif status not in META_STATUS \
                and not status.startswith("superseded-by:"):
            problems.append((name, f"bad status {status!r}"))
        verdict = meta.get("verdict")
        if verdict and verdict not in META_VERDICTS:
            problems.append((name, f"bad verdict {verdict!r}"))
        theory = meta.get("theory")
        if theory and not (ROOT / theory).exists():
            problems.append((name, f"theory {theory!r} does not exist"))
        for key in ("experiment", "baseline"):
            if meta.get(key):
                for eid in _split_ids(meta[key]):
                    if not _exp_path(eid).exists():
                        problems.append(
                            (name, f"{key} {eid!r} not in experiments/"))
    return problems


def _codemap_lines(metas):
    out = ["# Code Map",
           "",
           "> Generated by `python -m src.experiment harvest` from the",
           "> `[meta]` blocks in module docstrings — edit the docstrings,",
           "> never this file. Missing meta = undocumented = triage queue.",
           ""]
    buckets = {}
    for name, meta in sorted(metas.items()):
        status = meta.get("status", "undocumented") if meta else "undocumented"
        head = status.split(":")[0]
        buckets.setdefault(head, []).append((name, meta))

    out.append(" | ".join(f"**{k}**: {len(v)}" for k, v in
                          sorted(buckets.items())))
    out.append("")
    for bucket in sorted(buckets):
        out.append(f"## {bucket}")
        out.append("")
        out.append("| Module | Theory | Experiment | Variable | "
                   "Baseline | Verdict |")
        out.append("|---|---|---|---|---|---|")
        for name, meta in buckets[bucket]:
            theory = meta.get("theory", "").replace("theories/", "")
            out.append(
                f"| `{name}` | {theory} | {meta.get('experiment', '')} "
                f"| {meta.get('variable', '')} | {meta.get('baseline', '')} "
                f"| {meta.get('verdict', '')} |")
        out.append("")
    out.append(f"_Regenerated {_now()}._")
    return out


def cmd_harvest(args):
    metas = harvest_modules()
    _write_json(META_JSON, {k: v for k, v in metas.items() if v})
    CODEMAP_MD.write_text("\n".join(_codemap_lines(metas)) + "\n")
    problems = validate_meta(metas)
    n_doc = sum(1 for v in metas.values() if v)
    print(f"harvested {len(metas)} modules: {n_doc} documented, "
          f"{len(metas) - n_doc} undocumented -> src/meta.json, src/CODEMAP.md")
    for name, prob in problems:
        print(f"  INVALID {name}: {prob}")
    if problems:
        sys.exit(1)


# ── commands ─────────────────────────────────────────────────────────────


def cmd_theory(args):
    slug = args.slug.lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", slug):
        sys.exit("error: slug must be kebab-case lowercase")
    # theory docs live in their thread; an unknown thread slug starts one
    subdir = Path(args.thread)
    if not subdir.is_absolute():
        subdir = ROOT / subdir
    if subdir.name != "theories" and "threads" not in subdir.parts:
        subdir = ROOT / "threads" / args.thread
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "__init__.py").touch(exist_ok=True)
    path = subdir / f"{slug}.md"
    if path.exists():
        sys.exit(f"error: {path} already exists")
    text = _fill_template("THEORY.md", {"SLUG": slug, "DATE": _today()})
    path.write_text(text)
    print(f"created {path.relative_to(ROOT)}")
    print("next: fill 'Prior art' (>=2 external attempts) and 'Success")
    print("criterion' BEFORE writing any code that tests this theory.")
    print("prior-art tools: research/*.md | ls research/ | "
          "python src/arxiv_query.py -n 10 'all:<query>'")


def cmd_new(args):
    thread = args.thread
    if not thread and args.theory.startswith("threads/"):
        thread = args.theory.split("/")[1]
    if thread:
        dest_root = ROOT / "threads" / thread / "experiments"
    else:
        dest_root = _exp_path(args.exp_id).parent             if _exp_path(args.exp_id).exists() else None
        if dest_root is None:
            sys.exit("error: cannot infer thread from --theory "
                     f"({args.theory}); pass --thread <slug>")
    exp_dir = dest_root / args.exp_id
    if exp_dir.exists():
        sys.exit(f"error: {exp_dir} already exists; one dir per run, "
                 "pick the next id")

    theory_rel = args.theory
    theory_path = ROOT / theory_rel
    if not theory_path.exists():
        sys.exit(f"error: theory doc {theory_rel} does not exist.\n"
                 f"theory first, code second — scaffold one with:\n"
                 f"  python -m src.experiment theory <slug>")

    baseline = args.baseline
    if baseline and not _exp_path(baseline).exists():
        sys.exit(f"error: baseline experiment {baseline!r} not found under "
                 "experiments/ — relative to WHAT will you compare?")

    exp_dir.mkdir(parents=True)
    config = {
        "exp_id": args.exp_id,
        "thread": thread,
        "type": args.type,
        "theory": theory_rel,
        "hypothesis": args.hypothesis or "",
        "variable": args.variable,
        "baseline": baseline or "",
        "seed": 42,
        "created": _now(),
        "git_hash": _git_hash(),
        "verdict": "pending",
    }
    config.update(_parse_kv(args.set or []))
    _write_json(exp_dir / "config.json", config)

    result = _fill_template("EXPERIMENT.md", {
        "EXP_ID": args.exp_id,
        "THEORY": theory_rel,
        "HYPOTHESIS": args.hypothesis or "<one-line falsifiable claim>",
        "VARIABLE": args.variable,
        "BASELINE": baseline or "<baseline exp id>",
        "DATE": _today(),
        "GIT_HASH": config["git_hash"],
    })
    (exp_dir / "RESULT.md").write_text(result)
    print(f"created {exp_dir.relative_to(ROOT)}/ (config.json, RESULT.md)")
    print(f"next: smoke test first (<=60s CPU), then the real run; record with")
    print(f"  python -m src.experiment record {args.exp_id} best_loss=<v> ...")


def cmd_record(args):
    exp_dir = _exp_path(args.exp_id)
    if not exp_dir.exists():
        sys.exit(f"error: {exp_dir} not found")
    values = _parse_kv(args.values)
    row = {"ts": _now(), **values}

    jl = exp_dir / "metrics.jsonl"
    with jl.open("a") as fh:
        fh.write(json.dumps(row) + "\n")

    m_path = exp_dir / "metrics.json"
    metrics = _read_json(m_path)
    metrics.update(values)
    metrics["updated"] = row["ts"]
    _write_json(m_path, metrics)
    print(f"recorded {len(values)} metric(s) into {args.exp_id} "
          f"(metrics.json + metrics.jsonl)")


def _stamp_result_md(path: Path, verdict: str, note: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    text = text.replace("**Verdict:** _pending_", f"**Verdict:** {verdict}")
    # replace the body of the "## Verdict" section up to the next heading
    pattern = re.compile(
        r"(## Verdict\n)\*\*<supports \| refutes \| inconclusive>\*\*.*?"
        r"(?=\n## )",
        re.DOTALL,
    )
    block = f"**{verdict.upper()}** — {note}\n"
    if pattern.search(text):
        text = pattern.sub(lambda m: m.group(1) + block, text)
    else:
        text = text.rstrip() + f"\n\n## Verdict\n{block}"
    path.write_text(text)
    return True


def cmd_verdict(args):
    exp_dir = _exp_path(args.exp_id)
    if not exp_dir.exists():
        sys.exit(f"error: {exp_dir} not found")
    cfg_path = exp_dir / "config.json"
    config = _read_json(cfg_path)
    config["verdict"] = args.verdict
    config["verdict_note"] = args.note
    config["verdict_at"] = _now()
    _write_json(cfg_path, config)

    stamped = _stamp_result_md(exp_dir / "RESULT.md", args.verdict, args.note)
    print(f"{args.exp_id}: verdict = {args.verdict}"
          + ("" if stamped else " (no RESULT.md to stamp)"))
    print("to close the loop:")
    print(f"  1. add the claim to theories/proofs.md "
          f"(verdict: {args.verdict}, commit: {_git_hash()})")
    print( "  2. python -m src.experiment index          # regenerate INDEX.md")
    print( "  3. commit with an honest message (e.g. 'X: refuted, no signal')")


def _diff_row(key, a, b):
    mark = " " if a == b else "*"
    a_s = "—" if a is None else str(a)
    b_s = "—" if b is None else str(b)
    return f"  {mark} {key:24s} {a_s:>22s} | {b_s:<22s}", a != b


def cmd_compare(args):
    a_dir, b_dir = _exp_path(args.a), _exp_path(args.b)
    for p in (a_dir, b_dir):
        if not p.exists():
            sys.exit(f"error: {p} not found")
    ca, cb = _read_json(a_dir / "config.json"), _read_json(b_dir / "config.json")
    ma, mb = _read_json(a_dir / "metrics.json"), _read_json(b_dir / "metrics.json")

    print(f"== CONFIG DIFF  (* = differs) ".ljust(70, "="))
    print(f"    {'':24s} {args.a:>22s} | {args.b:<22s}")
    keys = sorted(set(ca) | set(cb))
    n_diff = 0
    for k in keys:
        line, differs = _diff_row(k, ca.get(k), cb.get(k))
        if differs and k not in ("exp_id", "created", "git_hash", "updated",
                                 "verdict", "verdict_note", "verdict_at",
                                 "hypothesis", "theory"):
            n_diff += 1
            print(line)
    if n_diff == 0:
        print("    (no differing fields besides bookkeeping)")
    if n_diff > 1:
        print(f"    WARNING: {n_diff} differing fields — ideally exactly ONE")
        print("    variable differs between baseline and experiment.")

    print(f"\n== METRICS ====".ljust(70, "="))
    mkeys = [k for k in sorted(set(ma) | set(mb))
             if isinstance(ma.get(k, mb.get(k)), (int, float))]
    for k in mkeys:
        line, _ = _diff_row(k, ma.get(k), mb.get(k))
        print(line)
    ka, va = _final_metric(a_dir)
    kb, vb = _final_metric(b_dir)
    print(f"\nverdicts: {args.a}={ca.get('verdict', '?')}  "
          f"{args.b}={cb.get('verdict', '?')}")
    if va is not None and vb is not None and ka == kb:
        delta = vb - va
        print(f"Δ {ka}: {va:.4g} -> {vb:.4g}  ({delta:+.4g})")


def _index_rows():
    rows = []
    seen = set()
    for exp_root in _all_exp_roots():
        for exp in sorted(p for p in exp_root.iterdir()
                          if p.is_dir() and not p.name.startswith(".")):
            if exp.name in seen:
                continue
            seen.add(exp.name)
            cfg = _read_json(exp / "config.json")
            key, val = _final_metric(exp)
            has_result = (exp / "RESULT.md").exists()
            rows.append({
                "id": exp.name,
                "thread": cfg.get("thread", _thread_of(exp)),
                "type": cfg.get("type", "?"),
                "created": str(cfg.get("created", "?"))[:10],
                "theory": cfg.get("theory", ""),
                "variable": cfg.get("variable", ""),
                "metric": f"{key}={val:.4g}" if val is not None else "—",
                "verdict": cfg.get("verdict", "?"),
                "result_md": has_result,
                "has_cfg": bool(cfg),
                "path": exp,
            })
    return rows


def cmd_index(args):
    rows = _index_rows()
    out = []
    out.append("# Experiment Index")
    out.append("")
    out.append("> Generated by `python -m src.experiment index` — do not edit "
               "by hand;")
    out.append("> fix `experiments/<id>/config.json` / `metrics.json` and "
               "regenerate.")
    out.append("")
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    out.append(f"**{len(rows)} experiments** ({summary or 'none yet'})")
    out.append("")
    header = ("| Experiment | Thread | Type | Date | Theory | Variable | "
              "Key metric | Verdict |")
    sep = "|---|---|---|---|---|---|---|---|"
    out.append(header)
    out.append(sep)
    for r in sorted(rows, key=lambda r: (r["created"], r["id"]), reverse=True):
        theory = r["theory"].replace("theories/", "").replace(
            "threads/", "").replace(".md", "")
        out.append(
            f"| `{r['id']}` | {r['thread']} | {r['type']} | {r['created']} "
            f"| {theory} | {r['variable']} | {r['metric']} | {r['verdict']} |"
        )
    out.append("")
    out.append(f"_Regenerated {_now()} at git {_git_hash()}._")
    target = args.out or (EXP_DIR / "INDEX.md")
    target.write_text("\n".join(out) + "\n")
    print(f"wrote {target.relative_to(ROOT)} "
          f"({len(rows)} experiments)")


def cmd_audit(args):
    rows = _index_rows()
    print("== EXPERIMENT AUDIT ".ljust(64, "="))
    print("required: config.json, metric artifact, config fields "
          + str(REQUIRED_CONFIG_FIELDS) + ",")
    print("          RESULT.md, recorded verdict\n")
    bad = 0
    legacy = 0
    for r in rows:
        if not r["has_cfg"]:
            continue
        exp = _exp_path(r["id"])
        cfg = _read_json(exp / "config.json")
        missing = []
        if not r["has_cfg"]:
            missing.append("config.json")
        if _final_metric(exp)[1] is None:
            missing.append("metrics")
        if not (exp / "RESULT.md").exists():
            missing.append("RESULT.md")
        if cfg.get("verdict", "pending") in ("pending", "?", ""):
            missing.append("verdict")
        for f in REQUIRED_CONFIG_FIELDS:
            if f not in cfg or cfg[f] in ("", "?"):
                missing.append(f"cfg.{f}")
        if missing:
            # experiments that predate the loop: ALL lifecycle fields missing
            is_legacy = all(f"cfg.{f}" in missing for f in
                            ("theory", "variable", "baseline"))
            tag = "legacy" if is_legacy and not args.strict else "FAIL"
            if is_legacy and not args.strict:
                legacy += 1
            else:
                bad += 1
            print(f"  [{tag:6s}] {r['id']:38s} missing: {', '.join(missing)}")
    print(f"\n{len(rows)} experiments: "
          f"{bad} failing, {legacy} legacy (pre-loop), "
          f"{len(rows) - bad - legacy} clean")
    print("tip: fill theory/variable/baseline retroactively in config.json, "
          "record a verdict,")
    print("     or accept legacy status. Nothing is deleted — "
          "negative results are results.")

    # ── code half of the audit: the [meta] docstring contract ──
    print("\n== CODE AUDIT ".ljust(64, "="))
    metas = harvest_modules()
    problems = validate_meta(metas)
    n_doc = sum(1 for v in metas.values() if v)
    print(f"{len(metas)} modules: {n_doc} documented, "
          f"{len(metas) - n_doc} undocumented (no [meta] block), "
          f"{len(problems)} invalid")
    for name, prob in problems:
        print(f"  [FAIL  ] {name:38s} {prob}")
    print("document a module by adding a [meta] block to its docstring — "
          "see templates/MODULE.py")
    sys.exit(1 if (bad or problems) else 0)


# ── main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        prog="python -m src.experiment",
        description="Experiment lifecycle: theory -> new -> record -> "
                    "verdict -> index/compare/audit",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("theory", help="scaffold a theory doc (write BEFORE code)")
    s.add_argument("slug", help="kebab-case name, e.g. 'entropy-gated-exit'")
    s.add_argument("--thread", default="analysis",
                   help="thread/dir the doc belongs to; default threads/analysis")
    s.set_defaults(func=cmd_theory)

    s = sub.add_parser("new", help="scaffold an experiment dir from the template")
    s.add_argument("exp_id")
    s.add_argument("--type", required=True,
                   help="experiment type (see `python -m src list`)")
    s.add_argument("--theory", required=True,
                   help="path to the theory doc this tests")
    s.add_argument("--thread", default="",
                   help="thread slug (default: inferred from --theory path)")
    s.add_argument("--variable", required=True,
                   help="the ONE variable this run changes")
    s.add_argument("--baseline", default="",
                   help="experiment id this is compared against")
    s.add_argument("--hypothesis", default="",
                   help="one-line falsifiable claim")
    s.add_argument("--set", action="append",
                   help="extra config key=value (repeatable)")
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("record", help="record metrics into metrics.json(l)")
    s.add_argument("exp_id")
    s.add_argument("values", nargs="+", help="key=value pairs")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("verdict", help="stamp the run's verdict")
    s.add_argument("exp_id")
    s.add_argument("verdict", choices=VERDICTS)
    s.add_argument("--note", required=True,
                   help="measured vs predicted; what belief changed")
    s.set_defaults(func=cmd_verdict)

    s = sub.add_parser("compare", help="side-by-side config diff + metrics")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("index", help="regenerate experiments/INDEX.md")
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("harvest", help="scrape [meta] docstring blocks from "
                                       "src/*.py into meta.json + CODEMAP.md")
    s.set_defaults(func=cmd_harvest)

    s = sub.add_parser("audit", help="check every experiment against the "
                                     "definition of done")
    s.add_argument("--strict", action="store_true",
                   help="count pre-loop experiments as failures too")
    s.set_defaults(func=cmd_audit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
