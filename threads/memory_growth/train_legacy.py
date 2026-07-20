#!/usr/bin/env python3
"""Train + compare BaselineAR vs LatentThink on the fast multi-query task.

Both models are sized to the SAME parameter budget. The point being tested:
a tiny model that thinks ONCE (encodes the context into a state) and speaks
MANY (answers each query cheaply from that state) should match the equal-size
autoregressive baseline that re-encodes the full context for every query --
while doing a fraction of the per-query compute.

Local, CPU, fast by design. Logs to experiments/<exp_id>/.


[meta]
status: triage-needed
[/meta]
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from domains.byte.tokenizer import SymbolTokenizer
from threads.memory_growth.dataset import build_vocab, generate_dataset
from threads.memory_growth.models import BaselineAR, LatentThink, count_params, match_baseline_hidden


def pad(seqs, max_len, pad):
    return [s[:max_len] + [pad] * (max_len - len(s[:max_len])) for s in seqs]


@torch.no_grad()
def evaluate(latent, baseline, ctx_t, q_t, widx, a_t, qtype, dev):
    # latent: think once per world, speak per query
    st = latent.think(ctx_t.to(dev))
    lat_logits = latent.speak(st[widx], q_t.to(dev))
    lat_pred = lat_logits.argmax(1)
    # baseline: re-encode context+question per query
    base_logits = baseline(ctx_t[widx].to(dev), q_t.to(dev))
    base_pred = base_logits.argmax(1)

    def acc(pred):
        correct = (pred == a_t.to(dev)).float()
        overall = correct.mean().item()
        by_type = {}
        for t in ("WHERE", "AT", "SAME"):
            m = torch.tensor([q == t for q in qtype])
            by_type[t] = correct[m].mean().item() if m.any() else None
        return overall, by_type

    return acc(lat_pred), acc(base_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--d_emb", type=int, default=16)
    ap.add_argument("--d_hidden_enc", type=int, default=32)
    ap.add_argument("--d_hidden_speak", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--exp_id", default="exp001")
    ap.add_argument("--max_events", type=int, default=8)
    args = ap.parse_args()
    dev = args.device

    syms, cats = build_vocab()
    tok = SymbolTokenizer(syms)
    V = tok.vocab_size
    worlds = generate_dataset(n=args.n, seed=args.seed, max_events=args.max_events)

    # Flatten worlds into (world_idx, context, question, answer, qtype) rows.
    ctx_ids, rows = [], []
    max_ctx = 0
    max_q = 0
    for wi, w in enumerate(worlds):
        c = tok.encode(w["context"])
        ctx_ids.append(c)
        max_ctx = max(max_ctx, len(c))
        for q, a in w["queries"]:
            qe = tok.encode(q)
            ae = tok.encode([a])[0]
            rows.append((wi, qe, ae, q[0]))
            max_q = max(max_q, len(qe))
    max_ctx = min(max_ctx, 48)
    max_q = max(max_q, 4)

    ctx_pad = pad(ctx_ids, max_ctx, tok.pad)
    ctx_t = torch.tensor(ctx_pad, dtype=torch.long)
    q_pad = pad([r[1] for r in rows], max_q, tok.pad)
    q_t = torch.tensor(q_pad, dtype=torch.long)
    a_t = torch.tensor([r[2] for r in rows], dtype=torch.long)
    widx = torch.tensor([r[0] for r in rows], dtype=torch.long)
    qtype = [r[3] for r in rows]
    N = len(rows)

    # Models: match param counts for a fair fight.
    latent = LatentThink(V, d_emb=args.d_emb,
                         d_hidden_enc=args.d_hidden_enc,
                         d_hidden_speak=args.d_hidden_speak).to(dev)
    nlat = count_params(latent)
    b_h = match_baseline_hidden(V, args.d_emb, args.d_hidden_enc,
                                args.d_hidden_speak, nlat)
    baseline = BaselineAR(V, d_emb=args.d_emb, d_hidden=b_h).to(dev)
    nbase = count_params(baseline)
    print(f"vocab={V}  latent_params={nlat}  baseline_params={nbase} "
          f"(d_hidden={b_h})  queries={N}")

    optL = torch.optim.Adam(latent.parameters(), lr=1e-3)
    optB = torch.optim.Adam(baseline.parameters(), lr=1e-3)

    # Token-processing counts (the "compute" the point is about).
    ctx_tok_total = sum(len(c) for c in ctx_ids)
    q_tok_total = sum(len(r[1]) for r in rows)
    base_compute = N * 0 + sum(len(c) for c in ctx_ids) * 0  # placeholder
    base_compute = sum(len(ctx_ids[wi]) + len(r[1]) for r in rows)  # per query: ctx+q
    lat_compute = ctx_tok_total + q_tok_total  # ctx once + q per query
    compute_ratio = base_compute / max(lat_compute, 1)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(N)
        tl = tb = 0.0
        n = 0
        for s in range(0, N, args.batch):
            idx = perm[s:s + args.batch]
            c = ctx_t[widx[idx]].to(dev)
            q = q_t[idx].to(dev)
            a = a_t[idx].to(dev)

            optB.zero_grad()
            lb = F.cross_entropy(baseline(c, q), a)
            lb.backward()
            optB.step()
            tb += lb.item()

            optL.zero_grad()
            st = latent.think(c)
            ls = F.cross_entropy(latent.speak(st, q), a)
            ls.backward()
            optL.step()
            tl += ls.item()
            n += 1
        lat_acc, base_acc = evaluate(latent, baseline, ctx_t, q_t, widx, a_t, qtype, dev)
        print(f"epoch {ep:2d}  loss_lat={tl/n:.3f} loss_base={tb/n:.3f}  "
              f"lat_acc={lat_acc[0]:.3f} base_acc={base_acc[0]:.3f}", flush=True)

    lat_acc, base_acc = evaluate(latent, baseline, ctx_t, q_t, widx, a_t, qtype, dev)
    elapsed = time.time() - t0

    # A few sample predictions (expected vs generated) for a sanity read.
    st = latent.think(ctx_t.to(dev))
    lat_pred = latent.speak(st[widx], q_t.to(dev)).argmax(1)
    base_pred = baseline(ctx_t[widx].to(dev), q_t.to(dev)).argmax(1)
    samples = []
    for i in range(min(8, N)):
        samples.append(dict(
            qtype=qtype[i],
            question=" ".join(tok.decode([r[1] for r in rows][i])),
            expected=tok.itos[a_t[i].item()],
            latent=tok.itos[lat_pred[i].item()],
            baseline=tok.itos[base_pred[i].item()],
        ))

    result = {
        "hypothesis": "think-once (latent) vs re-encode-per-query (baseline), equal params",
        "vocab": V,
        "latent_params": nlat,
        "baseline_params": nbase,
        "baseline_d_hidden": b_h,
        "n_worlds": args.n,
        "n_queries": N,
        "epochs": args.epochs,
        "elapsed_s": round(elapsed, 1),
        "latent_overall_acc": lat_acc[0],
        "baseline_overall_acc": base_acc[0],
        "latent_by_type": lat_acc[1],
        "baseline_by_type": base_acc[1],
        "compute_ratio_base_over_latent": round(compute_ratio, 2),
        "winner": ("latent" if lat_acc[0] > base_acc[0]
                   else "baseline" if base_acc[0] > lat_acc[0] else "tie"),
        "samples": samples,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))

    out = Path("experiments") / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with (out / "samples.txt").open("w") as f:
        for s in samples:
            f.write(f"[{s['qtype']}] Q: {s['question']}\n")
            f.write(f"   expected={s['expected']}  latent={s['latent']}  baseline={s['baseline']}\n")
    print(f"\nSaved to {out}/ (config.json, metrics.json, samples.txt)")


if __name__ == "__main__":
    main()
