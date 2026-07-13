#!/usr/bin/env python3
"""Training + evaluation for the converged hybrid latent-state design.

Trains BOTH the latent model (sequential SSM think + FFN speak) and an
equal-capacity token-by-token baseline (BaselineAR) on the same synthetic
long-horizon reasoning task, then reports val exact-match QA for each -- the
actual "latent thinking vs tokens" test.

The latent model builds its state ONCE from the source and answers every
question in the world from that single state (think once, speak many). The
baseline re-encodes the full source for each question. This is the win
condition made measurable.

Outputs modules_report.json (consumed by the notebook summary cell).

Local --quick = CPU sanity only (tiny data, 3 epochs, small state). Real run
on Kaggle GPU (--device cuda).
"""
import argparse
import json
import random
import math
from collections import defaultdict

import torch
import torch.optim as optim

from src.latent import (build_vocab, Tok, gen_world, compute_Lstar,
                        LatentModel, BaselineAR)


def main():
    print("=== train_converged main() entered (imports done) ===", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--d_state", type=int, default=0)   # 0 => use L*
    ap.add_argument("--d_emb", type=int, default=128)
    ap.add_argument("--d_hidden", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_events", type=int, default=14)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--recon_w", type=float, default=0.0,
                    help="T06 auxiliary reconstruction loss weight (0 = off)")
    ap.add_argument("--no_t05", action="store_true",
                    help="disable T05 uniqueness weighting (uniform answer weights)")
    args = ap.parse_args()
    if args.quick:
        args.n_samples = 400
        args.epochs = 3
        args.K = 2
        args.d_emb = 32
        args.d_hidden = 64
        args.device = "cpu"
        args.max_events = 8

    # device: prefer what was requested, fall back to cpu if unavailable
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable -> using cpu")
        args.device = "cpu"
    dev = args.device
    print("torch", torch.__version__, "device", dev,
          "cuda_available", torch.cuda.is_available())

    vocab, cats = build_vocab()
    tok = Tok(vocab, cats)
    Lstar_floats, Lstar_bits = compute_Lstar(cats)
    d_state = (args.d_state if args.d_state > 0
               else max(16, min(128, Lstar_floats + 24)))
    rng = random.Random(0)
    data = [gen_world(tok, rng, max_events=args.max_events)
            for _ in range(args.n_samples)]
    random.Random(1).shuffle(data)
    nval = max(1, int(args.val_frac * len(data)))
    val, tr = data[:nval], data[nval:]

    # T05 uniqueness-weighted loss: up-weight rare/informative answers so the
    # model can't "win" by always emitting the majority answer (e.g. "NONE",
    # ~87-89% of AT/SAME). w(a) = -log2 p(a) over the corpus; a frequent "NONE"
    # gets a tiny weight, a specific location/item (low p) gets a large one.
    # Applied to BOTH models for a clean A/B.
    ans_str = lambda x: "".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
    ans_counts = defaultdict(int)
    for s in tr:
        for (_q, a) in s["queries"]:
            ans_counts[ans_str(a)] += 1
    total_ans = sum(ans_counts.values()) or 1
    ans_w = {a: -math.log2(c / total_ans) for a, c in ans_counts.items()}
    print(f"T05 uniqueness weights: {len(ans_w)} distinct answers | "
          f"NONE weight={ans_w.get('NONE', 0.0):.3f} "
          f"mean={sum(ans_w.values()) / max(1, len(ans_w)):.3f}")
    if args.no_t05:
        ans_w = {a: 1.0 for a in ans_w}
        print("T05 disabled (--no_t05): uniform answer weights")

    latent = LatentModel(vocab, d_emb=args.d_emb, d_state=d_state,
                         d_hidden=args.d_hidden, n_locs=len(cats["loc"])).to(dev)
    latent.bos = tok.bos
    latent.eos = tok.eos
    baseline = BaselineAR(vocab, d_emb=args.d_emb, d_hidden=args.d_hidden).to(dev)
    baseline.bos = tok.bos
    baseline.eos = tok.eos
    optL = optim.Adam(latent.parameters(), lr=1e-3)
    optB = optim.Adam(baseline.parameters(), lr=1e-3)
    der_src = torch.tensor(0, device=dev)   # derive = SRC
    der_ans = torch.tensor(1, device=dev)   # derive = ANS

    n_lat = sum(p.numel() for p in latent.parameters())
    n_base = sum(p.numel() for p in baseline.parameters())
    print(f"vocab={len(vocab)} L*={Lstar_bits}b->{Lstar_floats}f "
          f"d_state={d_state} d_emb={args.d_emb} d_hidden={args.d_hidden} "
          f"n_train={len(tr)} n_val={len(val)} epochs={args.epochs} K={args.K} "
          f"max_events={args.max_events}")
    print(f"params: latent={n_lat} baseline={n_base}")

    for ep in range(args.epochs):
        for s in tr:
            src_t = torch.tensor(tok.enc(s["source"]), device=dev)
            # --- latent model: think ONCE over the source ---
            s_src = latent.think_state(src_t, der_src, args.K)   # L_src [1,d_state]
            optL.zero_grad()
            ls = 0.0
            if args.recon_w > 0:
                ls = ls + args.recon_w * latent.recon_loss(s_src, s.get("loc_of", {}), tok)
            for (q, a) in s["queries"]:
                q_ids = tok.enc(q)
                a_ids = tok.enc(a)
                a_str = ans_str(a)
                # INFERENCE PATH: answer from the source state (trained every batch)
                ls = ls + ans_w[a_str] * latent.ffn_loss(s_src, der_ans, q_ids, a_ids)
                # keep the source->question path alive (cheap regularizer)
                ls = ls + 0.3 * latent.ffn_loss(s_src, der_src, q_ids, q_ids)
            # ANS-mode booster (training only, discarded at inference):
            # think over the first answer, then reproduce it.
            (q0, a0) = s["queries"][0]
            a0_str = ans_str(a0)
            s_ans = latent.think_state(torch.tensor(tok.enc(a0), device=dev),
                                       der_ans, args.K)
            ls = ls + 0.5 * ans_w[a0_str] * latent.ffn_loss(s_ans, der_ans, tok.enc(q0), tok.enc(a0))
            ls.backward()
            optL.step()
            # --- baseline (token-by-token): re-encode source per question ---
            for (q, a) in s["queries"]:
                optB.zero_grad()
                a_str = ans_str(a)
                loss_b = ans_w[a_str] * baseline.forward_loss(tok.enc(s["source"]), tok.enc(q),
                                                              tok.enc(a), baseline.bos)
                loss_b.backward()
                optB.step()

        # --- validation (with per-query-type breakdown) ---
        accL = accB = n = 0
        tL = defaultdict(lambda: [0, 0])  # [correct, total] latent
        tB = defaultdict(lambda: [0, 0])  # [correct, total] baseline
        for s in val:
            src_t = torch.tensor(tok.enc(s["source"]), device=dev)
            s_src = latent.think_state(src_t, der_src, args.K)
            for (q, a) in s["queries"]:
                qt = q[0]  # 'WHERE' / 'AT' / 'SAME'
                gen = latent.ffn_gen(s_src, der_ans, tok.enc(q),
                                     max_len=16, tau=0.5)
                okL = tok.dec(gen) == tok.dec(tok.enc(a))
                if okL:
                    accL += 1
                    tL[qt][0] += 1
                tL[qt][1] += 1
                bg = baseline.generate(tok.enc(s["source"]), tok.enc(q),
                                      baseline.bos, max_len=16, tau=0.5)
                okB = tok.dec(bg) == tok.dec(tok.enc(a))
                if okB:
                    accB += 1
                    tB[qt][0] += 1
                tB[qt][1] += 1
                n += 1
        pct = lambda d: {k: round(d[k][0] / d[k][1], 4) if d[k][1] else 0.0
                      for k in sorted(d)}
        byL, byB = pct(tL), pct(tB)
        # per-query COMPUTE note: latent thinks ONCE (source, K steps) then
        # speaks N times from a fixed state; baseline re-encodes the full source
        # N times. So per query the latent model is far cheaper as N grows.
        print(f"  epoch {ep} val latent={accL / n:.3f} baseline={accB / n:.3f} "
              f"(n_q={n})  by_type L={byL} B={byB}", flush=True)

    report = {
        "design": "converged SSM(think)+FFN(speak) vs token-by-token baseline",
        "task": "long-horizon multi-hop tracking, multi-query worlds",
        "hypothesis_note": (
            "latent thinks ONCE from the source then answers N queries from a "
            "fixed-size state (amortized); baseline re-encodes the full source "
            "for every query, so it has more info at answer time but pays Nx cost."),
        "Lstar_bits": Lstar_bits,
        "Lstar_floats": Lstar_floats,
        "d_state": d_state,
        "d_emb": args.d_emb,
        "d_hidden": args.d_hidden,
        "vocab": len(vocab),
        "n_train": len(tr),
        "n_val": len(val),
        "n_val_queries": n,
        "q_per_world_avg": round(n / max(1, len(val)), 2),
        "epochs": args.epochs,
        "K": args.K,
        "max_events": args.max_events,
        "T05_uniqueness_weighted": not args.no_t05,
        "T06_recon_weight": args.recon_w,
        "params_latent": n_lat,
        "params_baseline": n_base,
        "val_latent_acc": accL / n,
        "val_baseline_acc": accB / n,
        "latent_by_type": byL,
        "baseline_by_type": byB,
        "interpretation": (
            "latent wins" if accL > accB else
            "baseline wins" if accB > accL else "tie"),
    }
    with open("reports/modules_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("REPORT " + json.dumps(report))


if __name__ == "__main__":
    main()
