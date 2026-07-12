#!/usr/bin/env python3
"""Training + evaluation for the converged hybrid latent-state design.

Trains BOTH the latent model (SSM think + FFN speak) and an equal-capacity
token-by-token baseline (BaselineAR) on the same synthetic task, then reports
val exact-match QA for each -- the actual "latent thinking vs tokens" test.

Outputs modules_report.json (consumed by the notebook summary cell).

Local --quick = CPU sanity only. Real run on Kaggle (--device cuda).
"""
import argparse
import json
import random

import torch
import torch.optim as optim

from src.latent import (build_vocab, Tok, gen_world, compute_Lstar,
                         LatentModel, BaselineAR)


def main():
    print("=== train_converged main() entered (imports done) ===", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--d_state", type=int, default=0)  # 0 => use L*
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_facts", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()
    if args.quick:
        args.n_samples = 400
        args.epochs = 3
        args.K = 4
        args.device = "cpu"

    vocab, cats = build_vocab()
    tok = Tok(vocab, cats)
    Lstar_floats, Lstar_bits = compute_Lstar(cats, args.max_facts)
    d_state = (args.d_state if args.d_state > 0
               else max(8, min(48, Lstar_floats + 8)))
    rng = random.Random(0)
    data = [gen_world(tok, rng, args.max_facts) for _ in range(args.n_samples)]
    random.Random(1).shuffle(data)
    nval = max(1, int(args.val_frac * len(data)))
    val, tr = data[:nval], data[nval:]

    dev = args.device
    latent = LatentModel(vocab, d_state=d_state).to(dev)
    latent.bos = tok.bos
    latent.eos = tok.eos
    baseline = BaselineAR(vocab).to(dev)
    baseline.bos = tok.bos
    baseline.eos = tok.eos
    optL = optim.Adam(latent.parameters(), lr=1e-3)
    optB = optim.Adam(baseline.parameters(), lr=1e-3)
    der_src = torch.tensor(0, device=dev)
    der_ans = torch.tensor(1, device=dev)

    print(f"vocab={len(vocab)} L*={Lstar_bits}b->{Lstar_floats}f "
          f"d_state={d_state} n_train={len(tr)} n_val={len(val)} "
          f"epochs={args.epochs} K={args.K} device={dev}")

    for ep in range(args.epochs):
        for s in tr:
            srcq = torch.tensor(tok.enc(s["source"] + s["question"]), device=dev)
            q = tok.enc(s["question"])
            a = tok.enc(s["answer"])
            qid = torch.tensor(q, device=dev)
            aid = torch.tensor(a, device=dev)
            # --- latent model ---
            c_src, t_src = latent.ctx(srcq), latent.target(srcq)
            c_ans, t_ans = latent.ctx(aid), latent.target(aid)
            l1, se = latent.ssm_loss(c_src, der_src, args.K, t_src)
            l2, ae = latent.ssm_loss(c_ans, der_ans, args.K, t_ans)
            l3 = latent.ffn_loss(se, der_src, qid, q)        # L_src -> question
            l4 = latent.ffn_loss(ae, der_ans, qid, a)        # L_ans -> answer
            l5 = latent.ffn_loss(se, der_ans, qid, a)        # L_src -> answer (cross-mode)
            optL.zero_grad()
            (l1 + l2 + l3 + l4 + l5).backward()
            optL.step()
            # --- baseline (token-by-token) ---
            optB.zero_grad()
            baseline.forward_loss(srcq.tolist(), q, a, baseline.bos).backward()
            optB.step()

        # --- validation ---
        accL = accB = n = 0
        for s in val:
            srcq = torch.tensor(tok.enc(s["source"] + s["question"]), device=dev)
            q = tok.enc(s["question"])
            a = tok.enc(s["answer"])
            c_src, t_src = latent.ctx(srcq), latent.target(srcq)
            s_src = latent.ssm_hard(c_src, der_src, args.K, t_src, tau=0.5)
            gen = latent.ffn_gen(s_src, der_ans, q, max_len=12, tau=0.5)
            if tok.dec(gen) == tok.dec(a):
                accL += 1
            bg = baseline.generate(srcq.tolist(), q, baseline.bos, max_len=12, tau=0.5)
            if tok.dec(bg) == tok.dec(a):
                accB += 1
            n += 1
        print(f"  epoch {ep} val latent={accL / n:.3f} baseline={accB / n:.3f}")

    report = {
        "design": "converged SSM(think)+FFN(speak) vs token-by-token baseline",
        "Lstar_bits": Lstar_bits,
        "Lstar_floats": Lstar_floats,
        "d_state": d_state,
        "vocab": len(vocab),
        "max_facts": args.max_facts,
        "n_train": len(tr),
        "n_val": len(val),
        "epochs": args.epochs,
        "val_latent_acc": accL / n,
        "val_baseline_acc": accB / n,
        "interpretation": (
            "latent wins" if accL > accB else
            "baseline wins" if accB > accL else "tie"),
    }
    with open("modules_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("REPORT " + json.dumps(report))


if __name__ == "__main__":
    main()
