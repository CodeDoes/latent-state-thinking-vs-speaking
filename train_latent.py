#!/usr/bin/env python3
"""Prototype of the converged hybrid latent-state design (full synthetic).

SSM (think): one shared SSM, `derive` switch, confidence = transformed loss,
             loops (soft-gated in training) until loss<eps.
FFN (speak): trained FORWARD (latent->tokens) + complete head, derive-conditioned.
Cross-mode FFN(L_src, derive=ANS)->Answer is trained so inference works.

Dataset: synthetic random worlds, ~256 categorized vocab, multi-fact,
rendered as A=prose / B=json / C=question / D=answer.
d_state is set from L* (information-theoretic size), NOT guessed.

Local --quick = CPU sanity only. No real training, no Kaggle.
"""
import argparse, math, random
import torch, torch.nn as nn, torch.nn.functional as F


# ---------------- vocab / dataset ----------------
def build_vocab():
    cats = {
        "name": [f"N{i}" for i in range(80)],
        "obj":  [f"O{i}" for i in range(80)],
        "verb": [f"V{i}" for i in range(32)],
        "pron": [f"P{i}" for i in range(24)],
        "loc":  [f"L{i}" for i in range(32)],
        "punct": [".", ",", '"', "'", "?", ":"],
        "data": ["{", "}", "JSON"],
    }
    syms = [s for v in cats.values() for s in v]
    special = ["<PAD>", "<BOS>", "<EOS>", "<DERIVE_SRC>", "<DERIVE_ANS>"]
    vocab = special + syms  # ~260 tokens
    return vocab, cats


class Tok:
    def __init__(self, vocab, cats):
        self.vocab = vocab
        self.cats = cats
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.itos = vocab
        self.pad = self.stoi["<PAD>"]
        self.bos = self.stoi["<BOS>"]
        self.eos = self.stoi["<EOS>"]
        self.der_src = self.stoi["<DERIVE_SRC>"]
        self.der_ans = self.stoi["<DERIVE_ANS>"]

    def enc(self, toks):
        return [self.stoi[t] for t in toks if t in self.stoi]

    def dec(self, ids):
        return [self.itos[i] for i in ids]


def gen_world(tok, rng, max_facts=4):
    k = rng.randint(1, max_facts)
    facts = []
    for _ in range(k):
        subj = rng.choice(tok.cats["name"])
        verb = rng.choice(tok.cats["verb"])
        obj = rng.choice(tok.cats["obj"] + tok.cats["loc"])
        facts.append((subj, verb, obj))
    # source (A = prose) and data (B = json-ish token form)
    source = [t for (s, v, o) in facts for t in (s, v, o, ".")]
    data = [t for (s, v, o) in facts for t in ("JSON", s, v, o)]
    # question (C) + answer (D): pick one fact, ask its object
    qi = rng.randrange(k)
    qs, qv, qo = facts[qi]
    question = [qs, qv, "?"]
    answer = [qo]
    return {"source": source, "data": data, "question": question,
            "answer": answer, "k": k}


def compute_Lstar(cats, max_facts=4, bits_per_float=16):
    """Theoretical best latent size (information-theoretic)."""
    d_fact = (math.log2(len(cats["name"])) + math.log2(len(cats["verb"]))
              + math.log2(len(cats["obj"]) + len(cats["loc"])))
    bits = max_facts * d_fact + math.log2(2) + 4 + 4  # +switch+storyVariant+misc
    floats = math.ceil(bits / bits_per_float)
    return int(floats), round(bits, 1)


# ---------------- model ----------------
class Net(nn.Module):
    def __init__(self, vocab, d_emb=32, d_ctx=48, d_state=16, d_der=8,
                 d_hidden=64):
        super().__init__()
        self.vocab = len(vocab)
        self.d_emb = d_emb
        self.d_state = d_state
        self.d_der = d_der
        self.emb = nn.Embedding(self.vocab, d_emb, padding_idx=0)
        self.ctx_enc = nn.Linear(d_emb, d_ctx)
        self.tgt_enc = nn.Linear(d_emb, d_state)
        self.q_enc = nn.Linear(d_emb, d_emb)  # question summary
        self.der_emb = nn.Embedding(2, d_der)
        self.ssm = nn.Sequential(nn.Linear(d_state + d_ctx + d_der, d_state),
                                 nn.Tanh())
        self.s0 = nn.Parameter(torch.zeros(d_state))
        self.gru = nn.GRU(d_emb + d_state + d_der + d_emb, d_hidden,
                          batch_first=True)
        self.out = nn.Linear(d_hidden, self.vocab)
        self.comp = nn.Linear(d_hidden, 1)

    def ctx(self, ids):
        return self.ctx_enc(self.emb(ids).mean(0))

    def target(self, ids):
        return self.tgt_enc(self.emb(ids).mean(0))

    def qvec(self, ids):
        return self.q_enc(self.emb(ids).mean(0))

    def ssm_loss(self, ctx, der_id, K, target, lam=0.1):
        der = self.der_emb(der_id)
        s = self.s0
        states, confs = [], []
        for _ in range(K):
            s = self.ssm(torch.cat([s, ctx, der]))
            states.append(s)
            confs.append(1.0 / (1.0 + F.mse_loss(s, target)))
        ws, prod = [], 1.0
        for c, o in zip(confs, [torch.ones_like(c) - c for c in confs]):
            ws.append(c * prod)
            prod = prod * o
        wsum = sum(ws) + prod
        ws = [w / wsum for w in ws]
        prod /= wsum
        s_eff = sum(w * st for w, st in zip(ws, states)) + prod * states[-1]
        loss = F.mse_loss(s_eff, target) + lam * sum(
            w * F.mse_loss(st, target) for w, st in zip(ws, states))
        return loss, s_eff

    def ssm_hard(self, ctx, der_id, K, target, tau=0.5):
        der = self.der_emb(der_id)
        s = self.s0
        for _ in range(K):
            s = self.ssm(torch.cat([s, ctx, der]))
            if (1.0 / (1.0 + F.mse_loss(s, target))).item() > tau:
                break
        return s

    def ffn_loss(self, s, der_id, q_ids, tgt_ids):
        if isinstance(tgt_ids, torch.Tensor):
            tgt_ids = tgt_ids.tolist()
        if isinstance(q_ids, torch.Tensor):
            q_ids = q_ids.tolist()
        der = self.der_emb(der_id)
        qv = self.qvec(torch.tensor(q_ids))
        h = torch.zeros(1, 1, self.gru.hidden_size)
        toks = [self.bos] + tgt_ids
        logits, comps = [], []
        for i in range(len(tgt_ids)):
            x = torch.cat([self.emb(torch.tensor(toks[i])), s, der, qv]
                          ).unsqueeze(0).unsqueeze(0)
            out, h = self.gru(x, h)
            logits.append(self.out(out[0, 0]))
            comps.append(torch.sigmoid(self.comp(out[0, 0])))
        logits = torch.stack(logits)
        comps = torch.stack(comps).squeeze(1)
        ce = F.cross_entropy(logits, torch.tensor(tgt_ids))
        tgt_comp = torch.zeros(len(tgt_ids))
        tgt_comp[-1] = 1.0
        bce = F.binary_cross_entropy(comps, tgt_comp)
        return ce + bce

    def ffn_gen(self, s, der_id, q_ids, max_len=12, tau=0.5):
        der = self.der_emb(der_id)
        qv = self.qvec(torch.tensor(q_ids))
        h = torch.zeros(1, 1, self.gru.hidden_size)
        toks = [self.bos]
        out_ids = []
        for _ in range(max_len):
            x = torch.cat([self.emb(torch.tensor(toks[-1])), s, der, qv]
                          ).unsqueeze(0).unsqueeze(0)
            out, h = self.gru(x, h)
            nid = int(self.out(out[0, 0]).argmax())
            out_ids.append(nid)
            if torch.sigmoid(self.comp(out[0, 0])).item() > tau or nid == self.eos:
                break
            toks.append(torch.tensor(nid))
        return out_ids


# ---------------- training ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--d_state", type=int, default=0)  # 0 => use L*
    ap.add_argument("--n_samples", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_facts", type=int, default=4)
    args = ap.parse_args()
    if args.quick:
        args.n_samples = 200
        args.epochs = 3
        args.K = 4

    vocab, cats = build_vocab()
    tok = Tok(vocab, cats)
    Lstar_floats, Lstar_bits = compute_Lstar(cats, args.max_facts)
    d_state = args.d_state if args.d_state > 0 else max(8, min(48, Lstar_floats + 8))
    rng = random.Random(0)
    data = [gen_world(tok, rng, args.max_facts) for _ in range(args.n_samples)]

    net = Net(vocab, d_state=d_state).to(args.device)
    net.bos = tok.bos
    net.eos = tok.eos
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    der_src = torch.tensor(0)
    der_ans = torch.tensor(1)

    print(f"vocab={len(vocab)} L*={Lstar_bits}b->{Lstar_floats}f "
          f"d_state={d_state} n={len(data)} epochs={args.epochs} K={args.K}")
    for ep in range(args.epochs):
        tot = 0.0
        for s in data:
            srcq = torch.tensor(tok.enc(s["source"] + s["question"]))
            q = torch.tensor(tok.enc(s["question"]))
            a = torch.tensor(tok.enc(s["answer"]))
            ctx_src, tgt_src = net.ctx(srcq), net.target(srcq)
            ctx_ans, tgt_ans = net.ctx(a), net.target(a)
            l1, se = net.ssm_loss(ctx_src, der_src, args.K, tgt_src)
            l2, ae = net.ssm_loss(ctx_ans, der_ans, args.K, tgt_ans)
            l3 = net.ffn_loss(se, der_src, tok.enc(s["question"]), tok.enc(s["question"]))   # L_src -> question
            l4 = net.ffn_loss(ae, der_ans, tok.enc(s["question"]), tok.enc(s["answer"]))     # L_ans -> answer
            l5 = net.ffn_loss(se, der_ans, tok.enc(s["question"]), tok.enc(s["answer"]))     # L_src -> answer (cross-mode)
            loss = l1 + l2 + l3 + l4 + l5
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  epoch {ep} loss={tot / len(data):.4f}")

    print("\nINFERENCE (L_src -> Answer):")
    correct = 0
    for s in data[:8]:
        srcq = torch.tensor(tok.enc(s["source"] + s["question"]))
        q = tok.enc(s["question"])
        a = tok.enc(s["answer"])
        ctx_src, tgt_src = net.ctx(srcq), net.target(srcq)
        s_src = net.ssm_hard(ctx_src, der_src, args.K, tgt_src, tau=0.5)
        gen = net.ffn_gen(s_src, der_ans, q, max_len=12, tau=0.5)
        pred = tok.dec(gen)
        ok = pred == tok.dec(a)
        correct += ok
        print(f"  k={s['k']} Q={s['question']} A={tok.dec(a)} pred={pred} "
              f"{'OK' if ok else 'xx'}")
    print(f"  sanity accuracy (8 samples) = {correct}/8")


if __name__ == "__main__":
    main()
