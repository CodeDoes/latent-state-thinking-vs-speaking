"""Converged hybrid latent-state model (SSM think + FFN speak).

- One shared SSM with a `derive` switch (source | answer). Confidence is the
  transformed training loss; it loops (soft-gated in training) until loss<eps.
- A forward FFN (latent->tokens) with a complete head, derive-conditioned.
- Cross-mode FFN(L_src, derive=ANS)->Answer is trained so inference works.
- A token-by-token BaselineAR of similar size for the "latent vs tokens" test.

Synthetic random dataset: ~256 categorized vocab, multi-fact worlds rendered
as A=prose / B=json / C=question / D=answer. d_state is set from L*.
"""
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    source = [t for (s, v, o) in facts for t in (s, v, o, ".")]
    data = [t for (s, v, o) in facts for t in ("JSON", s, v, o)]
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


# ---------------- latent model: SSM think + FFN speak ----------------
class LatentModel(nn.Module):
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
        self.q_enc = nn.Linear(d_emb, d_emb)
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
        dev = self.emb.weight.device
        qv = self.qvec(torch.tensor(q_ids, device=dev))
        h = torch.zeros(1, 1, self.gru.hidden_size, device=dev)
        toks = [self.bos] + tgt_ids
        logits, comps = [], []
        for i in range(len(tgt_ids)):
            x = torch.cat([self.emb(torch.tensor(toks[i], device=dev)), s, der, qv]
                          ).unsqueeze(0).unsqueeze(0)
            out, h = self.gru(x, h)
            logits.append(self.out(out[0, 0]))
            comps.append(torch.sigmoid(self.comp(out[0, 0])))
        logits = torch.stack(logits)
        comps = torch.stack(comps).squeeze(1)
        ce = F.cross_entropy(logits, torch.tensor(tgt_ids, device=dev))
        tgt_comp = torch.zeros(len(tgt_ids), device=dev)
        tgt_comp[-1] = 1.0
        bce = F.binary_cross_entropy(comps, tgt_comp)
        return ce + bce

    def ffn_gen(self, s, der_id, q_ids, max_len=12, tau=0.5):
        der = self.der_emb(der_id)
        dev = self.emb.weight.device
        qv = self.qvec(torch.tensor(q_ids, device=dev))
        h = torch.zeros(1, 1, self.gru.hidden_size, device=dev)
        toks = [self.bos]
        out_ids = []
        for _ in range(max_len):
            x = torch.cat([self.emb(torch.tensor(toks[-1], device=dev)), s, der, qv]
                          ).unsqueeze(0).unsqueeze(0)
            out, h = self.gru(x, h)
            nid = int(self.out(out[0, 0]).argmax())
            out_ids.append(nid)
            if torch.sigmoid(self.comp(out[0, 0])).item() > tau or nid == self.eos:
                break
            toks.append(torch.tensor(nid, device=dev))
        return out_ids


# ---------------- baseline: token-by-token AR (no latent) ----------------
class BaselineAR(nn.Module):
    """Standard autoregressive decoder: given (source+question) tokens, it
    generates the answer token-by-token. No latent think-loop / compression.
    Comparable capacity to LatentModel for the 'latent vs tokens' test."""

    def __init__(self, vocab, d_emb=32, d_hidden=64):
        super().__init__()
        self.vocab = len(vocab)
        self.emb = nn.Embedding(self.vocab, d_emb, padding_idx=0)
        self.ctx_enc = nn.Linear(d_emb, d_hidden)
        self.gru = nn.GRU(d_emb, d_hidden, batch_first=True)
        self.out = nn.Linear(d_hidden, self.vocab)
        self.comp = nn.Linear(d_hidden, 1)

    def forward_loss(self, ctx_ids, q_ids, tgt_ids, bos):
        if isinstance(tgt_ids, torch.Tensor):
            tgt_ids = tgt_ids.tolist()
        if isinstance(q_ids, torch.Tensor):
            q_ids = q_ids.tolist()
        if isinstance(ctx_ids, torch.Tensor):
            ctx_ids = ctx_ids.tolist()
        dev = self.emb.weight.device
        c = self.emb(torch.tensor(ctx_ids + q_ids, device=dev)).mean(0)
        h = self.ctx_enc(c).unsqueeze(0).unsqueeze(0)
        toks = [bos] + tgt_ids
        logits, comps = [], []
        for i in range(len(tgt_ids)):
            x = self.emb(torch.tensor(toks[i], device=dev)).unsqueeze(0).unsqueeze(0)
            out, h = self.gru(x, h)
            logits.append(self.out(out[0, 0]))
            comps.append(torch.sigmoid(self.comp(out[0, 0])))
        logits = torch.stack(logits)
        comps = torch.stack(comps).squeeze(1)
        ce = F.cross_entropy(logits, torch.tensor(tgt_ids, device=dev))
        tgt_comp = torch.zeros(len(tgt_ids), device=dev)
        tgt_comp[-1] = 1.0
        bce = F.binary_cross_entropy(comps, tgt_comp)
        return ce + bce

    def generate(self, ctx_ids, q_ids, bos, max_len=12, tau=0.5):
        if isinstance(ctx_ids, torch.Tensor):
            ctx_ids = ctx_ids.tolist()
        if isinstance(q_ids, torch.Tensor):
            q_ids = q_ids.tolist()
        dev = self.emb.weight.device
        c = self.emb(torch.tensor(ctx_ids + q_ids, device=dev)).mean(0)
        h = self.ctx_enc(c).unsqueeze(0).unsqueeze(0)
        toks = [bos]
        out_ids = []
        for _ in range(max_len):
            x = self.emb(torch.tensor(toks[-1], device=dev)).unsqueeze(0).unsqueeze(0)
            out, h = self.gru(x, h)
            nid = int(self.out(out[0, 0]).argmax())
            out_ids.append(nid)
            if torch.sigmoid(self.comp(out[0, 0])).item() > tau or nid == self.eos:
                break
            toks.append(torch.tensor(nid, device=dev))
        return out_ids
