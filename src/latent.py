"""Converged hybrid latent-state model (SSM think + FFN speak).

Key design decisions (the ones that actually test the hypothesis):

1. SEQUENTIAL THINKING. The SSM/recurrent cell processes the source
   TOKEN-BY-TOKEN, updating a fixed-size latent state `s` at every step
   (`s = think(s, x_t, derive)`). This is the whole point of a latent
   state: it is a *running summary* of the input, not a mean-pooled bag of
   tokens. The original mean-pooled version could not track order at all, so
   it could never test long-horizon reasoning.

2. THINK ONCE, SPEAK MANY. A world has one long source and several
   questions. The latent model builds the state `L_src` ONCE from the source,
   then answers every question cheaply from that single state. The token-by-
   token baseline must re-encode the full source for every question. This is
   the "thinking separated from speaking" win condition made measurable.

3. LONG-HORIZON + INTEGRATION TASKS. Worlds are move-chains with
   distractors; answers require integrating many remote events and can be
   multi-token (set aggregation). Single-token recall is replaced by
   reasoning over a long, interfering context.

4. The `derive` switch (SRC | ANS) and confidence-from-loss thinking loop
   are preserved; ANS mode is a training-only booster, discarded at inference.

A forward FFN (latent -> tokens) with a complete head, derive-conditioned,
is trained on the INFERENCE PATH `FFN(L_src, derive=ANS) -> Answer` in every
batch, so inference never silently breaks.
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- vocab / dataset -----------------------------
def build_vocab():
    cats = {
        "name": [f"N{i}" for i in range(80)],
        "obj": [f"O{i}" for i in range(80)],
        "verb": [f"V{i}" for i in range(32)],
        "pron": [f"P{i}" for i in range(24)],
        "loc": [f"L{i}" for i in range(32)],
        "punct": [".", ",", '"', "'", "?", ":"],
        "data": ["{", "}", "JSON"],
        # relation / query / answer-role tokens
        "rel": ["AT", "WHERE", "SAME", "NONE"],
    }
    syms = [s for v in cats.values() for s in v]
    special = ["<PAD>", "<BOS>", "<EOS>", "<DERIVE_SRC>", "<DERIVE_ANS>"]
    vocab = special + syms  # ~266 tokens
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
        return [self.itos[int(i)] for i in ids]


def gen_world(tok, rng, max_events=14, n_items_range=(3, 6)):
    """Generate one world: a long chain of item->location moves (with
    distractor moves interleaved) plus several questions whose answers
    require integrating many remote events.

    Returns:
        source  : list[str]  tokenized move log  ("O3 L5 . O3 L2 . ...")
        queries : list[(question_tokens, answer_tokens)]
        k       : number of (real) move events
    """
    locs = tok.cats["loc"]
    items = rng.sample(tok.cats["obj"], rng.randint(*n_items_range))  # O# items
    loc_of = {it: rng.choice(locs) for it in items}

    events = []  # (item, loc) real moves
    n_moves = rng.randint(max(4, max_events // 2), max_events)
    for _ in range(n_moves):
        it = rng.choice(items)
        newloc = rng.choice(locs)
        events.append((it, newloc))
        loc_of[it] = newloc
        # distractor: another item also moves (noise for the tracker)
        if rng.random() < 0.5:
            it2 = rng.choice([x for x in items if x != it])
            l2 = rng.choice(locs)
            events.append((it2, l2))
            loc_of[it2] = l2

    source = [t for (it, l) in events for t in (it, l, ".")]

    queries = []
    nq = rng.randint(4, 8)  # more queries/world -> amortized thinking matters
    for _ in range(nq):
        # integration-heavy mix: reasoning (AT/SAME) dominates; WHERE
        # (precise trajectory recall) is the tape's job, not the SSM's.
        qt = rng.choices(["WHERE", "AT", "SAME"], weights=[0.3, 0.35, 0.35], k=1)[0]
        if qt == "WHERE":
            it = rng.choice(items)
            q = ["WHERE", it, "?"]
            a = [loc_of[it]]
        elif qt == "AT":
            lk = rng.choice(locs)
            members = [it for it in items if loc_of[it] == lk]
            q = ["AT", lk, "?"]
            a = members if members else ["NONE"]
        else:  # SAME
            it = rng.choice(items)
            lk = loc_of[it]
            others = [x for x in items if x != it and loc_of[x] == lk]
            q = ["SAME", it, "?"]
            a = others if others else ["NONE"]
        queries.append((q, a))

    return {"source": source, "queries": queries, "k": len(events),
            "loc_of": loc_of, "items": items}


def compute_Lstar(cats, n_items=6, bits_per_float=16):
    """Theoretical best latent size (information-theoretic floor)."""
    d_item = (math.log2(len(cats["obj"])) + math.log2(len(cats["loc"])))
    bits = n_items * d_item + math.log2(len(cats["rel"])) + 8
    floats = math.ceil(bits / bits_per_float)
    return int(floats), round(bits, 1)


# ----------------- latent model: sequential think + speak -----------------
class LatentModel(nn.Module):
    def __init__(self, vocab, d_emb=128, d_state=64, d_der=8, d_hidden=256,
                 n_locs=32, scale=1):
        super().__init__()
        self.vocab = len(vocab)
        self.d_state = d_state
        self.n_locs = n_locs
        self.scale = scale
        # GRAND idea: a TRANSFORMER encodes the (small) context; the latent
        # state then loops (re-attends) the encoder reps to "derive the future",
        # then decodes rapidly. Scale the transformer by `scale` (= max_loop)
        # so the loop has a rich context representation; capped at 512-d to
        # preserve quality.
        d_ctx = min(d_emb * scale, 512)
        self.d_emb = d_ctx
        self.der_emb = nn.Embedding(2, d_der)
        self.emb = nn.Embedding(self.vocab, d_ctx, padding_idx=0)
        self.s0 = nn.Parameter(torch.zeros(d_state))
        nhead = 4 if d_ctx % 4 == 0 else 2
        self.ctx_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_ctx, nhead, dim_feedforward=2 * d_ctx,
                                       batch_first=True),
            num_layers=min(scale, 4))
        # recurrent think cell: (state, token_rep, derive) -> new_state
        self.think = nn.Sequential(
            nn.Linear(d_state + d_ctx + d_der, 2 * d_state),
            nn.Tanh(),
            nn.Linear(2 * d_state, d_state),
            nn.Tanh(),
        )
        self.qvec = nn.Linear(d_ctx, d_ctx)
        self.speak = nn.GRU(2 * d_ctx + d_state + d_der, d_hidden, batch_first=True)
        self.out = nn.Linear(d_hidden, self.vocab)
        self.comp = nn.Linear(d_hidden, 1)
        # T09 readiness head (loop early-stop)
        self.state_conf = nn.Linear(d_state, 1)
        # T06 auxiliary reconstruction head: predict each item's final
        # location from the latent state (forces the state to encode the
        # trajectory/relational info AT/SAME queries need).
        self.recon = nn.Linear(d_state + d_ctx, n_locs)

    # ---- thinking: transformer-encode context, fold, then loop-derive ----
    def think_state(self, ids, der_id, K=1, loop_max=0, min_certainty=0.9,
                    train=False):
        """Encode context ONCE with the transformer; fold state over the reps;
        then re-attend the reps `loop_max` times to 'derive the future' (cheap,
        small context, no re-tokenization). Stops early at inference when
        readiness >= min_certainty. Then rapid-decode many tokens from `s`."""
        der = self.der_emb(der_id).unsqueeze(0)
        s = self.s0.unsqueeze(0)
        if ids.numel() == 0:
            return s
        x = self.emb(ids).unsqueeze(0)            # [1, T, d_ctx]
        reps = self.ctx_enc(x)[0]                 # [T, d_ctx]  (context encoded ONCE)
        T = reps.size(0)
        for t in range(T):
            xt = reps[t].unsqueeze(0)
            for _ in range(K):
                s = self.think(torch.cat([s, xt, der], dim=1))
        if loop_max <= 0:
            return s
        for _ in range(loop_max):
            if not train and torch.sigmoid(self.state_conf(s.squeeze(0))).item() >= min_certainty:
                break
            for t in range(T):
                s = self.think(torch.cat([s, reps[t].unsqueeze(0), der], dim=1))
        return s

    def state_confidence(self, s):
        """Readiness in [0,1]: is the latent oriented / ready to answer?"""
        return torch.sigmoid(self.state_conf(s.squeeze(0)))

    def ffn_loss(self, s, der_id, q_ids, tgt_ids, lam_comp=1.0):
        """Train FFN(L, derive) -> target tokens (teacher-forced)."""
        if isinstance(tgt_ids, (list, tuple)):
            tgt_ids = torch.tensor(tgt_ids, device=s.device)
        if isinstance(q_ids, (list, tuple)):
            q_ids = torch.tensor(q_ids, device=s.device)
        der = self.der_emb(der_id).unsqueeze(0).unsqueeze(0)  # [1,1,d_der]
        s3 = s.unsqueeze(1)                                   # [1,1,d_state]
        qv = self.qvec(self.emb(q_ids).mean(0)).unsqueeze(0).unsqueeze(0)  # [1,1,d_emb]
        h = torch.zeros(1, 1, self.speak.hidden_size, device=s.device)
        toks = [self.bos] + tgt_ids.tolist()
        logits, comps = [], []
        for i in range(len(tgt_ids)):
            xt = self.emb(torch.tensor(toks[i], device=s.device)).unsqueeze(0).unsqueeze(0)
            out, h = self.speak(torch.cat([xt, s3, der, qv], dim=2), h)
            logits.append(self.out(out[0, 0]))
            comps.append(torch.sigmoid(self.comp(out[0, 0])))
        logits = torch.stack(logits)
        comps = torch.stack(comps).squeeze(1)
        ce = F.cross_entropy(logits, tgt_ids)
        tgt_comp = torch.zeros(len(tgt_ids), device=s.device)
        tgt_comp[-1] = 1.0
        bce = F.binary_cross_entropy(comps, tgt_comp)
        return ce + lam_comp * bce

    def ffn_gen(self, s, der_id, q_ids, max_len=16, tau=0.5):
        """Greedy generation from state `s`. Returns list[int] token ids."""
        if isinstance(q_ids, (list, tuple)):
            q_ids = torch.tensor(q_ids, device=s.device)
        der = self.der_emb(der_id).unsqueeze(0).unsqueeze(0)
        s3 = s.unsqueeze(1)
        qv = self.qvec(self.emb(q_ids).mean(0)).unsqueeze(0).unsqueeze(0)
        h = torch.zeros(1, 1, self.speak.hidden_size, device=s.device)
        toks = [self.bos]
        out_ids = []
        for _ in range(max_len):
            xt = self.emb(torch.tensor(toks[-1], device=s.device)).unsqueeze(0).unsqueeze(0)
            out, h = self.speak(torch.cat([xt, s3, der, qv], dim=2), h)
            nid = int(self.out(out[0, 0]).argmax())
            out_ids.append(nid)
            if torch.sigmoid(self.comp(out[0, 0])).item() > tau or nid == self.eos:
                break
            toks.append(nid)
        if out_ids and out_ids[-1] == self.eos:
            out_ids = out_ids[:-1]
        return out_ids

    # ---- T06 auxiliary reconstruction: item -> final location from state ----
    def recon_loss(self, s, loc_of, tok):
        """Force the latent state to encode each item's final location.

        AT ('is O3 at L5?') and SAME ('is O3 with O7?') can only be answered
        if the state retains the trajectory/relational mapping item->location,
        which the bare next-token objective discards (T04). This auxiliary CE
        over items makes that mapping explicit in `s`, so the speaker can
        decode relational answers instead of collapsing to 'NONE'.
        """
        if not loc_of:
            return torch.zeros((), device=s.device)
        s_flat = s.squeeze(0)                      # [d_state]
        losses = []
        for it, lk in loc_of.items():
            it_id = tok.stoi.get(it)
            if it_id is None or lk not in tok.cats["loc"]:
                continue
            lk_idx = tok.cats["loc"].index(lk)   # 0..n_locs-1, NOT vocab index
            ie = self.emb(torch.tensor(it_id, device=s.device))   # [d_emb]
            logits = self.recon(torch.cat([s_flat, ie], dim=-1))  # [n_locs]
            losses.append(F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor(lk_idx, device=s.device).unsqueeze(0)))
        if not losses:
            return torch.zeros((), device=s.device)
        return torch.stack(losses).mean()


# ----------------- baseline: token-by-token AR (no latent) -----------------
class BaselineAR(nn.Module):
    """Standard autoregressive decoder over the raw token sequence. It must
    re-encode the FULL source for every question (no reusable latent state).
    Comparable capacity to LatentModel for the 'latent vs tokens' test."""

    def __init__(self, vocab, d_emb=128, d_hidden=256):
        super().__init__()
        self.vocab = len(vocab)
        self.emb = nn.Embedding(self.vocab, d_emb, padding_idx=0)
        self.gru = nn.GRU(d_emb, d_hidden, batch_first=True)
        self.out = nn.Linear(d_hidden, self.vocab)
        self.comp = nn.Linear(d_hidden, 1)

    def _encode(self, ids):
        x = self.emb(ids).unsqueeze(0)          # [1, T, d_emb]
        _, h = self.gru(x)                       # h: [1, 1, d_hidden]
        return h

    def forward_loss(self, ctx_ids, q_ids, tgt_ids, bos, lam_comp=1.0):
        if isinstance(ctx_ids, (list, tuple)):
            ctx_ids = torch.tensor(ctx_ids, device=self.emb.weight.device)
        if isinstance(q_ids, (list, tuple)):
            q_ids = torch.tensor(q_ids, device=self.emb.weight.device)
        if isinstance(tgt_ids, (list, tuple)):
            tgt_ids = torch.tensor(tgt_ids, device=self.emb.weight.device)
        h = self._encode(torch.cat([ctx_ids, q_ids]))
        toks = [bos] + tgt_ids.tolist()
        logits, comps = [], []
        for i in range(len(tgt_ids)):
            xt = self.emb(torch.tensor(toks[i], device=self.emb.weight.device)).unsqueeze(0).unsqueeze(0)
            out, h = self.gru(xt, h)
            logits.append(self.out(out[0, 0]))
            comps.append(torch.sigmoid(self.comp(out[0, 0])))
        logits = torch.stack(logits)
        comps = torch.stack(comps).squeeze(1)
        ce = F.cross_entropy(logits, tgt_ids)
        tgt_comp = torch.zeros(len(tgt_ids), device=tgt_ids.device)
        tgt_comp[-1] = 1.0
        bce = F.binary_cross_entropy(comps, tgt_comp)
        return ce + lam_comp * bce

    def generate(self, ctx_ids, q_ids, bos, max_len=16, tau=0.5):
        if isinstance(ctx_ids, (list, tuple)):
            ctx_ids = torch.tensor(ctx_ids, device=self.emb.weight.device)
        if isinstance(q_ids, (list, tuple)):
            q_ids = torch.tensor(q_ids, device=self.emb.weight.device)
        h = self._encode(torch.cat([ctx_ids, q_ids]))
        toks = [bos]
        out_ids = []
        for _ in range(max_len):
            xt = self.emb(torch.tensor(toks[-1], device=self.emb.weight.device)).unsqueeze(0).unsqueeze(0)
            out, h = self.gru(xt, h)
            nid = int(self.out(out[0, 0]).argmax())
            out_ids.append(nid)
            if torch.sigmoid(self.comp(out[0, 0])).item() > tau or nid == self.eos:
                break
            toks.append(nid)
        if out_ids and out_ids[-1] == self.eos:
            out_ids = out_ids[:-1]
        return out_ids
