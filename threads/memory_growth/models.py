"""Two small, parameter-matched models for the think-once vs re-encode test.

BaselineAR   : concatenates (context + question) and runs one GRU, re-encoding
               the full context for *every* query. This is the token-by-token
               reference.

LatentThink  : encodes the context ONCE into a fixed state (the "think"), then
               answers each query from that state with a tiny MLP (the "speak").
               The expensive context pass happens a single time.

Both are sized to the SAME parameter count so the comparison isolates
architecture (amortized thinking) from capacity.
"""

import torch
import torch.nn as nn


def count_params(m):
    return sum(p.numel() for p in m.parameters())


class BaselineAR(nn.Module):
    """Autoregressive reference: re-encode context + question per query."""

    def __init__(self, vocab, d_emb=16, d_hidden=64):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, d_emb, padding_idx=0)
        self.gru = nn.GRU(d_emb, d_hidden, batch_first=True)
        self.out = nn.Linear(d_hidden, vocab)

    def forward(self, ctx_ids, q_ids):
        x = torch.cat([ctx_ids, q_ids], dim=1)
        mask = (x != 0).float()
        e = self.emb(x)
        lengths = mask.sum(1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.out(h[0])


class LatentThink(nn.Module):
    """Think once (context -> state), speak many (state + question -> answer)."""

    def __init__(self, vocab, d_emb=16, d_hidden_enc=32, d_hidden_speak=32):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, d_emb, padding_idx=0)
        self.enc = nn.GRU(d_emb, d_hidden_enc, batch_first=True)
        self.speak_fc = nn.Sequential(
            nn.Linear(d_hidden_enc + d_emb, d_hidden_speak),
            nn.SiLU(),
            nn.Linear(d_hidden_speak, vocab),
        )

    def think(self, ctx_ids):
        mask = (ctx_ids != 0).float()
        e = self.emb(ctx_ids)
        lengths = mask.sum(1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        return h[0]  # [B, d_hidden_enc]

    def speak(self, state, q_ids):
        mask = (q_ids != 0).float()
        e = self.emb(q_ids)
        qv = (e * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-6)
        z = torch.cat([state, qv], dim=1)
        return self.speak_fc(z)


def match_baseline_hidden(vocab, d_emb, d_hidden_enc, d_hidden_speak, target):
    """Binary-search a BaselineAR d_hidden whose param count is closest to
    `target` (the LatentThink param count), for a fair fight."""
    lo, hi = 8, 1024
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        p = count_params(BaselineAR(vocab, d_emb, mid))
        if p < target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    pa = count_params(BaselineAR(vocab, d_emb, best))
    pb = count_params(BaselineAR(vocab, d_emb, best + 1))
    return best if abs(pa - target) <= abs(pb - target) else best + 1
