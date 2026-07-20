"""Byte -> (state, mask) time-mix encoder model.

Shared by src/train_byte_time_mix.py and src/eval_byte_time_mix.py so neither
needs to import the other's training side-effects.

Input:  raw uint8[24] (stored as byte_ids = 2 + byte, padding = 0).
Output: tuple (state, mask_logits)
  state: (B, 2560) time-mix state (tm_out), a MASK-GATED mean-pool of the
         minGRU hidden states — padding positions are zeroed by the gate
         before pooling, so they cannot corrupt the tm_out estimate.
  mask:  (B, 24) per-position validity logit. Estimated FIRST from the raw
         uint8 sequence, then used to gate the state. find_split() returns
         the first position where sigmoid(mask) drops below threshold (the
         token boundary); the token bytes are byte_ids[:, :split].
"""
from pathlib import Path
import torch
import torch.nn as nn
from minGRU_pytorch import minGRU

VOCAB_PATH = Path(__file__).parent / "rwkv_vocab_v20230424.txt"


class ByteTimeMixEncoder(nn.Module):
    def __init__(self, dim=256, max_bytes=24):
        super().__init__()
        # Raw uint8[24]: each byte is a scalar in 0..255 -> dim.
        self.byte_proj = nn.Linear(1, dim)
        self.pos = nn.Embedding(max_bytes, dim)
        self.gru = minGRU(dim)
        self.out = nn.Linear(dim, 2560)          # state head (gated pool -> 2560)
        self.mask_head = nn.Linear(dim, 1)       # per-position validity logit

    def forward(self, byte_ids):
        # byte_ids: (B, T) with values (2 + b); recover true uint8, normalize.
        B, T = byte_ids.shape
        raw = (byte_ids.float() - 2).clamp(0, 255) / 255.0  # (B, T) in [0,1]
        x = self.byte_proj(raw.unsqueeze(-1))               # (B, T, dim)
        pos = torch.arange(T, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos(pos)
        h = self.gru(x)                                     # (B, T, dim)

        # 1) confidence/validity mask FIRST: per-position logit -> gate
        mask_logits = self.mask_head(h).squeeze(-1)          # (B, T)
        gate = torch.sigmoid(mask_logits).unsqueeze(-1)     # (B, T, 1)

        # 2) state estimate GATED by the mask: masked mean-pool of hiddens
        gated = h * gate                                     # (B, T, dim)
        denom = gate.sum(dim=1).clamp(min=1e-3)              # (B, 1)
        pooled = gated.sum(dim=1) / denom                    # (B, dim)
        state = self.out(pooled)                             # (B, 2560)
        return state, mask_logits


def find_split(mask_logits, threshold=0.5):
    """Split position = first index where sigmoid(mask) < threshold.

    The mask is a validity gate (1 = real byte, 0 = padding). The token ends
    at the first position that drops below threshold; the token's bytes are
    byte_ids[:, :split]. Returns (B,) long tensor (clamped to T if all valid).
    """
    B, T = mask_logits.shape
    valid = torch.sigmoid(mask_logits) >= threshold          # (B, T) bool
    inv = ~valid
    idx = torch.full((B,), T, dtype=torch.long, device=mask_logits.device)
    first = inv.float().argmax(dim=1)                        # first 1 (or 0 if none)
    has = inv.any(dim=1)
    idx[has] = first[has]
    return idx
