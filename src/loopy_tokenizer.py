#!/usr/bin/env python3
"""Loopy RNN tokenizer emulator.

A learned front-end that reads raw bytes and emits token IDs into a frozen
RWKV-7 model. It emulates the greedy TRIE tokenizer using a recurrent loop:
read bytes one at a time, accumulate state, decide when to emit a token.

Architecture:
    bytes → [loopy RNN (reads, accumulates, triggers)] → token_id → [frozen RWKV-7]

The loopy RNN learns the exact same byte→token mapping as the world
tokenizer's TRIE, but as a learned network. The frozen model never knows
the difference — it sees token IDs either way.

In future, the whole stack can be retrained end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.rwkv_nano import RWKV7Block, count_params
from minGRU_pytorch import minGRU

BYTE_VOCAB_SIZE = 258
BYTE_PAD = 0
BYTE_TO_ID = {b: 2 + b for b in range(256)}

WORLD_PAD_ID = 0


class AccumulatorCell(nn.Module):
    """Single step of the loopy RNN: read a byte, update accumulator state.

    The cell maintains a hidden state that accumulates information across
    bytes. At each step, it outputs:
    - A candidate token logit (what token would be emitted if we trigger now)
    - A trigger probability (should we emit now or keep reading?)
    """

    def __init__(self, dim: int, world_vocab: int, head_size: int = 64):
        super().__init__()
        self.dim = dim

        # Byte embedding
        self.byte_embed = nn.Embedding(BYTE_VOCAB_SIZE, dim, padding_idx=BYTE_PAD)

        # State update: minGRU acting as a recurrent cell
        self.cell = minGRU(dim)

        # Trigger head: decide when to emit
        self.trigger_head = nn.Linear(dim, 1)

        # Token head: predict which world token to emit
        # Small bottleneck dim to keep params low
        self.bottleneck_dim = 128
        self.token_proj = nn.Linear(dim, self.bottleneck_dim)
        self.token_head = nn.Linear(self.bottleneck_dim, world_vocab, bias=False)

    def forward(self, byte_id: torch.Tensor, prev_hidden: torch.Tensor = None):
        """One step: read a byte, update state, output trigger + token logits.

        Returns trigger as raw logits (pre-sigmoid) for numerical stability
        with binary_cross_entropy_with_logits.

        Args:
            byte_id: (B,) byte token IDs
            prev_hidden: (B, 1, dim) previous hidden or None
        Returns:
            logits: (B, world_vocab) candidate token logits
            trigger_logits: (B,) raw trigger logits (pre-sigmoid)
            new_hidden: (B, 1, dim) updated hidden
        """
        x = self.byte_embed(byte_id).unsqueeze(1)  # (B, 1, dim)
        x, new_hidden = self.cell(x, prev_hidden, return_next_prev_hidden=True)
        h = x.squeeze(1)
        logits = self.token_head(self.token_proj(h))  # (B, world_vocab)
        trigger_logits = self.trigger_head(h).squeeze(-1)  # (B,)
        return logits, trigger_logits, new_hidden


class LoopyTokenizer(nn.Module):
    """Loopy RNN that reads bytes and emits token IDs for a frozen model.

    For each byte position:
    1. Read byte, update accumulator state
    2. Check trigger: if triggered, emit the highest-probability token
       and feed it to the frozen model; reset accumulator for next token
    3. If not triggered, keep reading bytes into accumulator

    During training, the trigger threshold is relaxed (gumbel-softmax or
    straight-through) for differentiability. During inference, it's hard.
    """

    def __init__(self, dim: int, world_vocab: int, trigger_bias: float = -2.0, head_size: int = 64):
        super().__init__()
        self.dim = dim
        self.world_vocab = world_vocab

        # The accumulator cell — one per potential token position
        self.cell = AccumulatorCell(dim, world_vocab, head_size)

        # Bias trigger toward keeping-reading initially
        self.trigger_bias = nn.Parameter(torch.tensor(trigger_bias))

    def forward_emulate(
        self, byte_ids: torch.Tensor,
        target_tokens: Optional[list[list[int]]] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Emulate tokenizer: read bytes, emit tokens via adaptive triggering."""
        B, T = byte_ids.shape
        device = byte_ids.device

        acc_hidden = None
        emitted = []
        trigger_logs = []
        token_logits_list = []

        for t in range(T):
            byte_t = byte_ids[:, t]
            is_pad = (byte_t == BYTE_PAD)
            if is_pad.all():
                break

            logits, trigger, acc_hidden = self.cell(byte_t, acc_hidden)
            token_logits_list.append(logits)
            trigger = trigger + self.trigger_bias
            triggered = trigger > 0.0
            trigger_logs.append(trigger)

        token_logits = torch.stack(token_logits_list, dim=1)
        emitted_tokens = token_logits.argmax(dim=-1)
        pad_mask = (byte_ids == BYTE_PAD)
        emitted_tokens = emitted_tokens.masked_fill(pad_mask, WORLD_PAD_ID)

        info = {
            'trigger_rates': torch.stack(trigger_logs).mean().item(),
            'token_logits': token_logits,
        }

        return emitted_tokens, info


class AccumulatorCellWithPos(AccumulatorCell):
    """AccumulatorCell that also takes a position (0..255) as input.

    Position encoding helps the cell learn token boundaries by making
    byte position directly accessible, rather than relying solely on
    the recurrent state to encode position information.
    """
    def __init__(self, dim, world_vocab, head_size=64):
        super().__init__(dim, world_vocab, head_size)
        self.pos_embed = nn.Embedding(256, dim)

    def forward(self, byte_id, pos, prev_hidden=None):
        byte_emb = self.byte_embed(byte_id).unsqueeze(1)  # (B, 1, dim)
        pos_emb = self.pos_embed(pos)                      # (B, L, dim)
        x = byte_emb + pos_emb
        x, new_hidden = self.cell(x, prev_hidden, return_next_prev_hidden=True)
        h = x.squeeze(1)
        logits = self.token_head(self.token_proj(h))
        trigger_logits = self.trigger_head(h).squeeze(-1)  # (B,)
        return logits, trigger_logits, new_hidden


class LoopyTokenizerWithPos(LoopyTokenizer):
    """LoopyTokenizer that feeds byte position into the cell."""
    def __init__(self, dim, world_vocab, trigger_bias=-2.0, head_size=64):
        super().__init__(dim, world_vocab, trigger_bias, head_size)
        self.cell = AccumulatorCellWithPos(dim, world_vocab, head_size)

    def forward_bytes(self, byte_ids):
        """byte_ids: (B, T) → token_logits (B, T, V), triggers (B, T)"""
        B, T = byte_ids.shape
        device = byte_ids.device
        acc_hidden = None
        logits_list, trig_list = [], []
        for t in range(T):
            pos = torch.full((B, 1), min(t, 255), device=device, dtype=torch.long)
            logits, trigger, acc_hidden = self.cell(byte_ids[:, t], pos, acc_hidden)
            logits_list.append(logits)
            trig_list.append(trigger)
        token_logits = torch.stack(logits_list, dim=1)   # (B, T, V)
        trigger_logits = torch.stack(trig_list, dim=1) + self.trigger_bias  # (B, T)
        return token_logits, trigger_logits


class FrozenModelWithLoopyFront(nn.Module):
    """Frozen RWKV-7 with a loopy tokenizer front-end.

    Architecture:
        bytes → loopy_RNN → token_ids → [frozen RWKV-7 layers 0..L-1] → logits
    """

    def __init__(self, core_model, loopy: LoopyTokenizer):
        super().__init__()
        self.loopy = loopy
        self.core = core_model
        # Freeze everything in core
        for p in self.core.parameters():
            p.requires_grad = False

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        emitted_tokens, info = self.loopy.forward_emulate(byte_ids)
        logits, _ = self.core(emitted_tokens)
        return logits


# ── Quick test ──

if __name__ == "__main__":
    from src.rwkv_nano import RWKV7Nano

    core = RWKV7Nano(vocab_size=100, dim=64, head_size=32, num_layers=2)
    loopy = LoopyTokenizer(64, 100)
    model = FrozenModelWithLoopyFront(core, loopy)

    byte_ids = torch.randint(2, 258, (2, 32))
    logits = model(byte_ids)
    print(f"Input:  {byte_ids.shape}")
    print(f"Output: {logits.shape}")
    print(f"Params: {count_params(model.loopy):,} (trainable only)")
    print("OK")
