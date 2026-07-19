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

    def __init__(self, dim: int, world_vocab: int):
        super().__init__()
        self.dim = dim

        # Byte embedding
        self.byte_embed = nn.Embedding(BYTE_VOCAB_SIZE, dim, padding_idx=BYTE_PAD)

        # State update: RWKV-7 block acting as a recurrent cell
        self.cell = RWKV7Block(dim)

        # Trigger head: decide when to emit
        self.trigger_head = nn.Linear(dim, 1)

        # Token head: predict which world token to emit
        self.token_head = nn.Linear(dim, world_vocab, bias=False)

    def forward(self, byte_id: torch.Tensor, state: dict):
        """One step: read a byte, update state, output trigger + token logits.

        Args:
            byte_id: (B,) byte token IDs
            state: dict with 'xx' (B, dim) and 'state' (B, H, N, N)
        Returns:
            logits: (B, world_vocab) candidate token logits
            trigger: (B,) trigger probability (0=keep reading, 1=emit now)
            new_state: updated state dict
        """
        B = byte_id.shape[0]
        # Embed byte and reshape for RWKVBlock (expects B, T, dim)
        x = self.byte_embed(byte_id).unsqueeze(1)  # (B, 1, dim)
        x, new_state, _ = self.cell(x, state)

        x = x.squeeze(1)  # (B, dim)
        logits = self.token_head(x)  # (B, world_vocab)
        trigger = torch.sigmoid(self.trigger_head(x))  # (B,)

        return logits, trigger, new_state


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

    def __init__(self, dim: int, world_vocab: int, trigger_bias: float = -2.0):
        super().__init__()
        self.dim = dim
        self.world_vocab = world_vocab

        # The accumulator cell — one per potential token position
        self.cell = AccumulatorCell(dim, world_vocab)

        # Bias trigger toward keeping-reading initially
        self.trigger_bias = nn.Parameter(torch.tensor(trigger_bias))

    def forward_emulate(
        self, byte_ids: torch.Tensor,
        target_tokens: Optional[list[list[int]]] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Emulate tokenizer: read bytes, emit tokens via adaptive triggering.

        During training, if target_tokens is provided, the trigger decisions
        are supervised by the known token boundaries. During inference, the
        model decides autonomously.

        Args:
            byte_ids: (B, T_bytes) byte token IDs
            target_tokens: optional list of B lists of target token IDs
                          (from the real tokenizer) for supervision
        Returns:
            emitted_tokens: (B, max_tokens) token IDs emitted
            info: dict with trigger counts, losses
        """
        B, T = byte_ids.shape
        device = byte_ids.device
        H = self.cell.cell.n_head
        N = self.cell.cell.head_size

        # Initialize accumulator state
        acc_state = {
            'xx': torch.zeros(B, self.dim, device=device),
            'state': torch.zeros(B, H, N, N, device=device),
        }

        emitted = []
        trigger_logs = []
        token_logits_list = []

        for t in range(T):
            byte_t = byte_ids[:, t]  # (B,)

            # Skip padding
            is_pad = (byte_t == BYTE_PAD)
            if is_pad.all():
                break

            # Read byte → update accumulator
            logits, trigger, acc_state = self.cell(byte_t, acc_state)
            token_logits_list.append(logits)

            # Trigger decision
            trigger = trigger + self.trigger_bias  # apply bias
            triggered = trigger > 0.0  # (B,) hard decision for now
            trigger_logs.append(trigger)

            # When triggered, emit the predicted token and reset state for this sample
            # For now: emit at every byte (simplest case = byte-level tokens)
            # Later: learn when to trigger based on token boundaries

        # Simplest case: emit one token per byte (byte-level tokenization)
        # This gives us a working baseline before learning trigger boundaries
        token_logits = torch.stack(token_logits_list, dim=1)  # (B, T, V)
        emitted_tokens = token_logits.argmax(dim=-1)  # (B, T)

        # Zero out padding positions
        pad_mask = (byte_ids == BYTE_PAD)
        emitted_tokens = emitted_tokens.masked_fill(pad_mask, WORLD_PAD_ID)

        info = {
            'trigger_rates': torch.stack(trigger_logs).mean().item(),
            'token_logits': token_logits,
        }

        return emitted_tokens, info


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
