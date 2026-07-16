"""Byte-level adaptive encoder↔decoder loop.

BYTES → RNN-ENCODER → (next-byte | TRIGGER) → RNN-DECODER → (next-byte | TRIGGER) → BYTES

Encoder and decoder each decide per step: predict next byte, or emit TRIGGER
to hand off to the other. TRIGGER means "need more compute, you try".

Min length 8 before trigger likely; length 1 needs special reason.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TriggerHead(nn.Module):
    """Outputs: byte logits (258) + trigger logit (1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.byte_head = nn.Linear(dim, 258)
        self.trigger_head = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.byte_head(h), self.trigger_head(h).squeeze(-1)


class ByteRNN(nn.Module):
    """Simple RNN layer with receptance (like SimpleRNNReceptance)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W_in = nn.Linear(dim, dim, bias=False)
        self.W_rec = nn.Linear(dim, dim, bias=False)
        self.W_recept = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            state: [B, D] or None
        Returns:
            out: [B, T, D]
            h_final: [B, D]
        """
        B, T, D = x.shape
        if state is None:
            state = x.new_zeros(B, D)

        h = state
        outs = []
        for t in range(T):
            h_in = self.W_in(x[:, t])
            h_rec = self.W_rec(h)
            r = torch.sigmoid(self.W_recept(x[:, t]))
            h = r * h_in + (1 - r) * h_rec
            outs.append(h)
        h_final = h
        return torch.stack(outs, dim=1), h_final


class AdaptiveByteEncoder(nn.Module):
    """RNN encoder that emits (byte_pred, trigger_prob) at each step."""

    def __init__(self, dim: int = 64, n_layers: int = 2, min_len_before_trigger: int = 8):
        super().__init__()
        self.dim = dim
        self.min_len = min_len_before_trigger

        self.embed = nn.Embedding(258, dim, padding_idx=0)
        self.layers = nn.ModuleList([ByteRNN(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = TriggerHead(dim)

        # Bias trigger toward OFF for first min_len steps
        self.register_buffer("pos_bias", torch.zeros(1024))
        with torch.no_grad():
            self.pos_bias[:self.min_len] = -3.0  # suppress trigger early
            self.pos_bias[self.min_len:] = 0.0

    def forward(
        self,
        tokens: torch.Tensor,
        states: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict]:
        """
        Returns:
            byte_logits: [B, T, 258]
            trigger_logits: [B, T] (pre-sigmoid)
            final_states: list of [B, D]
            info: dict
        """
        B, T = tokens.shape
        x = self.embed(tokens)

        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        h = x
        for i, (layer, state) in enumerate(zip(self.layers, states)):
            h, h_final = layer(h, state)
            new_states.append(h_final)

        h = self.ln(h)
        byte_logits, trigger_logits = self.head(h)

        # Positional trigger bias
        seq_len = min(T, self.pos_bias.shape[0])
        trigger_logits[:, :seq_len] = trigger_logits[:, :seq_len] + self.pos_bias[:seq_len]

        return byte_logits, trigger_logits, new_states, {"trigger_prob": torch.sigmoid(trigger_logits).mean().item()}


class AdaptiveByteDecoder(nn.Module):
    """RNN decoder that takes encoder trigger + context, emits (byte_pred, trigger_prob)."""

    def __init__(self, dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.dim = dim

        self.trigger_embed = nn.Linear(1, dim)  # scalar trigger → dim
        self.layers = nn.ModuleList([ByteRNN(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = TriggerHead(dim)

    def forward(
        self,
        encoder_trigger: torch.Tensor,  # [B, T] trigger prob from encoder
        encoder_byte_logits: torch.Tensor,  # [B, T, 258] from encoder
        states: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict]:
        """
        Decoder input: encoder trigger (broadcast) + encoder byte logits (projected)
        """
        B, T = encoder_trigger.shape

        # Project encoder byte logits to dim (detach to avoid double-backprop through encoder head)
        byte_proj = nn.functional.linear(encoder_byte_logits.detach(), self.layers[0].W_in.weight[:258].T)

        # Trigger embedding
        trigger_emb = self.trigger_embed(encoder_trigger.unsqueeze(-1))

        x = byte_proj + trigger_emb

        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        h = x
        for i, (layer, state) in enumerate(zip(self.layers, states)):
            h, h_final = layer(h, state)
            new_states.append(h_final)

        h = self.ln(h)
        byte_logits, trigger_logits = self.head(h)

        return byte_logits, trigger_logits, new_states, {"trigger_prob": torch.sigmoid(trigger_logits).mean().item()}


class ByteLoopModel(nn.Module):
    """Full loop: Encoder → (byte|trigger) → Decoder → (byte|trigger) → ..."""

    def __init__(
        self,
        dim: int = 64,
        n_layers: int = 2,
        min_len_before_trigger: int = 8,
        max_loops: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.max_loops = max_loops

        self.encoder = AdaptiveByteEncoder(dim, n_layers, min_len_before_trigger)
        self.decoder = AdaptiveByteDecoder(dim, n_layers)

    def forward(
        self,
        tokens: torch.Tensor,
        enc_states: Optional[list[torch.Tensor]] = None,
        dec_states: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            final_byte_logits: [B, T, 258] — combined prediction
            info: dict with loop stats
        """
        B, T = tokens.shape

        # Initial encoder pass
        enc_byte_logits, enc_trigger_logits, enc_states, enc_info = self.encoder(tokens, enc_states)

        # Loop: decoder gets encoder trigger, produces its own trigger back to encoder
        # For now: single encoder→decoder pass, combine predictions weighted by (1-trigger)
        # Full loop would iterate encoder(decoder_trigger) → decoder(encoder_trigger) ...

        dec_byte_logits, dec_trigger_logits, dec_states, dec_info = self.decoder(
            torch.sigmoid(enc_trigger_logits), enc_byte_logits, dec_states
        )

        # Combine: where encoder says "not trigger", use encoder; where decoder says "not trigger", use decoder
        enc_weight = 1 - torch.sigmoid(enc_trigger_logits).unsqueeze(-1)  # [B, T, 1]
        dec_weight = 1 - torch.sigmoid(dec_trigger_logits).unsqueeze(-1)

        # Normalize
        total = enc_weight + dec_weight + 1e-7
        enc_weight = enc_weight / total
        dec_weight = dec_weight / total

        final_logits = enc_weight * enc_byte_logits + dec_weight * dec_byte_logits

        info = {
            "encoder": enc_info,
            "decoder": dec_info,
            "enc_trigger_rate": torch.sigmoid(enc_trigger_logits).mean().item(),
            "dec_trigger_rate": torch.sigmoid(dec_trigger_logits).mean().item(),
            "states": {"enc": enc_states, "dec": dec_states},
        }

        return final_logits, info


# ── Smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T = 2, 64
    tokens = torch.randint(1, 256, (B, T))

    model = ByteLoopModel(dim=64, n_layers=2, min_len_before_trigger=8, max_loops=4)

    logits, info = model(tokens)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    print(f"  encoder:  {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  decoder:  {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"  logits: {logits.shape}")
    print(f"  enc trigger: {info['enc_trigger_rate']:.3f}")
    print(f"  dec trigger: {info['dec_trigger_rate']:.3f}")

    # Gradient check
    loss = F.cross_entropy(logits.view(-1, 258), torch.randint(1, 256, (B * T,)))
    loss.backward()
    n_zero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0)
    print(f"Gradient: {n_zero}/{sum(1 for _ in model.parameters())} zero-grad")
    print("OK" if n_zero == 0 else "ISSUE")