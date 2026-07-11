"""
Model architectures for the hybrid latent-state language model.

Models:
  - BaselineTransformer: standard autoregressive transformer
  - LatentSSM: recurrent latent model with thinking loop
  - LatentSSMDecoder: SSM + FFN decoder (cheap token generation)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================
# Baseline Transformer
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BaselineTransformer(nn.Module):
    """
    Standard autoregressive transformer baseline.

    Architecture:
      tokens -> embedding + positional encoding -> transformer layers -> logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: [batch, seq_len] token IDs
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        mask = self._generate_square_subsequent_mask(src.size(1), src.device)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        logits = self.output_proj(x)
        return logits

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


# ============================================================
# SSM Layer (simplified Mamba-style)
# ============================================================

class SSMLayer(nn.Module):
    """
    Simplified State Space Model layer.

    Uses a recurrent update: h_t = A * h_{t-1} + B * x_t
    where A and B are learned projections.

    This is a simplified version — a full Mamba/S4 would use
    structured convolutions and selective scan.
    """

    def __init__(self, d_state: int, d_input: int):
        super().__init__()
        self.d_state = d_state
        self.d_input = d_input

        # State transition: h_t = A * h_{t-1} + B * x_t
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.1)
        self.B = nn.Linear(d_input, d_state)

        # Output projection
        self.C = nn.Linear(d_state, d_state)

        # Nonlinearity
        self.act = nn.SiLU()

        # Normalization
        self.norm = nn.LayerNorm(d_state)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_input] input
            h: [batch, d_state] initial state (optional)
        Returns:
            output: [batch, seq_len, d_state]
            h_final: [batch, d_state] final state
        """
        batch, seq_len, _ = x.shape
        if h is None:
            h = torch.zeros(batch, self.d_state, device=x.device)

        outputs = []
        for t in range(seq_len):
            # SSM update: h_t = act(A @ h_{t-1} + B(x_t))
            h = torch.einsum('ij,bj->bi', self.A, h) + self.B(x[:, t, :])
            h = self.act(h)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, d_state]
        outputs = self.norm(outputs)
        return outputs, h

    def step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Single recurrent step for latent thinking loop."""
        h = torch.einsum('ij,bj->bi', self.A, h) + self.B(x)
        h = self.act(h)
        h = self.norm(h.unsqueeze(1)).squeeze(1)
        return h


# ============================================================
# Latent SSM Model
# ============================================================

class LatentSSM(nn.Module):
    """
    Recurrent latent model with thinking loop.

    Architecture:
      tokens -> embedding -> encode to latent state
      latent_state = SSM_step(latent_state) × N  (thinking)
      latent_state -> decoder -> logits

    Two modes:
      - Training: encode all tokens, think, decode
      - Generation: state-only thinking, then cheap token decode
    """

    def __init__(
        self,
        vocab_size: int,
        d_state: int = 256,
        d_model: int = 256,
        num_ssm_layers: int = 2,
        latent_steps: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_state = d_state
        self.latent_steps = latent_steps

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.state_proj = nn.Linear(d_model, d_state)

        # Stack of SSM layers
        self.ssm_layers = nn.ModuleList([
            SSMLayer(d_state, d_state) for _ in range(num_ssm_layers)
        ])

        # FFN for latent thinking
        self.ffn = nn.Sequential(
            nn.Linear(d_state, d_state * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_state * 2, d_state),
        )

        # Token decoder (cheap readout)
        self.token_decoder = nn.Linear(d_state, vocab_size)

        # Residual gating
        self.gate = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.Sigmoid(),
        )

    def encode(self, src: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input tokens into latent state."""
        x = self.embedding(src)  # [batch, seq, d_model]
        # Pool sequence into state
        x = x.mean(dim=1)  # [batch, d_model]
        h = self.state_proj(x)  # [batch, d_state]
        return h

    def think(self, h: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """
        Run latent thinking steps without emitting tokens.

        Args:
            h: [batch, d_state] current state
            steps: number of thinking steps (default: self.latent_steps)
        Returns:
            h: [batch, d_state] updated state
        """
        steps = steps or self.latent_steps
        for _ in range(steps):
            h_new = h
            for ssm in self.ssm_layers:
                h_new = ssm.step(h_new, h_new)
            h_new = self.ffn(h_new)
            # Gated residual
            gate = self.gate(h)
            h = gate * h_new + (1 - gate) * h
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode latent state to token logits (cheap readout)."""
        logits = self.token_decoder(h)  # [batch, vocab_size]
        return logits

    def forward(
        self,
        src: torch.Tensor,
        latent_steps: Optional[int] = None,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> think -> decode.

        Args:
            src: [batch, seq_len] token IDs
            latent_steps: override number of thinking steps
            h: initial state (optional, for continuation)
        Returns:
            logits: [batch, vocab_size]
            h: [batch, d_state] final state
        """
        if h is None:
            h = self.encode(src)
        h = self.think(h, steps=latent_steps)
        logits = self.decode(h)
        return logits, h


# ============================================================
# Latent SSM + Decoder (cheap multi-token generation)
# ============================================================

class LatentSSMDecoder(nn.Module):
    """
    SSM + FFN decoder for multi-token generation.

    Key difference from LatentSSM:
    - Decoder is a separate small network (cheap)
    - Can generate multiple tokens per state update
    - State evolves slowly, tokens generated quickly
    """

    def __init__(
        self,
        vocab_size: int,
        d_state: int = 256,
        d_model: int = 256,
        num_ssm_layers: int = 2,
        latent_steps: int = 4,
        tokens_per_step: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_state = d_state
        self.latent_steps = latent_steps
        self.tokens_per_step = tokens_per_step

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.state_proj = nn.Linear(d_model, d_state)

        # SSM layers
        self.ssm_layers = nn.ModuleList([
            SSMLayer(d_state, d_state) for _ in range(num_ssm_layers)
        ])

        # FFN for latent thinking
        self.ffn = nn.Sequential(
            nn.Linear(d_state, d_state * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_state * 2, d_state),
        )

        # Token decoder with position encoding
        self.decoder_ffn = nn.Sequential(
            nn.Linear(d_state + 32, d_state),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_state, vocab_size),
        )
        self.pos_embed = nn.Parameter(torch.randn(tokens_per_step, 32))

        # Gating
        self.gate = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.Sigmoid(),
        )

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.embedding(src)
        x = x.mean(dim=1)
        return self.state_proj(x)

    def think(self, h: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        steps = steps or self.latent_steps
        for _ in range(steps):
            h_new = h
            for ssm in self.ssm_layers:
                h_new = ssm.step(h_new, h_new)
            h_new = self.ffn(h_new)
            gate = self.gate(h)
            h = gate * h_new + (1 - gate) * h
        return h

    def decode_tokens(self, h: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """
        Generate multiple tokens from a single state.

        Args:
            h: [batch, d_state] state
            n_tokens: number of tokens to generate
        Returns:
            logits: [batch, n_tokens, vocab_size]
        """
        n_tokens = min(n_tokens, self.tokens_per_step)
        batch = h.size(0)

        # Expand state with position embeddings
        h_expanded = h.unsqueeze(1).expand(-1, n_tokens, -1)  # [batch, n_tokens, d_state]
        pos = self.pos_embed[:n_tokens].unsqueeze(0).expand(batch, -1, -1)  # [batch, n_tokens, 32]

        x = torch.cat([h_expanded, pos], dim=-1)  # [batch, n_tokens, d_state + 32]
        logits = self.decoder_ffn(x)  # [batch, n_tokens, vocab_size]
        return logits

    def forward(
        self,
        src: torch.Tensor,
        latent_steps: Optional[int] = None,
        n_tokens: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward: encode -> think -> decode multiple tokens.

        Args:
            src: [batch, seq_len] token IDs
            latent_steps: number of thinking steps
            n_tokens: tokens to generate per state update
        Returns:
            logits: [batch, n_tokens, vocab_size]
            h: [batch, d_state] final state
        """
        h = self.encode(src)
        h = self.think(h, steps=latent_steps)
        logits = self.decode_tokens(h, n_tokens)
        return logits, h
