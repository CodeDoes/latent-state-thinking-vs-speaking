"""
Model architectures for the hybrid latent-state language model.

Models:
  - BaselineTransformer: standard autoregressive transformer
  - LatentSSM: recurrent latent model with thinking loop (causal LM)
  - LatentSSMDecoder: SSM + FFN decoder (cheap multi-token generation)

Key design: All models produce [batch, seq_len, vocab_size] output for
fair comparison on next-token prediction.

The latent models additionally perform N "thinking" steps in latent space
between processing input tokens, testing whether extra latent computation
improves reasoning over pure token-by-token processing.
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

    Output: [batch, seq_len, vocab_size]
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
    State Space Model layer with input-dependent dynamics (Mamba-style).

    Uses a recurrent update: h_t = A(x_t) * h_{t-1} + B(x_t) * x_t
    where A and B are functions of the input (selective mechanism).

    This allows the model to selectively remember/forget based on input.
    """

    def __init__(self, d_state: int, d_input: int, selective: bool = True):
        super().__init__()
        self.d_state = d_state
        self.d_input = d_input
        self.selective = selective

        # Base state transition matrix
        self.A_base = nn.Parameter(torch.randn(d_state, d_state) * 0.1)
        
        if selective:
            # Input-dependent modulation of A
            self.A_mod = nn.Linear(d_input, d_state * d_state, bias=False)
            # Input-dependent B
            self.B = nn.Linear(d_input, d_state, bias=False)
        else:
            # Simple fixed A and B
            self.A = self.A_base
            self.B = nn.Linear(d_input, d_state)

        # Output projection
        self.C = nn.Linear(d_state, d_state)

        # Nonlinearity
        self.act = nn.SiLU()

        # Normalization
        self.norm = nn.LayerNorm(d_state)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of inputs.

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
            h = self.step(x[:, t, :], h)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, d_state]
        return outputs, h

    def step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Single recurrent step with optional input-dependent dynamics.

        Args:
            x: [batch, d_input] input at current step
            h: [batch, d_state] current state
        Returns:
            h_new: [batch, d_state] updated state
        """
        if self.selective:
            # Compute input-dependent A: A(x) = A_base + A_mod(x)
            batch_size = x.size(0)
            A_delta = self.A_mod(x).view(batch_size, self.d_state, self.d_state)
            # Small modulation to keep stable
            A = self.A_base.unsqueeze(0) + 0.1 * torch.tanh(A_delta)
            # State update: h_new = A @ h + B(x)
            h = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1) + self.B(x)
        else:
            # Simple linear update
            h = torch.einsum('ij,bj->bi', self.A_base, h) + self.B(x)
        
        h = self.act(h)
        h = self.norm(h)
        return h


# ============================================================
# Latent SSM Model (Causal LM with thinking steps)
# ============================================================

class LatentSSM(nn.Module):
    """
    Recurrent latent model with thinking loop.

    Architecture:
      For each input token:
        1. Embed token
        2. Update SSM state
        3. Every `think_every` tokens: do N latent thinking steps
        4. Decode state -> logits for this position

    This produces [batch, seq_len, vocab_size] output, same as the baseline,
    enabling fair comparison. The latent model does extra computation
    (thinking steps) between token predictions.

    Output: [batch, seq_len, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        d_state: int = 256,
        d_model: int = 256,
        num_ssm_layers: int = 2,
        latent_steps: int = 4,
        think_every: int = 1,
        dropout: float = 0.1,
        selective: bool = True,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.latent_steps = latent_steps
        self.think_every = think_every

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Input projection (token embedding -> SSM input)
        self.input_proj = nn.Linear(d_model, d_state)

        # Stack of SSM layers
        self.ssm_layers = nn.ModuleList([
            SSMLayer(d_state, d_state, selective=selective) for _ in range(num_ssm_layers)
        ])

        # FFN for latent thinking
        self.ffn = nn.Sequential(
            nn.Linear(d_state, d_state * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_state * 2, d_state),
        )

        # Token decoder (from state to vocab)
        self.token_decoder = nn.Linear(d_state, vocab_size)

        # Residual gating for thinking steps
        self.gate = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.Sigmoid(),
        )

    def think(self, h: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """
        Run latent thinking steps without consuming input tokens.

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
                # Self-recurrence: use state as both input and state
                h_new = ssm.step(h_new, h_new)
            h_new = self.ffn(h_new)
            # Gated residual
            gate = self.gate(h)
            h = gate * h_new + (1 - gate) * h
        return h

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: process tokens sequentially with optional thinking.

        Args:
            src: [batch, seq_len] token IDs
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = src.shape
        device = src.device

        # Initialize state
        h = torch.zeros(batch, self.d_state, device=device)

        # Embed all tokens
        x = self.embedding(src)  # [batch, seq, d_model]
        x = self.input_proj(x)   # [batch, seq, d_state]

        all_logits = []

        for t in range(seq_len):
            # Update SSM state with current token
            for ssm in self.ssm_layers:
                h = ssm.step(x[:, t, :], h)

            # Periodic latent thinking
            if self.think_every > 0 and (t + 1) % self.think_every == 0:
                h = self.think(h)

            # Decode to logits
            logits_t = self.token_decoder(h)  # [batch, vocab_size]
            all_logits.append(logits_t)

        logits = torch.stack(all_logits, dim=1)  # [batch, seq_len, vocab_size]
        return logits


# ============================================================
# Latent SSM + Decoder (cheap multi-token generation)
# ============================================================

class LatentSSMDecoder(nn.Module):
    """
    SSM + FFN decoder for multi-token generation.

    Key difference from LatentSSM:
    - After thinking, can generate multiple tokens cheaply from same state
    - Decoder uses positional embeddings to generate tokens_per_step tokens
    - State evolves slowly, tokens generated quickly

    Training mode: produces [batch, seq_len, vocab_size] for fair comparison
    Generation mode: can produce multiple tokens per state update

    Output: [batch, seq_len, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        d_state: int = 256,
        d_model: int = 256,
        num_ssm_layers: int = 2,
        latent_steps: int = 4,
        tokens_per_step: int = 8,
        think_every: int = 4,
        dropout: float = 0.1,
        selective: bool = True,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.latent_steps = latent_steps
        self.tokens_per_step = tokens_per_step
        self.think_every = think_every

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.input_proj = nn.Linear(d_model, d_state)

        # SSM layers
        self.ssm_layers = nn.ModuleList([
            SSMLayer(d_state, d_state, selective=selective) for _ in range(num_ssm_layers)
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

    def think(self, h: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Run latent thinking steps."""
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

        h_expanded = h.unsqueeze(1).expand(-1, n_tokens, -1)
        pos = self.pos_embed[:n_tokens].unsqueeze(0).expand(batch, -1, -1)

        x = torch.cat([h_expanded, pos], dim=-1)
        logits = self.decoder_ffn(x)
        return logits

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass for training.

        Process tokens sequentially, periodically think, and decode
        multiple tokens per state. For fair comparison with baseline,
        produces [batch, seq_len, vocab_size].

        Args:
            src: [batch, seq_len] token IDs
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = src.shape
        device = src.device

        # Initialize state
        h = torch.zeros(batch, self.d_state, device=device)

        # Embed all tokens
        x = self.embedding(src)
        x = self.input_proj(x)

        all_logits = []
        token_idx = 0  # How many output tokens we've produced from current state

        for t in range(seq_len):
            # Update SSM state with current token
            for ssm in self.ssm_layers:
                h = ssm.step(x[:, t, :], h)

            # Periodic latent thinking
            if self.think_every > 0 and (t + 1) % self.think_every == 0:
                h = self.think(h)
                token_idx = 0  # Reset token counter after thinking

            # Decode from current state
            if token_idx < self.tokens_per_step:
                logits_t = self.decode_tokens(h, 1)  # [batch, 1, vocab]
                logits_t = logits_t.squeeze(1)  # [batch, vocab]
                token_idx += 1
            else:
                # Just decode directly without positional bias
                logits_t = self.decode_tokens(h, 1).squeeze(1)

            all_logits.append(logits_t)

        logits = torch.stack(all_logits, dim=1)  # [batch, seq_len, vocab_size]
        return logits
