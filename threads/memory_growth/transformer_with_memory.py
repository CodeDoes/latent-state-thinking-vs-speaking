"""Transformer model with Delta-Rule and RWKV-State Memory Adapters.

Enables matched-parameter single-variable ablations of associative memory
inside standard Transformer blocks on character/byte sequences.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from threads.delta_mem.delta_mem import DeltaRuleStateMemory, RWKVStateMemory


class CausalSelfAttentionWithMemory(nn.Module):
    """Causal Self-Attention with Delta-Rule/RWKV-State Memory Adapters."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        rwkv_mem_enabled: bool = False,
        rwkv_mem_mode: str = "delta_rule",
        rwkv_mem_rank: int = 8,
        rwkv_mem_num_state_heads: int = 1,
        rwkv_mem_alpha: float = 16.0,
        rwkv_mem_beta_bias_init: float = -1.5,
        rwkv_mem_output_init: str = "zero",
        rwkv_mem_output_init_scale: float = 0.02,
        rwkv_mem_delta_heads: Sequence[str] = ("q", "k", "v", "o"),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Standard projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Optional Memory Adapter
        self.rwkv_mem_enabled = rwkv_mem_enabled
        self.rwkv_mem_mode = rwkv_mem_mode
        self.rwkv_mem_delta_heads = tuple(rwkv_mem_delta_heads)

        self.rwkv_mem = None
        if rwkv_mem_enabled:
            if rwkv_mem_mode == "delta_rule":
                self.rwkv_mem = DeltaRuleStateMemory(
                    hidden_size=hidden_size,
                    query_size=hidden_size,
                    key_size=hidden_size,
                    value_size=hidden_size,
                    output_size=hidden_size,
                    rank=rwkv_mem_rank,
                    num_state_heads=rwkv_mem_num_state_heads,
                    alpha=rwkv_mem_alpha,
                    beta_bias_init=rwkv_mem_beta_bias_init,
                    delta_heads=rwkv_mem_delta_heads,
                    output_init=rwkv_mem_output_init,
                    output_init_scale=rwkv_mem_output_init_scale,
                )
            elif rwkv_mem_mode == "rwkv7":
                self.rwkv_mem = RWKVStateMemory(
                    hidden_size=hidden_size,
                    query_size=hidden_size,
                    key_size=hidden_size,
                    value_size=hidden_size,
                    output_size=hidden_size,
                    delta_heads=rwkv_mem_delta_heads,
                    output_init=rwkv_mem_output_init,
                    output_init_scale=rwkv_mem_output_init_scale,
                )
            else:
                raise ValueError(f"Unknown rwkv_mem_mode: {rwkv_mem_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape

        # Standard base projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Inject online memory deltas if enabled
        o_delta = None
        if self.rwkv_mem_enabled and self.rwkv_mem is not None:
            deltas = self.rwkv_mem(x)
            if "q" in deltas:
                q = q + deltas["q"]
            if "k" in deltas:
                k = k + deltas["k"]
            if "v" in deltas:
                v = v + deltas["v"]
            if "o" in deltas:
                o_delta = deltas["o"]

        # Multi-head attention split
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T, hd]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T, hd]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T, hd]

        # Scaled dot product
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, h, T, T]

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        # Softmax & output
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)  # [B, h, T, hd]

        # Concatenate heads and project out
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(attn_out)

        # Inject output delta
        if o_delta is not None:
            out = out + o_delta

        return out


class CausalTransformerBlockWithMemory(nn.Module):
    """Transformer block wrapping custom CausalSelfAttentionWithMemory and MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        hidden_scale: int = 4,
        rwkv_mem_enabled: bool = False,
        rwkv_mem_mode: str = "delta_rule",
        rwkv_mem_rank: int = 8,
        rwkv_mem_num_state_heads: int = 1,
        rwkv_mem_alpha: float = 16.0,
        rwkv_mem_beta_bias_init: float = -1.5,
        rwkv_mem_output_init: str = "zero",
        rwkv_mem_output_init_scale: float = 0.02,
        rwkv_mem_delta_heads: Sequence[str] = ("q", "k", "v", "o"),
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttentionWithMemory(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rwkv_mem_enabled=rwkv_mem_enabled,
            rwkv_mem_mode=rwkv_mem_mode,
            rwkv_mem_rank=rwkv_mem_rank,
            rwkv_mem_num_state_heads=rwkv_mem_num_state_heads,
            rwkv_mem_alpha=rwkv_mem_alpha,
            rwkv_mem_beta_bias_init=rwkv_mem_beta_bias_init,
            rwkv_mem_output_init=rwkv_mem_output_init,
            rwkv_mem_output_init_scale=rwkv_mem_output_init_scale,
            rwkv_mem_delta_heads=rwkv_mem_delta_heads,
        )

        self.ln2 = nn.LayerNorm(hidden_size)
        hidden_dim = hidden_size * hidden_scale
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual blocks
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerWithMemory(nn.Module):
    """Full Decoder-Only Transformer utilizing CausalTransformerBlockWithMemory."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        hidden_scale: int = 4,
        pad_token_id: int = 0,
        rwkv_mem_enabled: bool = False,
        rwkv_mem_mode: str = "delta_rule",
        rwkv_mem_rank: int = 8,
        rwkv_mem_num_state_heads: int = 1,
        rwkv_mem_alpha: float = 16.0,
        rwkv_mem_beta_bias_init: float = -1.5,
        rwkv_mem_output_init: str = "zero",
        rwkv_mem_output_init_scale: float = 0.02,
        rwkv_mem_delta_heads: Sequence[str] = ("q", "k", "v", "o"),
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([
            CausalTransformerBlockWithMemory(
                hidden_size=dim,
                num_heads=num_heads,
                hidden_scale=hidden_scale,
                rwkv_mem_enabled=rwkv_mem_enabled,
                rwkv_mem_mode=rwkv_mem_mode,
                rwkv_mem_rank=rwkv_mem_rank,
                rwkv_mem_num_state_heads=rwkv_mem_num_state_heads,
                rwkv_mem_alpha=rwkv_mem_alpha,
                rwkv_mem_beta_bias_init=rwkv_mem_beta_bias_init,
                rwkv_mem_output_init=rwkv_mem_output_init,
                rwkv_mem_output_init_scale=rwkv_mem_output_init_scale,
                rwkv_mem_delta_heads=rwkv_mem_delta_heads,
            )
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    nn.init.constant_(m.weight[m.padding_idx], 0.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        logits = self.head(x)
        return logits
