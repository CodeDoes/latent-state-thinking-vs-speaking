"""BLT with RWKV byte encoder/decoder, larger transformer patch mixer.

Experiment: patch model 3x larger capacity + slower RWKV decay + 1000 steps.
Tests if the patch model was capacity-starved in previous experiments.

Architecture:
    bytes -> embed -> [RWKVBlock × N] -> h_byte
                                    | (mean_pool every k bytes)
                                    v
                    [CausalTransformerBlock × M] -> h_patch  (3x larger dim)
                                    | (repeat to byte positions)
                                    v
                h_byte + h_patch_per_byte -> LayerNorm -> linear -> logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from domains.rwkv.rwkv_nano import RWKVBlock, count_params
from threads.blt.blt_pure import CausalTransformerBlock, broadcast_patches


class ByteEncoder(nn.Module):
    """embed -> N RWKV blocks -> LayerNorm, no head."""

    def __init__(self, vocab_size: int, dim: int, n_layers: int, pad_token_id: int = 0):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.pad_token_id = pad_token_id

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for block in self.blocks:
            h, _ = block(h)
        h = self.ln(h)
        return h


class PatchMixer(nn.Module):
    """Mean-pool bytes -> larger transformer blocks -> per-patch rep."""

    def __init__(self, dim: int, n_layers: int, patch_size: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(dim, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, h_byte: torch.Tensor) -> torch.Tensor:
        B, T, D = h_byte.shape
        k = self.patch_size
        T_trunc = (T // k) * k
        if T_trunc == 0:
            return torch.zeros(B, 0, D, device=h_byte.device, dtype=h_byte.dtype)
        h = h_byte[:, :T_trunc, :]
        h_patch = h.view(B, T_trunc // k, k, D).mean(dim=2)
        for block in self.blocks:
            h_patch = block(h_patch)
        return h_patch * torch.sigmoid(self.gate)


class ByteDecoder(nn.Module):
    """h_byte + broadcast(h_patch) -> LayerNorm -> linear."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)
        nn.init.zeros_(self.head.bias)

    def forward(self, h_byte: torch.Tensor, h_patch: torch.Tensor, patch_size: int) -> torch.Tensor:
        h = h_byte + h_patch
        h = self.ln(h)
        return self.head(h)


class BLT_RWKV_LargePatch(nn.Module):
    """BLT with RWKV byte encoder/decoder, 3x larger transformer patch mixer.

    Args:
        vocab_size: byte-level vocab size (typically 258)
        dim_byte:  hidden dim for the byte encoder / decoder
        dim_patch: hidden dim for the patch mixer (3x dim_byte)
        n_layers_inner: depth of the byte-level RWKV
        n_layers_outer: depth of the patch-level transformer
        n_heads: number of attention heads (for patch transformer)
        patch_size: fixed bytes per patch
        pad_token_id: token id used at sequence start / for padding
    """

    def __init__(
        self,
        vocab_size: int,
        dim_byte: int = 64,
        dim_patch: int = 192,  # 3x dim_byte
        n_layers_inner: int = 2,
        n_layers_outer: int = 2,
        n_heads: int = 8,
        patch_size: int = 4,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.dim_byte = dim_byte

        self.byte_enc = ByteEncoder(vocab_size, dim_byte, n_layers_inner, pad_token_id)
        self.to_patch = nn.Linear(dim_byte, dim_patch)
        self.patch_mixer = PatchMixer(dim_patch, n_layers_outer, patch_size, n_heads)
        self.to_byte = nn.Linear(dim_patch, dim_byte)
        self.byte_dec = ByteDecoder(dim_byte, vocab_size)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_byte = self.byte_enc(tokens)
        h_for_patch = self.to_patch(h_byte)
        h_patch = self.patch_mixer(h_for_patch)
        h_patch_back = self.to_byte(h_patch)
        h_patch_per_byte = broadcast_patches(
            h_patch_back, tokens.shape[1], self.patch_size
        )
        logits = self.byte_dec(h_byte, h_patch_per_byte, self.patch_size)
        return logits, h_byte


if __name__ == "__main__":
    model = BLT_RWKV_LargePatch(
        vocab_size=258,
        dim_byte=64,
        dim_patch=192,
        n_layers_inner=2,
        n_layers_outer=2,
        n_heads=8,
        patch_size=4,
    )
    print(f"Total params: {count_params(model):,}")
    print(f"  byte_enc (RWKV): {count_params(model.byte_enc):,}")
    print(f"  to_patch: {count_params(model.to_patch):,}")
    print(f"  patch_mixer (transformer): {count_params(model.patch_mixer):,}")
    print(f"  to_byte: {count_params(model.to_byte):,}")
    print(f"  byte_dec: {count_params(model.byte_dec):,}")

    B, T = 2, 64
    tokens = torch.randint(1, 256, (B, T))
    logits, h_byte = model(tokens)
    print(f"\nForward pass:")
    print(f"  input:  {tokens.shape}")
    print(f"  h_byte: {h_byte.shape}")
    print(f"  logits: {logits.shape}")

    loss = logits.sum()
    loss.backward()
    n_zero_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0
    )
    n_total = sum(1 for _ in model.parameters())
    print(f"\nGradient sanity:")
    print(f"  {n_zero_grad}/{n_total} params have zero gradient")
    print("Done.")
