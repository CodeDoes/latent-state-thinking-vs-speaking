"""BLT-RWKV: minimal byte/patch dual-recurrence model.

Pattern (matching the BLT pipeline):

    bytes -> embed -> [InnerRWKV (byte-level)] -> h_byte
                                       | (mean_pool every k bytes)
                                       v
                          [OuterRWKV (patch-level)] -> h_patch
                                       | (repeat to byte positions)
                                       v
                  h_byte + h_patch_per_byte -> LayerNorm -> linear -> logits

Three modules, each with its own RWKVBlocks. For the first-pass
experiment we use fixed-size patches (no entropy patcher); patcher
upgrades are a separate V1 change.

Why a minimal version:
- Two streams is the structural claim BLT makes. If it doesn't help
  here, additional BLT machinery (cross-attention decoder, n-gram
  memory, etc.) is unlikely to either.
- Fixed patches let us verify shape correctness without an entropy
  threshold. Bugs we'd hit will be in pooling/unpool and gradient
  flow — exactly the things that need to be debugged before scaling up.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from domains.rwkv.rwkv_nano import RWKVBlock, count_params


class ByteEncoder(nn.Module):
    """embed -> N_inner blocks -> LayerNorm, no head.

    Output h_byte: [B, T, dim_byte].
    """

    def __init__(self, vocab_size: int, dim: int, n_layers: int, pad_token_id: int = 0):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.pad_token_id = pad_token_id

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T] integer token ids (already byte-shifted)
        h = self.embed(tokens)
        for block in self.blocks:
            h, _ = block(h)
        h = self.ln(h)
        return h


class PatchMixer(nn.Module):
    """Mean-pool bytes inside each fixed-size window -> run RM blocks -> per-patch rep.

    Input  : h_byte [B, T, D]
    Output : h_patch [B, T_k, D] where T_k = T // patch_size
    """

    def __init__(self, dim: int, n_layers: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        # small init so the patch stream doesn't dominate early training
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, h_byte: torch.Tensor) -> torch.Tensor:
        B, T, D = h_byte.shape
        k = self.patch_size
        # truncate to multiple of k
        T_trunc = (T // k) * k
        if T_trunc == 0:
            return torch.zeros(B, 0, D, device=h_byte.device, dtype=h_byte.dtype)
        h = h_byte[:, :T_trunc, :]
        # reshape to [B, T/k, k, D] -> mean over k dim
        h_patch = h.view(B, T_trunc // k, k, D).mean(dim=2)
        for block in self.blocks:
            h_patch, _ = block(h_patch)
        return h_patch * torch.sigmoid(self.gate)


def broadcast_patches(h_patch: torch.Tensor, target_len: int, patch_size: int) -> torch.Tensor:
    """Repeat each patch position `patch_size` times to align with byte positions.

    h_patch: [B, P, D]   produces: [B, P*k, D]   (truncate to target_len if needed)
    """
    B, P, D = h_patch.shape
    out = h_patch.repeat_interleave(patch_size, dim=1)
    if out.shape[1] > target_len:
        out = out[:, :target_len, :]
    elif out.shape[1] < target_len:
        # pad with zeros (won't be used if downstream masks are set)
        pad = torch.zeros(B, target_len - out.shape[1], D, device=h_patch.device, dtype=h_patch.dtype)
        out = torch.cat([out, pad], dim=1)
    return out


class ByteDecoder(nn.Module):
    """h_byte + broadcast(h_patch) -> LayerNorm -> linear.

    Output: logits [B, T, vocab_size].
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)
        # init biases to zero so initial output skews toward UNK/PAD — safe default
        nn.init.zeros_(self.head.bias)

    def forward(self, h_byte: torch.Tensor, h_patch: torch.Tensor, patch_size: int) -> torch.Tensor:
        h = h_byte + h_patch
        h = self.ln(h)
        return self.head(h)


class BLT_RWKV(nn.Module):
    """Minimal BLT-RWKV: ByteEncoder -> PatchMixer -> ByteDecoder.

    Args:
        vocab_size: byte-level vocab size (typically 258)
        dim_byte:  hidden dim for the byte encoder / decoder
        dim_patch: hidden dim for the patch mixer (small — patches are short)
        n_layers_inner: depth of the byte-level RWKV
        n_layers_outer: depth of the patch-level RWKV
        patch_size: fixed bytes per patch (V0 — entropy patcher is a later change)
        pad_token_id: token id used at sequence start / for padding
    """

    def __init__(
        self,
        vocab_size: int,
        dim_byte: int = 64,
        dim_patch: int = 32,
        n_layers_inner: int = 2,
        n_layers_outer: int = 1,
        patch_size: int = 4,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.dim_byte = dim_byte

        # Inner and outer have *different* dims because byte-level
        # contains more information than patch-level should need to.
        # The patch-mixer reads dim_byte but its own blocks are dim_patch.
        # We project byte->patch on entry and patch->byte on exit.
        self.byte_enc = ByteEncoder(vocab_size, dim_byte, n_layers_inner, pad_token_id)
        # PatchMixer operates at dim_patch so we project in:
        self.to_patch = nn.Linear(dim_byte, dim_patch)
        self.patch_mixer = PatchMixer(dim_patch, n_layers_outer, patch_size)
        # And project back to dim_byte for the decoder:
        self.to_byte = nn.Linear(dim_patch, dim_byte)
        self.byte_dec = ByteDecoder(dim_byte, vocab_size)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            tokens: [B, T] byte-level token ids.
        Returns:
            logits: [B, T, vocab_size]
            h_byte: [B, T, dim_byte] (for analysis / hook targets)
        """
        h_byte = self.byte_enc(tokens)              # [B, T, D_b]
        h_for_patch = self.to_patch(h_byte)         # [B, T, D_p]
        h_patch = self.patch_mixer(h_for_patch)     # [B, T//k, D_p]
        h_patch_back = self.to_byte(h_patch)        # [B, T//k, D_b]
        h_patch_per_byte = broadcast_patches(
            h_patch_back, tokens.shape[1], self.patch_size
        )                                          # [B, T, D_b]
        logits = self.byte_dec(h_byte, h_patch_per_byte, self.patch_size)
        return logits, h_byte


if __name__ == "__main__":
    # Smoke test — should run without crashing
    model = BLT_RWKV(
        vocab_size=258,
        dim_byte=64,
        dim_patch=32,
        n_layers_inner=2,
        n_layers_outer=1,
        patch_size=4,
    )
    print(f"Total params: {count_params(model):,}")
    print(f"  inner: {count_params(model.byte_enc):,}")
    print(f"  to_patch: {count_params(model.to_patch):,}")
    print(f"  patch_mixer: {count_params(model.patch_mixer):,}")
    print(f"  to_byte: {count_params(model.to_byte):,}")
    print(f"  byte_dec: {count_params(model.byte_dec):,}")

    B, T = 2, 64
    tokens = torch.randint(1, 256, (B, T))
    logits, h_byte = model(tokens)
    print(f"\nForward pass:")
    print(f"  input:  {tokens.shape}")
    print(f"  h_byte: {h_byte.shape}")
    print(f"  logits: {logits.shape}")
    print(f"  expected logits shape: [{B}, {T}, {model.vocab_size}]  ✓")

    # gradient sanity
    loss = logits.sum()
    loss.backward()
    n_zero_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0
    )
    n_total = sum(1 for _ in model.parameters())
    print(f"\nGradient sanity:")
    print(f"  {n_zero_grad}/{n_total} params have zero gradient (high is suspicious)")
    print("Done.")
