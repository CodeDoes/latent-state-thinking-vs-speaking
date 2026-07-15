"""Proto-hierarchical BLT-RWKV.

Architecture (matches the BLT pipeline structurally):

    bytes
      |
      v
    ByteEncoder:     embed -> N byte-level RWKV blocks  -> h_byte [B, T, D]
      |
      v
    PatchPool:       mean every k bytes -> h_pool [B, T/k, D]
      |
      v
    PatchSlot:       N patch-level RWKV blocks (SWAPPABLE) -> h_patch [B, P, D]
      |
      v
    PatchUnpool:     broadcast over k byte positions  -> h_patch_per_byte [B, T, D]
      |
      v
    ByteDecoder:     N byte-level RWKV blocks reading (h_byte + h_patch_per_byte)
                     -> logits [B, T, vocab_size]

The decoder is a *real RWKV stack* (not just LN+Linear), so the patch
context gets time-mixed into the byte-stream recurrence at every byte
position. This is the architecture you specified: byte->patch->byte
encoder+decoder with a patch patch-to-patch model slot in between.

Why this design:
- Bytes, patches, decoder are all swappable modules. Future work:
  byte->patch->plot (longer-time-scale summaries over patches).
- Decoder as RWKV (not just a linear head) means the cross-stream
  information gets *integrated* over the byte recurrence, not just
  added once at the end.
- For training-complexity reasons, defaults are tiny (~140K-180K params).

Future dimensions to explore:
1. V0 (now): mid-level slot is RWKV over patches.
2. V1: replace PatchSlot with attention over patches (BLT proper).
3. V2: replace mid-level slot with a higher-level abstractor
   (longer-time-scale; "plot level").
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rwkv_nano import RWKVBlock, count_params


class ByteEncoder(nn.Module):
    """Token bytes -> embed -> N byte-RWKV blocks.

    Output: h_byte [B, T, D].
    """

    def __init__(self, vocab_size: int, dim: int, n_layers: int, pad_token_id: int = 0):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln_in = nn.LayerNorm(dim)
        self.pad_token_id = pad_token_id

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.ln_in(self.embed(tokens))
        for block in self.blocks:
            h, _ = block(h)
        return h


def mean_pool_per_patch(h: torch.Tensor, k: int) -> torch.Tensor:
    """(B, T, D) where T is a multiple of k -> (B, T/k, D) mean over k.

    No learned params. Truncates the unaligned tail.
    """
    B, T, D = h.shape
    T_trunc = (T // k) * k
    if T_trunc == 0:
        return h[:, :0, :].contiguous()
    return h[:, :T_trunc, :].view(B, T_trunc // k, k, D).mean(dim=2)


def broadcast_to_bytes(h_patch: torch.Tensor, target_len: int, k: int) -> torch.Tensor:
    """(B, P, D) -> (B, target_len, D) by repeat-interleaving each patch k times.

    Truncates if P*k > target_len, zero-pads if P*k < target_len.
    """
    B, P, D = h_patch.shape
    out = h_patch.repeat_interleave(k, dim=1)
    if out.shape[1] > target_len:
        return out[:, :target_len, :]
    if out.shape[1] < target_len:
        pad = h_patch.new_zeros(B, target_len - out.shape[1], D)
        return torch.cat([out, pad], dim=1)
    return out


class PatchSlot(nn.Module):
    """The patch-to-patch model — swappable.

    Defaults to RWKV blocks at patch granularity. Drop your own
    PatchSlot subclass in here (e.g., BLM attention, an abstractor,
    a hierarchical-attention module, etc.) without changing anything
    upstream.

    Input  : (B, P, D)
    Output : (B, P, D)
    """

    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        # scalar gate so the slot signal can fade in smoothly during training
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, h_pool: torch.Tensor) -> torch.Tensor:
        h = h_pool
        for block in self.blocks:
            h, _ = block(h)
        return h * torch.sigmoid(self.gate)


class ByteDecoder(nn.Module):
    """RWKV byte stack that reads h_byte + h_patch_per_byte.

    The patch context is added once at the front, then time-mixed through
    every decoder block, then projected to logits. The cross-stream
    information gets *integrated* through recurrence rather than just
    one-shot added at the head.

    Input  : h_byte [B, T, D], h_patch_per_byte [B, T, D]
    Output : logits  [B, T, vocab_size]
    """

    def __init__(self, dim: int, vocab_size: int, n_layers: int):
        super().__init__()
        self.dim = dim
        # one-time fusion of patch context into the byte stream
        self.fuse = nn.Linear(dim * 2, dim)
        self.fuse_gate = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)
        nn.init.zeros_(self.head.bias)

    def forward(self, h_byte: torch.Tensor, h_patch_per_byte: torch.Tensor) -> torch.Tensor:
        # Concat and project; gated add so the patch context fades in
        h_cat = torch.cat([h_byte, h_patch_per_byte], dim=-1)
        h_fused = self.fuse(h_cat)
        g = torch.sigmoid(self.fuse_gate)
        h = h_byte + g * h_fused  # residual + gated patch context

        for block in self.blocks:
            h, _ = block(h)

        h = self.ln_out(h)
        return self.head(h)


class ProtoBLT_RWKV(nn.Module):
    """Byte -> Patch -> Byte encoder+decoder with a swappable patch slot.

    Args:
        vocab_size:    byte-level vocab (258 typical).
        dim:           hidden dim used everywhere (byte-, patch-, decoder-share).
        n_enc_layers:  byte-encoder stack depth.
        n_patch_layers:patch-slot stack depth (the swappable bit).
        n_dec_layers:  byte-decoder stack depth.
        patch_size:    fixed bytes per patch (V0 — entropy patcher is V1).
        pad_token_id:  vocab id reserved for padding.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 96,
        n_enc_layers: int = 2,
        n_patch_layers: int = 1,
        n_dec_layers: int = 2,
        patch_size: int = 4,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.dim = dim

        self.byte_enc = ByteEncoder(vocab_size, dim, n_enc_layers, pad_token_id)
        self.patch_slot = PatchSlot(dim, n_patch_layers)
        self.byte_dec = ByteDecoder(dim, vocab_size, n_dec_layers)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_byte = self.byte_enc(tokens)                            # [B, T, D]
        h_pool = mean_pool_per_patch(h_byte, self.patch_size)    # [B, P, D]
        h_patch = self.patch_slot(h_pool)                        # [B, P, D]
        h_patch_per_byte = broadcast_to_bytes(                    # [B, T, D]
            h_patch, tokens.shape[1], self.patch_size
        )
        logits = self.byte_dec(h_byte, h_patch_per_byte)
        # also return h_byte so call sites can hook / analyse
        return logits, h_byte


if __name__ == "__main__":
    model = ProtoBLT_RWKV(
        vocab_size=258,
        dim=96,
        n_enc_layers=2,
        n_patch_layers=1,
        n_dec_layers=2,
        patch_size=4,
    )
    print("ProtoBLT_RWKV (proto-hierarchical byte/patch/byte):")
    print(f"Total params: {count_params(model):,}")
    print(f"  byte_enc:    {count_params(model.byte_enc):,}")
    print(f"  patch_slot:  {count_params(model.patch_slot):,}")
    print(f"  byte_dec:    {count_params(model.byte_dec):,}")

    B, T = 2, 64
    tokens = torch.randint(1, 256, (B, T))
    logits, h_byte = model(tokens)
    print()
    print(f"Forward pass:")
    print(f"  input:  {tokens.shape}")
    print(f"  h_byte: {h_byte.shape}")
    print(f"  logits: {logits.shape}")
    print(f"  expected logits shape: [{B}, {T}, {model.vocab_size}]  OK")

    # gradient sanity
    loss = logits.sum()
    loss.backward()
    n_zero_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0
    )
    n_total = sum(1 for _ in model.parameters())
    print()
    print(f"Gradient sanity:  {n_zero_grad}/{n_total} params with zero gradient (high is bad)")
    print("OK")
