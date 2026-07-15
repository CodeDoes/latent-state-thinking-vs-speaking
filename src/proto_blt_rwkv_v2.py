"""ProtoBLT_RWKV v2: surprise-patcher + accumulated-state token routing.

Implements three architectural changes vs the V0 fixed-window
implementation in src/proto_blt_rwkv.py:

1. **Per-block time_decay init toward horizons**.
   Trainable decay is initialised toward `log(K)` per block where K is
   the *desired* receptive-field horizon in bytes. Without this the
   optimizer collapsed all 640 trained channels to w≈0.34 (~1-byte
   memory). At w=0.34, every channel behaves as a sliding window of
   size ~3 bytes — the architecture's "many parallel timescales" promise
   was unreachable. Targeted init biases toward a broader distribution.

2. **Surprise-driven variable patches**.
   Patch boundaries are placed where the byte-RWKV's per-step
   state-delta exceeds the surprise threshold, NOT every k bytes
   uniformly. Bytes whose state-delta is below threshold get folded
   into the next promotion's accumulated state. Promoted bytes form
   variable-length patches.

3. **Two-token patch inputs**.
   At each patch boundary we pass two vectors into the patch-RWKV:
     - `h_pool`   = mean of byte hidden states inside the patch (local)
     - `h_acc`    = the byte-RWKV's running state at boundary (context)
   The patch-RWKV treats these as a 2-token sequence per patch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rwkv_nano import RWKVBlock, count_params


# ──────────────────────────────────────────────────────────────────────────────────
# 1. Time-decay init to specific horizons
# ──────────────────────────────────────────────────────────────────────────────────

def init_block_time_decay(blocks: nn.ModuleList, horizon_bytes: int,
                          per_channel_noise: float = 0.15) -> None:
    """Initialise `time_decay` raw params toward `log(K)` so the effective
    receptive-field horizon is ≈ K bytes.

    Derivation: w = exp(-exp(raw)). Horizon H satisfies w^H ≈ 1/e ⇒ H = -1/ln(w).
    For raw = log(K):  w = exp(-K^(-1)) ⇒ H = -log(K) which is wrong.

    Correct derivation: we WANT H = K ⇒ w = exp(-1/K) ⇒ raw = log(1/K) = -log(K).
    So the raw param initialisation is `raw_init = -log(K)`.

    Per-channel noise breaks symmetry so different channels can specialise.
    Stddev 0.15 in raw space spread decays ~0.05 each side of the target.
    """
    raw_init = -math.log(horizon_bytes)
    for block in blocks:
        noise = torch.randn_like(block.time_decay) * per_channel_noise
        block.time_decay.data.copy_(raw_init + noise)


# ──────────────────────────────────────────────────────────────────────────────────
# 2. Byte encoder + surprise extraction
# ──────────────────────────────────────────────────────────────────────────────────

class ByteEncoder(nn.Module):
    """Token bytes -> embed -> N byte-RWKV blocks.

    Returns:
       h_byte:    [B, T, D]    per-byte output states
       surprise:  [B, T-1]     per-position state-delta surprise
    """

    def __init__(self, vocab_size: int, dim: int, n_layers: int,
                 pad_token_id: int = 0):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln_in = nn.LayerNorm(dim)
        self.pad_token_id = pad_token_id

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.ln_in(self.embed(tokens))  # [B, T, D]

        # We accumulate state-deltas through the *first* layer's per-step
        # output. The deeper layers' outputs also encode rhythm but the
        # first one is most direct: it sees raw embeddings.
        per_step: list[torch.Tensor] = []
        prev = None
        for t in range(h.shape[1]):
            h_t, _ = self.blocks[0](h[:, t:t+1, :])
            if prev is not None:
                per_step.append((h_t.squeeze(1) - prev).abs().mean(dim=-1))  # [B]
            prev = h_t.squeeze(1)

        # Run deeper layers over the full sequence for final per-byte output
        for layer in self.blocks[1:]:
            h, _ = layer(h)

        surprise = None
        if per_step:
            surprise = torch.stack(per_step, dim=1)  # [B, T-1]
        return h, surprise


# ──────────────────────────────────────────────────────────────────────────────────
# 3. Surprise → variable patch boundaries
# ──────────────────────────────────────────────────────────────────────────────────

def surprise_to_patch_lengths(surprise: torch.Tensor,
                                threshold: float,
                                min_patch: int = 1,
                                max_patch: int | None = None) -> torch.Tensor:
    """Convert a per-byte surprise curve into patch_lengths.

    Args:
        surprise:       [B, T-1]
        threshold:      surprise > threshold  ⇒  new patch starts here
        min_patch:      smallest number of bytes per patch (>=1)
        max_patch:      hard cap on each patch

    Returns:
        patch_lengths: [B, max_patches_observed]  long.  Each row's sum = T.
    """
    B, T_minus_one = surprise.shape
    boundaries_list: list[list[int]] = []
    for b in range(B):
        bs = [0]
        for i, val in enumerate(surprise[b].tolist(), start=1):
            if val > threshold:
                bs.append(i)
        if bs[-1] != T_minus_one + 1:
            bs.append(T_minus_one + 1)
        # Convert to lengths
        lens = [bs[i+1] - bs[i] for i in range(len(bs) - 1)]
        # Enforce min_patch
        merged: list[int] = []
        cur = 0
        for L in lens:
            cur += L
            if cur >= min_patch or not merged:
                merged.append(cur)
                cur = 0
        if cur > 0:
            # leftover bytes get folded into the last patch
            if merged:
                merged[-1] += cur
            else:
                merged.append(cur)
        # Cap patch size
        if max_patch is not None:
            fixed: list[int] = []
            for L in merged:
                while L > max_patch:
                    fixed.append(max_patch)
                    L -= max_patch
                if L > 0:
                    fixed.append(L)
            merged = fixed
        boundaries_list.append(merged)

    P = max(len(p) for p in boundaries_list)
    out = torch.zeros(B, P, dtype=torch.long)
    for b in range(B):
        for i, v in enumerate(boundaries_list[b]):
            out[b, i] = v
    return out


# ──────────────────────────────────────────────────────────────────────────────────
# 4. Two-token patch gather
# ──────────────────────────────────────────────────────────────────────────────────

def gather_patch_tokens(
    h_byte: torch.Tensor,    # [B, T, D]
    surprise: torch.Tensor,  # [B, T-1]
    threshold: float,
    min_patch: int = 1,
    max_patch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-token-per-patch: `(h_pool, h_acc)` for each patch.

    Returns:
        h_pool:  [B, P, D]   mean bytes per patch
        h_acc:   [B, P, D]   byte-RWKV 'accumulated state' at boundary —
                              captured as the mean of h_byte over the k
                              bytes immediately BEFORE the boundary (analog
                              of "context that led up to this patch").
        patch_lengths: [B, P]  long.  sum == T.

    Note on h_acc: we approximate the byte-RWKV's running state at the
    boundary by mean-pooling the last 3 (or k/2) bytes of h_byte just
    inside the patch. A true implementation would plumb the recurrent
    state vector through RWKVBlock.forward(return_state=True); this proxy
    is faithful enough for V2 because the WKV state materialises in the
    per-step output anyway.
    """
    B, T, D = h_byte.shape

    # Determine patches from surprise
    patch_lengths = surprise_to_patch_lengths(
        surprise, threshold, min_patch=min_patch, max_patch=max_patch,
    )
    P = patch_lengths.shape[1]

    h_pool = h_byte.new_zeros(B, P, D)
    h_acc  = h_byte.new_zeros(B, P, D)

    for b in range(B):
        cur = 0
        for p in range(P):
            L = int(patch_lengths[b, p].item())
            seg = h_byte[b, cur:cur+L, :]         # [L, D]
            # local pool: mean bytes inside the patch
            h_pool[b, p, :] = seg.mean(dim=0)
            # accumulating-state proxy at boundary: mean of last 3 bytes
            # in the segment (the bytes closest to the boundary carry the
            # most 'accumulated' state in the WKV sense).
            tail = min(3, L)
            h_acc[b, p, :]  = seg[-tail:, :].mean(dim=0)
            cur += L

    return h_pool, h_acc, patch_lengths


# ──────────────────────────────────────────────────────────────────────────────────
# 5. PatchRWKV slot — treats per-patch tokens as a 2-token sequence
# ──────────────────────────────────────────────────────────────────────────────────

class PatchSlot(nn.Module):
    """Patch slot — currently RWKV over 2-token-per-patch sequences.

    For each patch boundary we receive (h_pool, h_acc). The slot runs
    RWKVBlock(s) over the interleaved sequence, treating each pair as
    consecutive time-steps. Swappable: any sublayer with
    input [B, 2*num_patches, D] -> output [B, 2*num_patches, D] can replace.
    """

    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.gate = nn.Parameter(torch.zeros(dim))  # safe zero init

    def forward(self, h_pool: torch.Tensor, h_acc: torch.Tensor) -> torch.Tensor:
        # Interleave: [h_pool[0], h_acc[0], h_pool[1], h_acc[1], ...]
        B, P, D = h_pool.shape
        seq = torch.stack([h_pool, h_acc], dim=2).reshape(B, 2*P, D)
        h = seq
        for block in self.blocks:
            h, _ = block(h)
        # Reduce the 2-token-per-patch back to 1 token per patch by taking
        # the last step (more "settled"; matches our intent of one patch rep)
        h_per_patch = h[:, 1::2, :]   # take the (post-RWKV) h_acc positions
        return h_per_patch * torch.sigmoid(self.gate)


# ──────────────────────────────────────────────────────────────────────────────────
# 6. Byte decoder — RWKV over (h_byte + recontextualised patch info)
# ──────────────────────────────────────────────────────────────────────────────────

class ByteDecoder(nn.Module):
    def __init__(self, dim: int, vocab_size: int, n_layers: int):
        super().__init__()
        self.dim = dim
        self.fuse = nn.Linear(dim * 2, dim)
        self.fuse_gate = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)
        nn.init.zeros_(self.head.bias)

    def forward(self, h_byte, h_patch_per_byte):
        h_cat = torch.cat([h_byte, h_patch_per_byte], dim=-1)
        h_fused = self.fuse(h_cat)
        g = torch.sigmoid(self.fuse_gate)
        h = h_byte + g * h_fused
        for block in self.blocks:
            h, _ = block(h)
        h = self.ln_out(h)
        return self.head(h)


# ──────────────────────────────────────────────────────────────────────────────────
# 7. Main model: byte -> patch -> byte with all 3 changes
# ──────────────────────────────────────────────────────────────────────────────────

class ProtoBLT_RWKV_V2(nn.Module):
    """Proto-hierarchical BLT-RWKV v2.

    Args:
        vocab_size: byte-level vocab (258 typical).
        dim:        hidden dim used everywhere.
        n_enc_layers:  byte-encoder stack depth.
        n_patch_layers:patch-slot stack depth.
        n_dec_layers:  byte-decoder stack depth.
        horizon_enc:   per-block init target for encoder (bytes).
        horizon_patch: per-block init target for patch-slot (bytes).
        horizon_dec:   per-block init target for decoder (bytes).
        threshold:     surprise threshold for patch boundaries.
        min_patch:     smallest bytes/patch (>=1).
        max_patch:     cap on bytes per patch.
        pad_token_id:  vocab id for padding.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 96,
        n_enc_layers: int = 2,
        n_patch_layers: int = 1,
        n_dec_layers: int = 2,
        horizon_enc: int = 8,
        horizon_patch: int = 16,
        horizon_dec: int = 8,
        threshold: float = 0.30,
        min_patch: int = 2,
        max_patch: int = 32,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.min_patch = min_patch
        self.max_patch = max_patch
        self.threshold = threshold

        self.byte_enc = ByteEncoder(vocab_size, dim, n_enc_layers, pad_token_id)
        self.patch_slot = PatchSlot(dim, n_patch_layers)
        self.byte_dec = ByteDecoder(dim, vocab_size, n_dec_layers)

        # Re-initialise each block class toward its target horizon
        init_block_time_decay(self.byte_enc.blocks,  horizon_enc)
        init_block_time_decay(self.patch_slot.blocks, horizon_patch)
        init_block_time_decay(self.byte_dec.blocks,  horizon_dec)

    def forward(self, tokens):
        h_byte, surprise = self.byte_enc(tokens)
        # surprise is [B, T-1], h_byte is [B, T, D]
        h_pool, h_acc, patch_lengths = gather_patch_tokens(
            h_byte, surprise, threshold=self.threshold,
            min_patch=self.min_patch, max_patch=self.max_patch,
        )
        h_patch = self.patch_slot(h_pool, h_acc)
        # unpool over bytes: each byte inherits h_patch from its patch
        B, T = tokens.shape
        P = patch_lengths.shape[1]
        h_per_byte = h_byte.new_zeros(B, T, self.dim)
        cur = 0
        for b in range(B):
            for p in range(P):
                L = int(patch_lengths[b, p].item())
                h_per_byte[b, cur:cur+L, :] = h_patch[b, p, :]
                cur += L
            cur = 0
        logits = self.byte_dec(h_byte, h_per_byte)
        return logits, h_byte


if __name__ == "__main__":
    model = ProtoBLT_RWKV_V2(
        vocab_size=258, dim=96,
        n_enc_layers=2, n_patch_layers=1, n_dec_layers=2,
        horizon_enc=8, horizon_patch=16, horizon_dec=8,
        threshold=0.30,
    )
    print("ProtoBLT_RWKV_V2 (surprise-patch, accumulated-state, horizon init):")
    print(f"Total params: {count_params(model):,}")

    # Verify time_decay init landed where we want
    import torch
    for block in model.byte_enc.blocks:
        d = torch.exp(-torch.exp(block.time_decay))
        print(f"  encoder block  init decay  min={d.min():.3f}  med={d.median():.3f}  max={d.max():.3f}")
    for block in model.patch_slot.blocks:
        d = torch.exp(-torch.exp(block.time_decay))
        print(f"  patch_slot block  init decay  min={d.min():.3f}  med={d.median():.3f}  max={d.max():.3f}")
    for block in model.byte_dec.blocks:
        d = torch.exp(-torch.exp(block.time_decay))
        print(f"  decoder block init decay  min={d.min():.3f}  med={d.median():.3f}  max={d.max():.3f}")

    B, T = 2, 64
    x = torch.randint(1, 256, (B, T))
    logits, h_byte = model(x)
    print(f"\nForward pass:")
    print(f"  input: {x.shape}")
    print(f"  h_byte: {h_byte.shape}")
    print(f"  logits: {logits.shape}")

    loss = logits.sum()
    loss.backward()
    zero_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0
    )
    total = sum(1 for _ in model.parameters())
    print(f"\nGradient sanity: {zero_grad}/{total} params with zero grad")
