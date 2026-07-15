"""Surprise-patch primitives.

A *function* (not a module) that derives the surprise signal from a
sequence of byte-level hidden states, and a patcher that turns that
signal into `patch_lengths`. No parameters to learn.

Verified on `experiments/gpu_proto_blt_001/checkpoint.pt`:
  per-step state deltas (|h(t)-h(t-1)|, channel-mean) cluster around:
     1.0-1.5   for normal letters
     0.1-0.2   at double-letter positions ('tt', 'ee', 'oo' in "needle")
     1.8+      at word boundaries and punctuation
  → the signal is real, semantic, and strong enough to drive patching.
"""

from __future__ import annotations

import torch


def surprise_per_step(h: torch.Tensor) -> torch.Tensor:
    """Per-step state-delta surprise signal.

    Args:
        h: [B, T, D] byte-level hidden states (e.g. byte_enc output).

    Returns:
        [B, T-1] per-position surprise scalar. Lower = more predictable
        byte. Higher = more novel/shift-inducing byte in context.
    """
    return (h[:, 1:] - h[:, :-1]).abs().mean(dim=-1)


def surprise_to_patch_lengths(
    surprise: torch.Tensor,
    threshold: float,
    min_patch: int = 1,
    max_patch: int | None = None,
) -> torch.Tensor:
    """Convert a per-byte surprise curve into patch boundaries.

    Args:
        surprise: [B, T-1]          surprise scalar per byte position.
        threshold: float            Surprise values at or below this do NOT
                                    start a new patch. Values above threshold
                                    do. Position 0 always starts a patch.
        min_patch: int              Minimum bytes per patch (>=1).
        max_patch: int | None       Cap patch size if set.

    Returns:
        [B, num_patches] patch_lengths tensor (long). Always at least one
        patch per row.
    """
    B, T_minus_one = surprise.shape
    is_boundary = surprise > threshold  # [B, T-1]
    # Position 0 is always the start of a patch (so first byte is a patch).
    # We treat byte index `i` (0..T) as a boundary iff i==0 or
    # is_boundary[i-1] is True. Patch length = next_boundary - this_boundary.
    boundaries_flat = []
    for b in range(B):
        bs = [0]
        for i, decided in enumerate(is_boundary[b].tolist(), start=1):
            if decided:
                bs.append(i)
        if bs[-1] != T_minus_one + 1:
            bs.append(T_minus_one + 1)
        # Compute lengths
        lens = [bs[i+1] - bs[i] for i in range(len(bs) - 1)]
        # Apply min_patch by merging adjacent short patches
        merged: list[int] = []
        cur = 0
        for L in lens:
            cur += L
            if cur >= min_patch or not merged:
                merged.append(cur)
                cur = 0
        if cur > 0:
            merged[-1] += cur
        if max_patch is not None:
            fixed: list[int] = []
            for L in merged:
                while L > max_patch:
                    fixed.append(max_patch)
                    L -= max_patch
                fixed.append(L)
            merged = fixed
        boundaries_flat.append(merged)

    # pad to equal length
    max_p = max(len(p) for p in boundaries_flat)
    out = torch.zeros(B, max_p, dtype=torch.long)
    for b in range(B):
        for i, v in enumerate(boundaries_flat[b]):
            out[b, i] = v
    return out


def variable_pool_by_patch(
    h: torch.Tensor,
    patch_lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool bytes inside variable-length patches.

    Args:
        h: [B, T, D]               byte-level hidden states.
        patch_lengths: [B, P]      long tensor. sum == T per row.

    Returns:
        [B, P, D] per-patch pooled representation.
    """
    B, T, D = h.shape
    _, P = patch_lengths.shape
    out = h.new_zeros(B, P, D)
    cur = 0
    for b in range(B):
        for p_idx in range(P):
            L = int(patch_lengths[b, p_idx].item())
            seg = h[b, cur:cur+L, :]
            out[b, p_idx, :] = seg.mean(dim=0)
            cur += L
        cur = 0
    return out


# Smoke test
if __name__ == "__main__":
    torch.manual_seed(0)
    h = torch.randn(2, 12, 8)
    s = surprise_per_step(h)
    print(f"surprise shape: {s.shape}  mean={s.mean():.4f}")
    pl = surprise_to_patch_lengths(s, threshold=0.7, min_patch=2, max_patch=5)
    print(f"patch_lengths: {pl.tolist()}")
    pooled = variable_pool_by_patch(h, pl)
    print(f"pooled shape: {pooled.shape}")
