"""B3D-RWKV nano: Triplet-Block Diffusion RWKV at small scale.

Paper: Triplet-Block Diffusion RWKV (Lin et al. 2026) – https://arxiv.org/abs/2605.25969
HF: https://huggingface.co/leonardklin/B3D-RWKV

Core idea (implemented minimal):
Each logical block of size B appears as three physical blocks:
  b1 = masked copy (random masks)
  b2 = identical masked copy (lossable)
  b3 = clean copy (refreshes state)

Because RWKV is causal left→right, hidden state arriving at masked pos in b2
has already seen all unmasked tokens from b1 (including future positions within block),
giving pseudo-bidirectional context while staying strictly causal.

This file implements:
- B3DRWKVModel: wraps RWKVNano, provides triplet construction + diffusion inference
- Functions to build triplet sequences from clean tokens
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.rwkv_nano import RWKVNano, count_params

MASK_ID = None  # will be set to vocab_size-1 or dedicated mask token; for byte vocab we use 257 (PAD=0, byte 1..256, mask 257, unk 258? But we use 258 vocab)
# For simplicity we reuse PAD_ID as mask? No, need distinct.
# We'll define mask token as vocab_size-1 = 257 for 258 vocab (0=pad, 1-256=byte, 257=mask)
# For char vocab of 74, mask = len(vocab)-1 as well.

def build_triplet_batch(
    clean_tokens: torch.Tensor,  # [B, L] logical clean
    block_size: int,
    mask_ratio: float,
    mask_id: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Build triplet physical sequence from clean logical tokens.

    Args:
        clean_tokens: [B, L] clean tokens (L divisible by block_size)
        block_size: B in paper (e.g., 8)
        mask_ratio: probability of masking
        mask_id: token id for mask
        generator: optional torch generator for reproducibility

    Returns:
        physical_tokens: [B, 3L] = concat per block [b1,b2,b3]
        loss_mask: [B, 3L] bool, 1 only where b2 is masked (loss computed)
        targets: [B, 3L] clean tokens repeated for loss reference (only b2 masked positions matter)
        meta: dict with per-block info
    """
    B, L = clean_tokens.shape
    assert L % block_size == 0, f"L {L} not divisible by block_size {block_size}"
    num_blocks = L // block_size

    physical_list = []
    loss_mask_list = []
    targets_list = []

    for blk_idx in range(num_blocks):
        blk_clean = clean_tokens[:, blk_idx*block_size:(blk_idx+1)*block_size]  # [B, Bsize]

        # random mask [B, Bsize]
        if generator is not None:
            rand = torch.rand(blk_clean.shape, device=blk_clean.device, generator=generator)
        else:
            rand = torch.rand(blk_clean.shape, device=blk_clean.device)
        mask = rand < mask_ratio  # [B, Bsize] bool

        b1 = torch.where(mask, torch.full_like(blk_clean, mask_id), blk_clean)
        b2 = b1.clone()  # identical masked copy
        b3 = blk_clean  # clean copy

        # physical for this block: b1 || b2 || b3
        physical_blk = torch.cat([b1, b2, b3], dim=1)  # [B, 3*Bsize]
        physical_list.append(physical_blk)

        # loss mask: only b2 masked positions (middle third)
        loss_blk = torch.zeros_like(physical_blk, dtype=torch.bool)
        # b1: 0..Bsize-1, b2: Bsize..2*Bsize-1, b3: 2Bsize..3Bsize-1
        loss_blk[:, block_size:2*block_size] = mask

        loss_mask_list.append(loss_blk)

        # targets: for loss positions, clean token; otherwise ignored (set to 0)
        target_blk = torch.zeros_like(physical_blk)
        # copy clean to b2 positions for reference
        target_blk[:, block_size:2*block_size] = blk_clean
        targets_list.append(target_blk)

    physical = torch.cat(physical_list, dim=1)  # [B, 3L]
    loss_mask = torch.cat(loss_mask_list, dim=1)
    targets = torch.cat(targets_list, dim=1)

    meta = {"num_blocks": num_blocks, "block_size": block_size, "mask_ratio": mask_ratio}

    return physical, loss_mask, targets, meta


class B3DRWKVModel(nn.Module):
    """Nano B3D-RWKV wrapper."""

    def __init__(self, vocab_size: int = 258, dim: int = 128, num_layers: int = 3, hidden_scale: int = 4, pad_token_id: int = 0, mask_token_id: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size - 1
        self.backbone = RWKVNano(vocab_size=vocab_size, dim=dim, num_layers=num_layers, hidden_scale=hidden_scale, pad_token_id=pad_token_id)

    def forward(self, physical_tokens: torch.Tensor, state: Optional[list[dict]] = None, return_state: bool = False):
        """
        Args:
            physical_tokens: [B, 3L] triplet sequence
            state: optional RWKV state carry
            return_state: if True return new_state
        Returns:
            logits: [B, 3L, vocab]
            new_state: optional
        """
        logits, new_state = self.backbone(physical_tokens, state=state, return_state=return_state)
        if return_state:
            return logits, new_state
        return logits

    def diffusion_decode_block(
        self,
        context_tokens: torch.Tensor,  # [B, Lc] clean context (previous blocks' b3s)
        block_size: int,
        mask_id: int,
        tau: float = 0.9,
        max_iters: int = 10,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, int, dict]:
        """
        Per-block iterative denoising at inference (Figure 1b).

        Args:
            context_tokens: [B, Lc] clean context (could be empty)
            block_size: logical block size to decode (B)
            mask_id: mask token id
            tau: commit threshold (commit if top1 prob > tau)
            max_iters: max diffusion iterations
            device: device

        Returns:
            decoded_block: [B, block_size] clean block after diffusion (fully committed)
            num_iters: number of iterations used
            info: dict with commit history
        """
        B = context_tokens.shape[0] if context_tokens.numel()>0 else 1
        if device is None:
            device = context_tokens.device if context_tokens.numel()>0 else torch.device("cpu")

        # Start with fully masked block as b1
        b1 = torch.full((B, block_size), mask_id, dtype=torch.long, device=device)
        # For demo, we assume no unmasked tokens initially (could have partial)
        # In real use, b1 may have some known unmasked tokens from previous commits

        commit_history = []
        num_iters = 0

        current_state = None
        # First, run context to get state
        if context_tokens.numel()>0:
            _, current_state = self.backbone(context_tokens, return_state=True)

        for it in range(max_iters):
            # Build triplet for current iteration: b1 (current masked), b2 = b1 copy, b3 dummy (not needed for inference of this block, but we include empty)
            # For inference we only need b1 + b2 to predict b2
            b2 = b1.clone()
            # physical for this block inference = context + b1 + b2
            # Actually per paper inference runs over [c + b1 + b2], and commits in b2, then copies commit to b1 for next iter
            # We'll do: input = context + b1 + b2, get logits for b2 part, commit
            physical = torch.cat([context_tokens, b1, b2], dim=1) if context_tokens.numel()>0 else torch.cat([b1, b2], dim=1)

            with torch.no_grad():
                logits, _ = self.backbone(physical, state=current_state, return_state=False)
                # logits for b2 part: context_len + block_size .. context_len + 2*block_size -1
                ctx_len = context_tokens.shape[1] if context_tokens.numel()>0 else 0
                b2_logits = logits[:, ctx_len+block_size:ctx_len+2*block_size, :]  # [B, block_size, vocab]
                probs = F.softmax(b2_logits, dim=-1)
                top1_prob, top1_id = probs.max(dim=-1)  # [B, block_size]

                # commit where top1_prob > tau and currently masked
                currently_masked = (b1 == mask_id)
                commit_mask = (top1_prob > tau) & currently_masked

                num_committed = int(commit_mask.sum().item())
                commit_history.append({"iter": it, "committed": num_committed, "total_masked_before": int(currently_masked.sum().item())})

                if num_committed == 0 and it > 0:
                    # no progress, break to avoid infinite loop; optionally lower tau?
                    # For now break
                    pass

                # commit
                b1 = torch.where(commit_mask, top1_id, b1)

                # check if fully committed
                if not (b1 == mask_id).any():
                    num_iters = it + 1
                    break

                num_iters = it + 1

                # optional: re-run context state? State should be preserved from context only, not include b1/b2?
                # Per paper, b3 refreshes state, but during block diffusion, state from context is fixed, b1/b2 are processed with that state as start.
                # So we keep current_state from context.

        # If still masked after max_iters, force commit with top1 (greedy)
        if (b1 == mask_id).any():
            # final greedy commit
            physical = torch.cat([context_tokens, b1, b1.clone()], dim=1) if context_tokens.numel()>0 else torch.cat([b1, b1.clone()], dim=1)
            with torch.no_grad():
                logits, _ = self.backbone(physical, state=current_state, return_state=False)
                ctx_len = context_tokens.shape[1] if context_tokens.numel()>0 else 0
                b2_logits = logits[:, ctx_len+block_size:ctx_len+2*block_size, :]
                top1_id = b2_logits.argmax(dim=-1)
                b1 = torch.where(b1 == mask_id, top1_id, b1)

        return b1, num_iters, {"commit_history": commit_history}

    def get_throughput_metrics(self, commit_histories):
        """Compute avg iters per block"""
        avg_iters = sum(h["num_iters"] for h in commit_histories) / max(len(commit_histories),1)
        return {"avg_iters_per_block": avg_iters, "speedup_vs_AR": 1.0}  # placeholder, AR needs block_size passes


# Smoke test
if __name__ == "__main__":
    vocab=70
    mask_id=vocab-1
    model=B3DRWKVModel(vocab_size=vocab, dim=32, num_layers=2, mask_token_id=mask_id)
    B=2
    L=16
    block_size=8
    clean=torch.randint(1, vocab-1, (B, L))
    physical, loss_mask, targets, meta = build_triplet_batch(clean, block_size, mask_ratio=0.3, mask_id=mask_id)
    print(f"clean {clean.shape} -> physical {physical.shape} loss_mask sum {loss_mask.sum()}")
    logits=model(physical)
    print(f"logits {logits.shape} params {count_params(model)}")
    # diffusion decode one block
    ctx=torch.randint(1, vocab-1, (B, 8))
    decoded, iters, info = model.diffusion_decode_block(ctx, block_size=8, mask_id=mask_id, tau=0.9, max_iters=5)
    print(f"decoded {decoded.shape} iters {iters} info {info}")
