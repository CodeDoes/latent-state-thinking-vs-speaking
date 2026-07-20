"""Diffusion Grid Terminal model – screen -> screen-rwkv -> screen with certainty trigger.

Implements theories/diffusion-grid-terminal.md:

- Grid of bytes H×W (e.g., 16×32=512) as screen buffer
- RWKV diffusion denoising with triplet-block layout (from b3d_rwkv_model)
- Stochastic output: commit cells where p(top1) > τ, otherwise repeat diffusion
- Recurrent state carry across screens (temporal awareness)
- Reasoning traces: dedicated trace rows that model can use as scratchpad
- Tool calling: typing triggered only at full certainty

This nano version is CPU runnable, synthetic data.

Two modes:
- grid_recon: reconstruct masked grid (GT1, GT6 vision box)
- terminal_chain: 3-step chain where screen_t contains info needed at screen_{t+2} (GT2 temporal)
- trace: dedicated trace rows load-bearing (GT3)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import random

from domains.rwkv.rwkv_nano import RWKVNano, count_params
from threads.b3d.b3d_rwkv_model import build_triplet_batch, B3DRWKVModel

# For grid, we use byte vocab 258 (0 PAD, 1-256 byte, 257 MASK)
BYTE_PAD = 0
BYTE_UNK = 1
BYTE_MASK = 257
BYTE_VOCAB = 258

class GridRWKVModel(nn.Module):
    """Grid diffusion model with 2D row/col embeddings."""

    def __init__(self, H=16, W=32, dim=64, num_layers=2, vocab_size=BYTE_VOCAB, pad_id=BYTE_PAD, mask_id=BYTE_MASK, use_2d_pos=True):
        super().__init__()
        self.H = H
        self.W = W
        self.dim = dim
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.use_2d_pos = use_2d_pos

        self.backbone = B3DRWKVModel(vocab_size=vocab_size, dim=dim, num_layers=num_layers, pad_token_id=pad_id, mask_token_id=mask_id)

        if use_2d_pos:
            self.row_embed = nn.Embedding(H, dim)
            self.col_embed = nn.Embedding(W, dim)

    def add_2d_pos(self, tokens: torch.Tensor) -> torch.Tensor:
        """Not used directly – we need to add to backbone's embedding. Instead we override backbone? Simpler: we keep backbone's embed and add pos after embed.
        We'll implement custom forward that adds 2D pos to RWKV input after embed.
        """
        # This method is placeholder – actual addition done in forward_grid
        return tokens

    def forward_grid(self, physical_tokens: torch.Tensor, state: Optional[list[dict]] = None, return_state: bool = False):
        """Forward with optional 2D pos injection.
        physical_tokens: [B, L] where L = 3*H*W or H*W (triplet or not)
        We inject row/col pos by adding to the embedded representation inside RWKVNano?
        For simplicity, we add a learned offset to logits? Better: we intercept backbone embed.

        Minimal approach: we add row/col embeddings to the token embeddings before RWKV blocks.
        So we need to manually do embed + pos, then run blocks.
        """
        B, T = physical_tokens.shape
        # get backbone embed
        x = self.backbone.backbone.embed(physical_tokens)  # [B,T,dim]

        if self.use_2d_pos:
            # map each position in flattened grid to row/col
            # For physical tokens that are triplet expanded (3*H*W), we need to map modulo H*W
            # Position p in [0, H*W-1] -> row = p // W, col = p % W
            # For triplet positions, we do modulo to get logical position
            H, W = self.H, self.W
            HW = H * W
            # compute logical position = token index % HW (since triplet repeats)
            # Actually for triplet, each logical block is block_size = e.g., 8, but for grid we treat whole grid as one block
            # So for T = 3*HW, positions 0..HW-1 = b1, HW..2HW-1 = b2, 2HW..3HW-1 = b3, all map to same logical row/col
            # So pos_logical = pos % HW
            pos = torch.arange(T, device=physical_tokens.device).unsqueeze(0).expand(B, -1)  # [B,T]
            pos_logical = pos % HW
            rows = pos_logical // W  # [B,T]
            cols = pos_logical % W
            row_e = self.row_embed(rows)  # [B,T,dim]
            col_e = self.col_embed(cols)
            x = x + row_e + col_e

        # now run through RWKV blocks manually (since we already did embed)
        new_state = [] if return_state else None
        h = x
        for i, block in enumerate(self.backbone.backbone.blocks):
            layer_state = state[i] if state is not None else None
            h, s = block(h, layer_state)
            if return_state:
                new_state.append(s)
        h = self.backbone.backbone.ln_out(h)
        logits = self.backbone.backbone.head(h)

        if return_state:
            return logits, new_state
        return logits

    def diffusion_decode_grid(
        self,
        context_state: Optional[list[dict]],
        H: int,
        W: int,
        mask_id: int,
        tau: float = 0.9,
        max_iters: int = 10,
        device: torch.device = torch.device("cpu"),
        initial_grid: Optional[torch.Tensor] = None,  # [B, H*W] partially known
    ) -> Tuple[torch.Tensor, int, dict]:
        """
        Diffusion decode a full grid.

        Args:
            context_state: RWKV state from previous screens (temporal carry)
            H,W: grid dims
            mask_id: mask token
            tau: certainty threshold
            max_iters: max diffusion steps
            initial_grid: [B, H*W] with some known tokens and MASK elsewhere. If None, fully masked.

        Returns:
            decoded_grid: [B, H*W] fully decoded
            num_iters: steps used
            info: commit history
        """
        B = 1
        if initial_grid is not None:
            B = initial_grid.shape[0]
            grid = initial_grid.clone()
        else:
            grid = torch.full((B, H*W), mask_id, dtype=torch.long, device=device)

        HW = H*W
        commit_history = []
        num_iters = 0

        for it in range(max_iters):
            # build triplet for current grid: b1 = current masked grid, b2 = same, b3 dummy
            b1 = grid
            b2 = grid.clone()
            physical = torch.cat([b1, b2], dim=1)  # [B, 2*HW] – we omit b3 for inference grid (no clean refresh needed within same screen)

            with torch.no_grad():
                logits = self.forward_grid(physical, state=context_state, return_state=False)
                # logits for b2 part: positions HW .. 2HW-1
                b2_logits = logits[:, HW:2*HW, :]  # [B, HW, vocab]
                probs = F.softmax(b2_logits, dim=-1)
                top_prob, top_id = probs.max(dim=-1)  # [B, HW]

                masked = (grid == mask_id)
                commit = (top_prob > tau) & masked
                num_comm = int(commit.sum().item())
                commit_history.append({"iter": it, "committed": num_comm, "masked_before": int(masked.sum().item())})

                grid = torch.where(commit, top_id, grid)

                if not (grid == mask_id).any():
                    num_iters = it + 1
                    break

                num_iters = it + 1

        # final greedy if still masked
        if (grid == mask_id).any():
            physical = torch.cat([grid, grid.clone()], dim=1)
            with torch.no_grad():
                logits = self.forward_grid(physical, state=context_state, return_state=False)
                b2_logits = logits[:, HW:2*HW, :]
                top_id = b2_logits.argmax(dim=-1)
                grid = torch.where(grid == mask_id, top_id, grid)

        return grid, num_iters, {"commit_history": commit_history}

# Synthetic grid generators for experiments

def generate_grid_random(H=16, W=32, vocab_low=32, vocab_high=126, device="cpu"):
    """Random printable ASCII grid."""
    B=1
    HW=H*W
    # random printable
    grid = torch.randint(vocab_low, vocab_high, (B, HW), device=device, dtype=torch.long)
    return grid

def generate_grid_box(H=16, W=32, device="cpu"):
    """Grid with box border +- | – tests 2D vision (GT6)."""
    B=1
    HW=H*W
    grid = torch.full((B, HW), 32, dtype=torch.long, device=device)  # space
    for r in range(H):
        for c in range(W):
            idx = r*W + c
            if r==0 or r==H-1:
                if c==0 or c==W-1:
                    grid[0, idx] = ord('+')
                else:
                    grid[0, idx] = ord('-')
            elif c==0 or c==W-1:
                grid[0, idx] = ord('|')
    return grid

def generate_terminal_chain(H=16, W=32, device="cpu"):
    """Synthetic 3-step terminal chain:
    Screen1: shows file list e.g., 'file1.txt file2.txt'
    Screen2: after cat file1.txt, shows content 'SECRET=42'
    Screen3: prompt asks 'What was SECRET?' – answer needs memory of screen2
    Returns list of 3 grids.
    """
    B=1
    HW=H*W
    # helper to write string into grid row
    def write_str(grid, row, col, s):
        for i,ch in enumerate(s):
            if col+i < W and row < H:
                grid[0, row*W + col + i] = ord(ch)

    screens=[]
    # screen1
    g1 = torch.full((B, HW), 32, dtype=torch.long, device=device)
    write_str(g1, 0, 0, "$ ls")
    write_str(g1, 1, 0, "file1.txt file2.txt notes.md")
    screens.append(g1)

    # screen2
    g2 = torch.full((B, HW), 32, dtype=torch.long, device=device)
    write_str(g2, 0, 0, "$ cat file1.txt")
    write_str(g2, 1, 0, "SECRET=42")
    write_str(g2, 2, 0, "DATA=hello world")
    screens.append(g2)

    # screen3
    g3 = torch.full((B, HW), 32, dtype=torch.long, device=device)
    write_str(g3, 0, 0, "$")
    write_str(g3, 1, 0, "What was SECRET? Answer:")
    screens.append(g3)

    return screens

def generate_grid_with_trace(H=16, W=32, device="cpu"):
    """Grid with trace rows 10-13 reserved for scratchpad.
    Task: sum numbers in rows 0-2, store running total in trace.
    """
    B=1
    HW=H*W
    g = torch.full((B, HW), 32, dtype=torch.long, device=device)
    def write_str(row, col, s):
        for i,ch in enumerate(s):
            if col+i < W:
                g[0, row*W+col+i] = ord(ch)
    # rows 0-2 have numbers
    write_str(0, 0, "NUMS: 5 7 3")
    write_str(1, 0, "Running total should be in trace row 10")
    write_str(2, 0, "Trace row 10-13 is scratchpad")
    # trace rows empty initially
    return g

if __name__ == "__main__":
    model = GridRWKVModel(H=8, W=8, dim=32, num_layers=2)
    print(f"params {count_params(model)}")
    # test random grid recon
    grid = generate_grid_box(H=8, W=8, device="cpu")
    print(f"grid shape {grid.shape}")
    # build triplet
    block_size = 8*8
    physical, loss_mask, targets, meta = build_triplet_batch(grid, block_size, mask_ratio=0.3, mask_id=BYTE_MASK)
    print(f"physical {physical.shape} loss_mask sum {loss_mask.sum()}")
    logits = model.forward_grid(physical)
    print(f"logits {logits.shape}")
    # diffusion decode
    masked = grid.clone()
    # mask 30%
    mask = torch.rand(grid.shape) < 0.3
    masked[mask] = BYTE_MASK
    decoded, iters, info = model.diffusion_decode_grid(None, H=8, W=8, mask_id=BYTE_MASK, tau=0.9, max_iters=10, device=torch.device("cpu"), initial_grid=masked)
    print(f"decoded iters {iters} info {info}")
