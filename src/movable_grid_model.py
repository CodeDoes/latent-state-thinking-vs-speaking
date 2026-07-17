"""Movable Grid Scratchboard RWKV – minimal implementation for MG1.

Primitives:
- Grid memory G [B,H,W,D] per sample (scratchboard)
- Position pos [B,2] (x,y) integer, movable
- Read: crop 3x3 patch around pos (or single cell) -> read vector
- Write: w [D], gate g in [0,1], G[y,x] = (1-g)*G + g*w
- Move: Δ from head or teacher forced to described pos

Task: Place&Query 8x8
- World 8x8 random symbols A-Z
- Input: 10 placements (sym, x, y)
- Model must build grid, then answer query (x_q, y_q) -> symbol

Variants:
- static: pos fixed at (0,0), no movement
- nogrid: no grid, answer from RWKV state only (like exp001)
- full: movable + grid, pos teacher forced to described coords (or learned)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import random

from src.rwkv_nano import RWKVNano, count_params

# Vocab for symbols: 0=PAD, 1-26=A-Z, 27=UNK, etc.
SYMBOL_VOCAB = 32
PAD_ID = 0

class MovableGridMemory(nn.Module):
    """Grid memory H×W×D per batch, differentiable read/write."""

    def __init__(self, H=8, W=8, D=32):
        super().__init__()
        self.H = H
        self.W = W
        self.D = D

    def init_grid(self, B, device):
        # zeros
        return torch.zeros(B, self.H, self.W, self.D, device=device)

    def read(self, grid: torch.Tensor, pos: torch.Tensor, patch=1) -> torch.Tensor:
        """
        Read from grid at pos.
        pos: [B,2] (x,y) integer 0..W-1, 0..H-1
        patch: 1 = single cell, 3 = 3x3 mean
        Returns: [B, D] or [B, patch*patch*D] if patch>1 flattened?
        For simplicity patch=1 -> [B,D], patch=3 -> mean over 3x3 -> [B,D]
        """
        B, H, W, D = grid.shape
        # clamp pos
        x = pos[:, 0].clamp(0, W-1).long()
        y = pos[:, 1].clamp(0, H-1).long()

        if patch == 1:
            # gather
            # grid [B,H,W,D], we want [B,D] at (y,x)
            # use advanced indexing
            batch_idx = torch.arange(B, device=grid.device)
            out = grid[batch_idx, y, x]  # [B,D]
            return out
        else:
            # 3x3 mean
            # collect neighborhood
            # For simplicity, loop (small H,W)
            out = torch.zeros(B, D, device=grid.device)
            count = 0
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    ny = (y + dy).clamp(0, H-1)
                    nx = (x + dx).clamp(0, W-1)
                    batch_idx = torch.arange(B, device=grid.device)
                    out = out + grid[batch_idx, ny, nx]
                    count += 1
            return out / count

    def write(self, grid: torch.Tensor, pos: torch.Tensor, w: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Write w to grid at pos with gate g.
        grid: [B,H,W,D]
        pos: [B,2] (x,y)
        w: [B,D]
        g: [B,1] or [B] in [0,1]
        Returns: new grid (clone)
        """
        B, H, W, D = grid.shape
        x = pos[:, 0].clamp(0, W-1).long()
        y = pos[:, 1].clamp(0, H-1).long()
        batch_idx = torch.arange(B, device=grid.device)

        # compute new value
        # g shape [B,1] -> [B,D]
        if g.dim() == 1:
            g = g.unsqueeze(-1)
        if g.shape[-1] == 1:
            g = g.expand(-1, D)

        # Use scatter-like update: we need to update grid[batch, y, x] = (1-g)*old + g*w
        # Clone
        new_grid = grid.clone()
        old = grid[batch_idx, y, x]  # [B,D]
        updated = (1 - g) * old + g * w
        # write back
        new_grid[batch_idx, y, x] = updated
        return new_grid


class MovableGridModel(nn.Module):
    """Full model with RWKV core + movable grid."""

    def __init__(self, H=8, W=8, D=32, dim=64, num_layers=2, symbol_vocab=SYMBOL_VOCAB, mode="full", use_3x3_read=False):
        """
        mode: 'static', 'nogrid', 'full', 'nopos' (no pos text)
        """
        super().__init__()
        self.H = H
        self.W = W
        self.D = D
        self.dim = dim
        self.mode = mode
        self.use_3x3 = use_3x3_read

        # RWKV processes placement embeddings
        # Input to RWKV: sym_emb + read_vector (if grid) -> dim
        self.sym_embed = nn.Embedding(symbol_vocab, dim)
        self.pos_embed_x = nn.Embedding(W, dim//2)
        self.pos_embed_y = nn.Embedding(H, dim//2)

        # If grid mode, RWKV input is sym_emb + read (D) projected to dim
        self.read_proj = nn.Linear(D, dim) if mode != "nogrid" else None

        vocab_size = symbol_vocab + 10  # extra for byte chars if needed, but we use symbol vocab for RWKV input
        # RWKV backbone: we feed sequence of placement embeddings (len = num_placements)
        self.rwkv = RWKVNano(vocab_size=symbol_vocab, dim=dim, num_layers=num_layers, pad_token_id=PAD_ID)

        self.grid_mem = MovableGridMemory(H, W, D) if mode != "nogrid" else None

        # Heads
        self.write_head = nn.Linear(dim, D) if mode != "nogrid" else None
        self.gate_head = nn.Linear(dim, 1) if mode != "nogrid" else None
        self.move_head = nn.Linear(dim, 2) if mode == "full" else None  # Δx, Δy
        self.qa_head = nn.Linear(D if mode != "nogrid" else dim, symbol_vocab)  # predict symbol from grid read or state

    def forward(self, placements: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], query_pos: torch.Tensor, teacher_pos: Optional[List[torch.Tensor]] = None):
        """
        placements: list of length num_placements, each is (sym [B], x [B], y [B])
        query_pos: [B,2] query x,y
        teacher_pos: optional list of pos for teacher forcing movement (same as placements x,y)

        Returns:
            logits: [B, symbol_vocab] answer logits for query
            grid: final grid [B,H,W,D] (if applicable)
            pos_traj: list of pos [B,2] per step
            extra: dict
        """
        B = placements[0][0].shape[0]
        device = placements[0][0].device

        if self.mode != "nogrid":
            grid = self.grid_mem.init_grid(B, device)
        else:
            grid = None

        # start pos at (0,0)
        pos = torch.zeros(B, 2, dtype=torch.long, device=device)  # x,y

        pos_traj = [pos.clone()]

        # RWKV state (for processing placement sequence)
        rwkv_state = None

        # Process each placement
        for idx, (sym, x, y) in enumerate(placements):
            # x,y are [B] ints
            # embedding
            sym_e = self.sym_embed(sym)  # [B,dim]

            if self.mode != "nogrid":
                # read from current pos (before move/write, or after? We read before write)
                read_vec = self.grid_mem.read(grid, pos, patch=3 if self.use_3x3 else 1)  # [B,D]
                read_proj = self.read_proj(read_vec)  # [B,dim]
                inp = sym_e + read_proj
            else:
                inp = sym_e

            # we need to feed inp as token embedding to RWKV – but RWKV expects token ids, not embeddings.
            # For simplicity, we bypass RWKV token embed and directly feed inp as if it were embedded token sequence of length 1.
            # We'll run RWKV block manually: we can use rwkv.blocks with inp unsqueezed as [B,1,dim] and state.
            # For minimal, we use rwkv backbone's embed as identity? Let's instead use rwkv as sequence processor of inp.
            # Implementation: we have rwkv.blocks, we can run forward with custom x.

            # Manual forward through RWKV blocks for single step
            # We need to handle state
            # x input to blocks is inp [B,dim] -> unsqueeze [B,1,dim]
            x_seq = inp.unsqueeze(1)  # [B,1,dim]
            # We need to run through blocks with state
            # Use rwkv.blocks directly
            new_state = []
            h = x_seq
            for i, block in enumerate(self.rwkv.blocks):
                layer_state = rwkv_state[i] if rwkv_state is not None else None
                h, s = block(h, layer_state)
                new_state.append(s)
            rwkv_state = new_state
            h = self.rwkv.ln_out(h)  # [B,1,dim]
            h = h.squeeze(1)  # [B,dim]

            # heads
            if self.mode != "nogrid":
                w = self.write_head(h)  # [B,D]
                g = torch.sigmoid(self.gate_head(h))  # [B,1]

                # movement
                if self.mode == "full":
                    if teacher_pos is not None:
                        # teacher forced: pos = teacher target (x,y of this placement)
                        # teacher_pos[idx] is [B,2] target pos
                        pos = teacher_pos[idx]
                    else:
                        # predicted move
                        delta = torch.tanh(self.move_head(h))  # [B,2] in [-1,1]
                        # scale and round to int step
                        delta_int = torch.round(delta).long()  # -1,0,1
                        pos = pos + delta_int
                        pos[:, 0] = pos[:, 0].clamp(0, self.W-1)
                        pos[:, 1] = pos[:, 1].clamp(0, self.H-1)
                else:
                    # static mode: pos fixed at (0,0)
                    pass

                # write to grid at (maybe new pos or old pos? Write at target pos = x,y of placement)
                # For teacher forcing, we write at described position, not current pos before move
                # Let's write at described (x,y) which is the placement's target
                write_pos = torch.stack([x, y], dim=1) if self.mode in ("full", "static") else pos
                # For static mode, write_pos is still forced to described? But static means we ignore movement and fix at (0,0) for both read and write
                if self.mode == "static":
                    write_pos = torch.zeros_like(write_pos)  # force (0,0)

                grid = self.grid_mem.write(grid, write_pos, w, g)

            pos_traj.append(pos.clone())

        # After processing placements, answer query
        if self.mode != "nogrid":
            # read at query pos
            read_q = self.grid_mem.read(grid, query_pos, patch=3 if self.use_3x3 else 1)  # [B,D]
            logits = self.qa_head(read_q)  # [B, vocab]
        else:
            # no grid, answer from last RWKV state h (which is last placement hidden)
            logits = self.qa_head(h)  # [B, vocab]

        return logits, grid, pos_traj, {"final_pos": pos}

def generate_place_query_batch(B=4, H=8, W=8, num_placements=10, num_symbols=26, device="cpu"):
    """
    Generate random world and placements.
    Returns:
      world [B,H,W] int symbol ids (0 empty, 1..num_symbols)
      placements list of (sym, x, y) each [B]
      query_pos [B,2], query_sym [B]
      text description (optional)
    """
    world = torch.zeros(B, H, W, dtype=torch.long, device=device)
    placements = []  # list of tuples
    # For each placement, pick random pos and symbol that not overlapping? Allow overwrite for MG4 test
    for _ in range(num_placements):
        sym = torch.randint(1, num_symbols+1, (B,), device=device)
        x = torch.randint(0, W, (B,), device=device)
        y = torch.randint(0, H, (B,), device=device)
        # write to world (last write wins)
        for b in range(B):
            world[b, y[b], x[b]] = sym[b]
        placements.append((sym, x, y))

    # query: pick random placement index
    q_idx = random.randint(0, num_placements-1)
    # query pos = placements[q_idx] x,y, query sym = world at that pos (should be last written if overlapping)
    # Instead of using placements[q_idx], use world random pos that has symbol?
    # For simplicity query random pos that has non-zero in world, if none pick random placement
    query_pos = torch.zeros(B, 2, dtype=torch.long, device=device)
    query_sym = torch.zeros(B, dtype=torch.long, device=device)
    for b in range(B):
        # find non-zero cells
        nz = (world[b] != 0).nonzero(as_tuple=False)  # [N,2] y,x
        if len(nz) > 0:
            pick = nz[random.randint(0, len(nz)-1)]
            yq, xq = pick[0].item(), pick[1].item()
            query_pos[b, 0] = xq
            query_pos[b, 1] = yq
            query_sym[b] = world[b, yq, xq]
        else:
            # fallback to first placement
            sym, x, y = placements[0]
            query_pos[b, 0] = x[b]
            query_pos[b, 1] = y[b]
            query_sym[b] = sym[b]

    # teacher positions for movement: list of [B,2] pos per step (where agent should move)
    teacher_pos = []
    for (sym, x, y) in placements:
        pos = torch.stack([x, y], dim=1)  # [B,2]
        teacher_pos.append(pos)

    return world, placements, query_pos, query_sym, teacher_pos

if __name__ == "__main__":
    B=2
    H=W=8
    D=32
    dim=64
    model_full = MovableGridModel(H=H, W=W, D=D, dim=dim, num_layers=2, mode="full")
    model_static = MovableGridModel(H=H, W=W, D=D, dim=dim, num_layers=2, mode="static")
    model_nogrid = MovableGridModel(H=H, W=W, D=D, dim=dim, num_layers=2, mode="nogrid")
    print(f"full params {count_params(model_full):,} static {count_params(model_static):,} nogrid {count_params(model_nogrid):,}")

    world, placements, qpos, qsym, teacher_pos = generate_place_query_batch(B=B, H=H, W=W, num_placements=5, device="cpu")
    print(f"world {world.shape} placements {len(placements)} qpos {qpos} qsym {qsym}")

    logits, grid, traj, extra = model_full(placements, qpos, teacher_pos=teacher_pos)
    print(f"logits {logits.shape} grid {grid.shape if grid is not None else None} traj len {len(traj)}")
    pred = logits.argmax(dim=-1)
    print(f"pred {pred} true {qsym} acc {(pred==qsym).float().mean():.2f}")
