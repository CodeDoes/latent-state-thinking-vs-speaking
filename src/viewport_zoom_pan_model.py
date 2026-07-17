"""Viewport with Zoom/Pan/Quant to keep context small – for massive content.

World: H_world×W_world large (e.g., 32×32 or 100×80)
Viewport: H_view×W_view small fixed (e.g., 8×8) – per-step context budget

Controllable:
- pos (x,y) – pan/scroll
- zoom z ∈ {1,2,4,8} – coverage: viewport covers H_view*z × W_view*z world area, downsampled to H_view×W_view
- quant q ∈ {1,2,4} – value quantization: bin symbol ids into coarse bins, losing detail

Read:
  world_patch = crop(world, center=pos, size=H_view*z × W_view*z)
  viewport = downsample(world_patch, factor=z)  # mean pool z×z → H_view×W_view
  viewport = quantize_value(viewport, q)  # bin values

Write: same as movable grid (write to world at pos)

Actions from RWKV hidden h:
  Δx,Δy ∈ [-1,0,1], Δz, Δq

This keeps per-step context = H_view×W_view = 64 tokens even if world = 64×64=4096.

Proves VZ1-VZ6.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import random

from src.movable_grid_model import MovableGridMemory, count_params
from src.rwkv_nano import RWKVNano

SYMBOL_VOCAB = 32
PAD_ID = 0

class ViewportMemory(nn.Module):
    def __init__(self, H_world=32, W_world=32, H_view=8, W_view=8, D=32):
        super().__init__()
        self.H_world = H_world
        self.W_world = W_world
        self.H_view = H_view
        self.W_view = W_view
        self.D = D
        self.grid_mem = MovableGridMemory(H_world, W_world, D)

    def init_world(self, B, device):
        return self.grid_mem.init_grid(B, device)  # [B,H_world,W_world,D]

    def crop_and_downsample(self, world: torch.Tensor, pos: torch.Tensor, zoom: int, quant: int):
        """
        world: [B,H_world,W_world,D]
        pos: [B,2] (x,y) center
        zoom: int 1,2,4,8 – coverage factor
        quant: int 1,2,4 – value quant factor (for now spatial extra downsample, then upsample back? We'll implement value quant as embedding binning later – for now quant as extra spatial downsample then repeat)
        Returns: viewport [B,H_view,W_view,D]
        """
        B, Hw, Ww, D = world.shape
        Hv, Wv = self.H_view, self.W_view

        # compute crop size
        crop_h = Hv * zoom
        crop_w = Wv * zoom

        # pos is (x,y) center
        x = pos[:, 0]
        y = pos[:, 1]

        # top-left
        x1 = (x - crop_w // 2).clamp(0, Ww - crop_w).long()
        y1 = (y - crop_h // 2).clamp(0, Hw - crop_h).long()

        viewports = []
        for b in range(B):
            # crop
            patch = world[b, y1[b]:y1[b]+crop_h, x1[b]:x1[b]+crop_w, :]  # [crop_h, crop_w, D]
            # if patch smaller than crop_h/crop_w due to edge, pad?
            if patch.shape[0] < crop_h or patch.shape[1] < crop_w:
                # pad zeros
                ph, pw = patch.shape[0], patch.shape[1]
                padded = torch.zeros(crop_h, crop_w, D, device=world.device, dtype=world.dtype)
                padded[:ph, :pw, :] = patch
                patch = padded

            # downsample by zoom (mean pool zoom×zoom) to Hv×Wv
            # patch [crop_h, crop_w, D] -> [Hv, Wv, D] via avg pool
            # reshape: crop_h = Hv*zoom, crop_w = Wv*zoom
            # We can do: patch.view(Hv, zoom, Wv, zoom, D).mean(1,3)
            try:
                patch_reshaped = patch.view(Hv, zoom, Wv, zoom, D)
                down = patch_reshaped.mean(dim=1).mean(dim=2)  # [Hv, Wv, D] ??? Let's do stepwise
                # Actually view as [Hv, zoom, Wv, zoom, D]
                # Mean over zoom dimensions
                # patch.view(Hv, zoom, Wv, zoom, D) -> mean dim 1 and 3
                # Need to check: crop_h = Hv*zoom, so first dim Hv*zoom = Hv*zoom, we want to split into Hv groups of zoom
                # So patch [Hv*zoom, Wv*zoom, D] -> [Hv, zoom, Wv, zoom, D]
            except:
                # fallback: use adaptive avg pool via interpolation
                # Convert to [D, H, W] for pooling
                patch_trans = patch.permute(2,0,1).unsqueeze(0)  # [1,D,crop_h,crop_w]
                down_trans = F.adaptive_avg_pool2d(patch_trans, (Hv, Wv))  # [1,D,Hv,Wv]
                down = down_trans.squeeze(0).permute(1,2,0)  # [Hv,Wv,D]
                viewports.append(down)
                continue

            # Correct downsample via view
            # patch [Hv*zoom, Wv*zoom, D]
            # We want [Hv, Wv, D] where each cell is mean of zoom×zoom block
            # Reshape to [Hv, zoom, Wv, zoom, D]
            patch_view = patch.view(Hv, zoom, Wv, zoom, D)
            down = patch_view.mean(dim=1).mean(dim=2)  # [Hv, Wv, D] after mean over both zoom dims? Let's do mean dim=1 and then dim=2 (which after first mean becomes dim=2)
            # Actually after first mean dim=1: [Hv, Wv, zoom, D], after second mean dim=2: [Hv, Wv, D]
            # We did mean dim=1 then mean dim=2 – need to adjust
            # Let's do two means sequentially
            # First mean over zoom at dim=1
            # patch_view [Hv, zoom, Wv, zoom, D]
            down1 = patch_view.mean(dim=1)  # [Hv, Wv, zoom, D]
            down2 = down1.mean(dim=2)  # [Hv, Wv, D]
            viewports.append(down2)

        viewport = torch.stack(viewports, dim=0)  # [B,Hv,Wv,D]

        # Quantization as value binning: if quant>1, we quantize embedding values to coarser bins
        # For simplicity, we simulate by adding noise or rounding: viewport = round(viewport * (1/quant)) * quant ? 
        # Or for symbol case where viewport stores embeddings that will be classified, we can map via quantize: if quant=4, we bin D dim? Simpler: mean pool quant×quant again then repeat?
        # For now, implement quant as extra spatial downsample then nearest upsample back to Hv×Wv (loses detail)
        if quant > 1:
            # Downsample Hv×Wv by quant then upsample back via repeat (loses detail)
            # e.g., Hv=8, quant=2 => down to 4×4 then upsample to 8×8 via repeat
            B_, Hv_, Wv_, D_ = viewport.shape
            # ensure divisible
            if Hv_ % quant == 0 and Wv_ % quant == 0:
                # mean pool quant×quant
                vp_view = viewport.view(B_, Hv_//quant, quant, Wv_//quant, quant, D_)
                vp_mean = vp_view.mean(dim=2).mean(dim=3)  # [B, Hv//q, Wv//q, D]
                # upsample by repeat
                vp_up = vp_mean.repeat_interleave(quant, dim=1).repeat_interleave(quant, dim=2)  # [B,Hv,Wv,D]
                viewport = vp_up

        return viewport

    def read_viewport(self, world: torch.Tensor, pos: torch.Tensor, zoom: int, quant: int):
        return self.crop_and_downsample(world, pos, zoom, quant)

    def write_world(self, world: torch.Tensor, pos: torch.Tensor, w: torch.Tensor, g: torch.Tensor):
        # write to world at pos (single cell) – same as movable grid
        return self.grid_mem.write(world, pos, w, g)


class ViewportModel(nn.Module):
    def __init__(self, H_world=32, W_world=32, H_view=8, W_view=8, D=32, dim=64, num_layers=2, mode="movable", symbol_vocab=SYMBOL_VOCAB):
        super().__init__()
        self.H_world = H_world
        self.W_world = W_world
        self.H_view = H_view
        self.W_view = W_view
        self.D = D
        self.dim = dim
        self.mode = mode  # movable, fixed, zoom_fixed, etc.

        self.viewport_mem = ViewportMemory(H_world, W_world, H_view, W_view, D)
        self.rwkv = RWKVNano(vocab_size=symbol_vocab, dim=dim, num_layers=num_layers, pad_token_id=PAD_ID)

        self.sym_embed = nn.Embedding(symbol_vocab, dim)
        self.read_proj = nn.Linear(D * H_view * W_view, dim)  # read is whole viewport flattened

        self.write_head = nn.Linear(dim, D)
        self.gate_head = nn.Linear(dim, 1)
        self.move_head = nn.Linear(dim, 2)
        self.zoom_head = nn.Linear(dim, 1)  # predicts zoom logit
        self.quant_head = nn.Linear(dim, 1)
        self.qa_head = nn.Linear(D * H_view * W_view, symbol_vocab)  # for query from viewport

    def forward(self, placements, query_pos, query_zoom=1, query_quant=1, teacher_pos=None, teacher_zoom=None, teacher_quant=None):
        """
        placements: list of (sym, x, y) each [B]
        query_pos: [B,2]
        For simplicity, we process placements sequentially, updating world.
        Then at query, we read viewport at query_pos with query_zoom/quant.
        """
        B = placements[0][0].shape[0]
        device = placements[0][0].device

        world = self.viewport_mem.init_world(B, device)
        pos = torch.zeros(B, 2, dtype=torch.long, device=device)  # start 0,0
        zoom = torch.ones(B, dtype=torch.long, device=device)  # zoom=1
        quant = torch.ones(B, dtype=torch.long, device=device)

        pos_traj = [pos.clone()]
        zoom_traj = [zoom.clone()]
        quant_traj = [quant.clone()]

        rwkv_state = None

        for idx, (sym, x, y) in enumerate(placements):
            # read viewport at current pos/zoom/quant
            viewport = self.viewport_mem.read_viewport(world, pos, int(zoom[0].item()) if self.mode!='fixed' else 1, int(quant[0].item()) if self.mode!='fixed' else 1)  # [B,Hv,Wv,D]
            read_flat = viewport.view(B, -1)  # [B, Hv*Wv*D]
            read_proj = self.read_proj(read_flat)  # [B,dim]

            sym_e = self.sym_embed(sym)  # [B,dim]
            inp = sym_e + read_proj

            x_seq = inp.unsqueeze(1)
            new_state = []
            h = x_seq
            for i, block in enumerate(self.rwkv.blocks):
                ls = rwkv_state[i] if rwkv_state is not None else None
                h, s = block(h, ls)
                new_state.append(s)
            rwkv_state = new_state
            h = self.rwkv.ln_out(h).squeeze(1)  # [B,dim]

            # write
            w = self.write_head(h)
            g = torch.sigmoid(self.gate_head(h))

            # movement
            if self.mode != "fixed":
                if teacher_pos is not None:
                    pos = teacher_pos[idx]
                    if teacher_zoom is not None:
                        zoom = teacher_zoom[idx]
                    if teacher_quant is not None:
                        quant = teacher_quant[idx]
                else:
                    delta = torch.tanh(self.move_head(h))
                    delta_int = torch.round(delta).long()
                    pos = pos + delta_int
                    pos[:, 0] = pos[:, 0].clamp(0, self.W_world-1)
                    pos[:, 1] = pos[:, 1].clamp(0, self.H_world-1)
                    # zoom
                    if self.mode in ("zoom_control", "combined"):
                        dz_logit = self.zoom_head(h).squeeze(-1)
                        # map to zoom values 1,2,4
                        # simple: if dz_logit >0.5 -> zoom=1, else zoom=4? For first proof, we can have binary
                        # We'll map sigmoid to 1 or 4
                        zoom_pred = (torch.sigmoid(dz_logit) > 0.5).long() * 3 + 1  # 1 or 4
                        zoom = zoom_pred.clamp(1, 4)
                    if self.mode in ("quant_control", "combined"):
                        dq_logit = self.quant_head(h).squeeze(-1)
                        quant_pred = (torch.sigmoid(dq_logit) > 0.5).long() * 3 + 1
                        quant = quant_pred.clamp(1, 4)

            # write to world at described position (x,y) of placement
            if self.mode == "fixed":
                write_pos = torch.zeros(B, 2, dtype=torch.long, device=device)
            else:
                write_pos = torch.stack([x, y], dim=1)

            world = self.viewport_mem.write_world(world, write_pos, w, g)

            pos_traj.append(pos.clone())
            zoom_traj.append(zoom.clone())
            quant_traj.append(quant.clone())

        # query: read viewport at query_pos with given zoom/quant
        q_viewport = self.viewport_mem.read_viewport(world, query_pos, query_zoom, query_quant)
        q_flat = q_viewport.view(B, -1)
        logits = self.qa_head(q_flat)

        return logits, world, pos_traj, zoom_traj, quant_traj, {"final_pos": pos, "final_zoom": zoom, "final_quant": quant}


def generate_viewport_task(B=4, H_world=32, W_world=32, num_placements=5, device="cpu", hidden_symbol_pos=None):
    """Generate task with one hidden symbol at random pos (for VZ1)."""
    import random
    world_true = torch.zeros(B, H_world, W_world, dtype=torch.long, device=device)
    placements = []
    # place a hidden symbol that is the query target, at random pos
    if hidden_symbol_pos is None:
        hx = torch.randint(0, W_world, (B,), device=device)
        hy = torch.randint(0, H_world, (B,), device=device)
    else:
        hx, hy = hidden_symbol_pos

    hidden_sym = torch.randint(1, 10, (B,), device=device)  # symbol 1-9

    # For VZ1, we place hidden symbol plus some distractors
    for _ in range(num_placements-1):
        sym = torch.randint(1, 20, (B,), device=device)
        x = torch.randint(0, W_world, (B,), device=device)
        y = torch.randint(0, H_world, (B,), device=device)
        placements.append((sym, x, y))
        for b in range(B):
            world_true[b, y[b], x[b]] = sym[b]

    # hidden placement last
    placements.append((hidden_sym, hx, hy))
    for b in range(B):
        world_true[b, hy[b], hx[b]] = hidden_sym[b]

    # query pos = hidden pos, query sym = hidden sym
    query_pos = torch.stack([hx, hy], dim=1)
    query_sym = hidden_sym

    # teacher trajectory: move towards hidden pos? For simplicity, teacher pos = placements pos sequence
    teacher_pos = []
    for (_, x, y) in placements:
        teacher_pos.append(torch.stack([x, y], dim=1))

    return world_true, placements, query_pos, query_sym, teacher_pos

if __name__ == "__main__":
    B=2
    model = ViewportModel(H_world=32, W_world=32, H_view=8, W_view=8, D=16, dim=32, num_layers=2, mode="movable")
    print(f"params {count_params(model):,}")
    world_true, placements, qpos, qsym, teacher_pos = generate_viewport_task(B=B, H_world=32, W_world=32, num_placements=3, device="cpu")
    logits, world, pos_traj, zoom_traj, quant_traj, extra = model(placements, qpos, query_zoom=1, query_quant=1, teacher_pos=teacher_pos)
    print(f"logits {logits.shape} world {world.shape} pos_traj {len(pos_traj)}")
    pred = logits.argmax(dim=-1)
    print(f"pred {pred} true {qsym} acc {(pred==qsym).float().mean()}")
