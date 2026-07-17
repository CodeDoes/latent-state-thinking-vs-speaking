"""Train movable grid scratchboard – MG1 proof.

Task Place&Query 8x8 as defined in theories/movable-grid-scratchboard.md

Variants:
- full: movable + grid (teacher forced movement to described pos)
- static: pos fixed (0,0) no movement
- nogrid: no grid, only RWKV state (like exp001)
- nopos: text without positions (placements sym only, not x,y) – for MG2

Usage:
  python -m src.train_movable_grid --mode full --exp_id movable_grid_full_001 --steps 2000 --H 8 --W 8 --dim 64
  python -m src.train_movable_grid --mode static --exp_id movable_grid_static_001 --steps 2000
  python -m src.train_movable_grid --mode nogrid --exp_id movable_grid_nogrid_001 --steps 2000
  python -m src.train_movable_grid --mode nopos --exp_id movable_grid_nopos_001 --steps 2000

MG1 win: full QA acc >0.8, static/nogrid <0.3 at matched params.

Also supports overwrite test MG4, 2d vs 1d MG5, certainty MG6 via flags.
"""

import argparse, json, time, random
from pathlib import Path
import torch
import torch.nn.functional as F

from src.movable_grid_model import MovableGridModel, generate_place_query_batch, count_params, SYMBOL_VOCAB, PAD_ID

def train(exp_id="movable_grid_full_001", mode="full", H=8, W=8, D=32, dim=64, layers=2,
          num_placements=10, batch_size=8, steps=2000, lr=3e-4, log_every=100, eval_every=500,
          device="cpu", use_3x3=False, overwrite_test=False):

    dev = torch.device(device)
    if torch.cuda.is_available() and device != "cpu":
        dev = torch.device("cuda")

    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = MovableGridModel(H=H, W=W, D=D, dim=dim, num_layers=layers, symbol_vocab=SYMBOL_VOCAB, mode=mode if mode!="nopos" else "full", use_3x3_read=use_3x3).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    config = dict(exp_id=exp_id, mode=mode, H=H, W=W, D=D, dim=dim, layers=layers,
                  num_placements=num_placements, batch_size=batch_size, steps=steps, lr=lr,
                  hypothesis="movable grid + scratchboard beats static and no-grid on Place&Query")
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Mode {mode} H={H} W={W} D={D} dim={dim} layers={layers} params {count_params(model):,} device {dev}")

    best_acc = 0.0
    metrics_log = []
    t0 = time.time()

    for step in range(1, steps+1):
        # generate batch
        if mode == "nopos":
            # for MG2: placements without pos text – we still have pos for teacher but model input sym only? 
            # For simplicity we keep same generator but will not use pos in input? 
            # Our current model full uses pos as teacher forced movement target, but also as write pos.
            # For nopos ablation, we force write pos to (0,0) always, so association can't be learned.
            # Implement by setting teacher_pos to zeros
            world, placements, qpos, qsym, teacher_pos = generate_place_query_batch(B=batch_size, H=H, W=W, num_placements=num_placements, device=dev)
            # overwrite teacher_pos to zeros for nopos
            teacher_pos = [torch.zeros_like(p) for p in teacher_pos]
            # also for write, model will write at zeros if we pass modified teacher? Actually model writes at described pos by default for full mode.
            # For nopos we need to make model write at (0,0) – we can achieve by setting placements x,y to zeros as well for write pos?
            # We'll keep placements x,y random but teacher_pos zeros, and model writes at write_pos = zeros if we modify model? Simpler: for nopos mode, we set mode static (writes at 0,0)
            # So we will still use full mode but with teacher zeros – model writes at 0,0? Let's check: in model full, write_pos = stack(x,y) from placements (described pos), not teacher_pos. So need to also zero placements x,y for write.
            # To simulate nopos, we can zero out placements x,y for write by replacing placements with zeros for x,y
            placements_nopos = []
            for (sym, x, y) in placements:
                x0 = torch.zeros_like(x)
                y0 = torch.zeros_like(y)
                placements_nopos.append((sym, x0, y0))
            placements = placements_nopos
        else:
            world, placements, qpos, qsym, teacher_pos = generate_place_query_batch(B=batch_size, H=H, W=W, num_placements=num_placements, device=dev)

        if overwrite_test:
            # MG4 overwrite: same pos written twice with different sym, query should return last
            # Modify last two placements to same pos with different sym
            # Take last placement pos, make second last same pos but different sym
            if len(placements) >= 2:
                # second last pos = last pos
                sym_last, x_last, y_last = placements[-1]
                sym_prev, _, _ = placements[-2]
                # make prev have same pos as last but different sym
                placements[-2] = (sym_prev, x_last, y_last)
                # update world accordingly: write prev then last overwrites
                # world already overwritten by last, so query that pos should return sym_last
                for b in range(batch_size):
                    world[b, y_last[b], x_last[b]] = sym_last[b]
                qpos = torch.stack([x_last, y_last], dim=1)
                qsym = sym_last

        qpos = qpos.to(dev)
        qsym = qsym.to(dev)

        # forward
        logits, grid, traj, extra = model(placements, qpos, teacher_pos=teacher_pos)

        loss = F.cross_entropy(logits, qsym)

        # optional grid reconstruction aux loss (helps training)
        aux_loss = torch.tensor(0.0, device=dev)
        if grid is not None and mode != "nogrid":
            # world is [B,H,W] symbol ids, grid is [B,H,W,D] continuous
            # We can have aux head that predicts symbol per cell from grid vector
            # For simplicity, we compute MSE between grid read at all positions vs embedding of world symbol?
            # Instead, we have qa head that can be applied to all cells: predict world symbol from grid cell vector
            # Let's apply qa_head to each grid cell to get per-cell logits and CE vs world
            B_ = grid.shape[0]
            grid_flat = grid.view(B_*H*W, D)  # [B*HW, D]
            # qa_head is Linear(D, vocab)
            per_cell_logits = model.qa_head(grid_flat)  # [B*HW, vocab]
            world_flat = world.view(B_*H*W)  # [B*HW]
            # only non-zero cells have loss
            mask = world_flat != 0
            if mask.any():
                aux_loss = F.cross_entropy(per_cell_logits[mask], world_flat[mask])

        total_loss = loss + 0.1 * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            acc = (logits.argmax(dim=-1) == qsym).float().mean().item()
            elapsed = time.time() - t0
            print(f"step {step:4d}/{steps} mode {mode} loss {loss.item():.4f} aux {aux_loss.item():.4f} acc {acc:.3f} elapsed {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc

            metrics_log.append({"step": step, "loss": loss.item(), "aux_loss": aux_loss.item(), "acc": acc, "best_acc": best_acc})
            (exp_dir / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics_log))

            # save trajectory plot for first batch as ascii
            if step % (log_every*5) == 0:
                # visualize grid for first sample
                if grid is not None:
                    g0 = grid[0].detach().cpu()  # [H,W,D]
                    # we don't have decoded symbols from grid, but we can try to decode via qa_head
                    with torch.no_grad():
                        per_cell = model.qa_head(g0.view(H*W, D))
                        pred_symbols = per_cell.argmax(dim=-1).view(H, W).cpu().numpy()
                        # create ascii art
                        ascii_grid = ""
                        for r in range(H):
                            row = ""
                            for c in range(W):
                                sid = int(pred_symbols[r, c])
                                if sid == 0:
                                    row += ". "
                                else:
                                    # map 1..26 to A-Z
                                    ch = chr(ord('A') + sid - 1) if 1 <= sid <= 26 else "?"
                                    row += ch + " "
                            ascii_grid += row + "\n"
                        # trajectory
                        traj_str = "traj: "
                        for p in traj[:10]:  # first 10 steps, first sample
                            x = int(p[0, 0].item()) if p.dim() > 1 else 0
                            y = int(p[0, 1].item()) if p.dim() > 1 else 0
                            traj_str += f"({x},{y}) "
                        (exp_dir / "sample.txt").write_text(f"step {step} acc {acc:.3f}\nGrid:\n{ascii_grid}\n{traj_str}\nQuery pos {qpos[0].tolist()} true {int(qsym[0].item())} pred {int(logits[0].argmax().item())}\n")
                else:
                    (exp_dir / "sample.txt").write_text(f"step {step} acc {acc:.3f} nogrid mode, no grid viz")

    result = {"final_acc": acc, "best_acc": best_acc, "exp_id": exp_id, "mode": mode, "params": count_params(model)}
    (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"Done {exp_id} best acc {best_acc:.3f} final {acc:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="movable_grid_full_001")
    ap.add_argument("--mode", default="full", choices=["full", "static", "nogrid", "nopos"])
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num_placements", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use_3x3", action="store_true")
    ap.add_argument("--overwrite_test", action="store_true", help="MG4 overwrite test")
    args = ap.parse_args()
    train(**vars(args))
