"""Train viewport zoom/pan/quant – VZ1-VZ6.

Proves screen viewport keeps context small while handling massive content.

Modes:
- fixed: viewport fixed at (0,0) zoom=1 quant=1, cannot move – baseline, context small but can't find hidden
- movable: can move pos, zoom=1 quant=1 – tests scroll/pan
- zoom_control: can move and zoom (1 vs 4)
- quant_control: can move and quant (1 vs 4)
- combined: move+zoom+quant

Tasks:
- VZ1: 32x32 world, hidden symbol random, viewport 8x8 – movable should find >80% vs fixed 6%
- VZ5: 100x80 file, viewport 10x80 – scroll to TODO at line 75

Usage:
  python -m src.train_viewport_zoom_pan --mode movable --exp_id viewport_movable_001 --steps 2000 --H_world 32 --W_world 32 --H_view 8 --W_view 8
  python -m src.train_viewport_zoom_pan --mode fixed --exp_id viewport_fixed_001 --steps 2000 --H_world 32 --W_world 32 --H_view 8 --W_view 8
"""

import argparse, json, time, random
from pathlib import Path
import torch
import torch.nn.functional as F

from src.viewport_zoom_pan_model import ViewportModel, generate_viewport_task, count_params, SYMBOL_VOCAB

def train(exp_id="viewport_movable_001", mode="movable", H_world=32, W_world=32, H_view=8, W_view=8, D=16,
          dim=64, layers=2, num_placements=5, batch_size=8, steps=2000, lr=3e-4, log_every=100,
          device="cpu"):

    dev = torch.device(device)
    if torch.cuda.is_available() and device != "cpu":
        dev = torch.device("cuda")

    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = ViewportModel(H_world=H_world, W_world=W_world, H_view=H_view, W_view=W_view, D=D, dim=dim, num_layers=layers, mode=mode).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    config = dict(exp_id=exp_id, mode=mode, H_world=H_world, W_world=W_world, H_view=H_view, W_view=W_view, D=D, dim=dim, layers=layers,
                  num_placements=num_placements, batch_size=batch_size, steps=steps,
                  hypothesis="movable viewport keeps context small (Hv*Wv) while handling massive world (Hw*Ww) via pan/zoom/quant")
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Mode {mode} world {H_world}x{W_world}={H_world*W_world} view {H_view}x{W_view}={H_view*W_view} params {count_params(model):,} device {dev}")
    print(f"Context budget per step: {H_view*W_view} tokens vs world {H_world*W_world} tokens – ratio {H_world*W_world/(H_view*W_view):.1f}x massive")

    best_acc = 0.0
    metrics_log = []
    t0 = time.time()

    for step in range(1, steps+1):
        world_true, placements, qpos, qsym, teacher_pos = generate_viewport_task(B=batch_size, H_world=H_world, W_world=W_world, num_placements=num_placements, device=dev)

        qpos = qpos.to(dev)
        qsym = qsym.to(dev)

        # teacher pos for movable modes: use placements pos as teacher trajectory (move towards hidden)
        # For fixed mode, teacher_pos ignored (pos fixed at 0,0)
        logits, world, pos_traj, zoom_traj, quant_traj, extra = model(placements, qpos, query_zoom=1, query_quant=1, teacher_pos=teacher_pos)

        loss = F.cross_entropy(logits, qsym)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            acc = (logits.argmax(dim=-1) == qsym).float().mean().item()
            elapsed = time.time() - t0
            if acc > best_acc:
                best_acc = acc

            # trajectory for first sample
            traj_str = ""
            for i, p in enumerate(pos_traj[:6]):
                x = int(p[0,0].item())
                y = int(p[0,1].item())
                traj_str += f"({x},{y}) "

            print(f"step {step:4d}/{steps} mode {mode} loss {loss.item():.4f} acc {acc:.3f} best {best_acc:.3f} traj {traj_str} elapsed {elapsed:.0f}s per-step ctx {H_view*W_view} world {H_world*W_world}")

            metrics_log.append({"step": step, "loss": loss.item(), "acc": acc, "best_acc": best_acc})
            (exp_dir / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics_log))

            if step % (log_every*5) == 0:
                # ascii world vs viewport for first sample
                w_true = world_true[0].cpu().numpy()
                # readable: . for empty, A-Z for symbols
                ascii_world = ""
                for r in range(min(H_world, 16)):
                    row = ""
                    for c in range(min(W_world, 16)):
                        v = int(w_true[r,c])
                        if v==0:
                            row += ". "
                        else:
                            ch = chr(ord('A')+v-1) if 1<=v<=26 else str(v%10)
                            row += ch + " "
                    ascii_world += row + "\n"
                (exp_dir / "sample.txt").write_text(f"step {step} acc {acc:.3f} mode {mode}\nWorld (16x16 crop):\n{ascii_world}\nTraj {traj_str}\nQuery pos {qpos[0].tolist()} true {int(qsym[0].item())} pred {int(logits[0].argmax().item())}\nContext budget {H_view*W_view} vs world {H_world*W_world}\n")

    result = {"final_acc": acc, "best_acc": best_acc, "exp_id": exp_id, "mode": mode, "params": count_params(model),
              "context_per_step": H_view*W_view, "world_size": H_world*W_world, "ratio": H_world*W_world/(H_view*W_view)}
    (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"Done {exp_id} best acc {best_acc:.3f} final {acc:.3f} context {H_view*W_view} vs world {H_world*W_world}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="viewport_movable_001")
    ap.add_argument("--mode", default="movable", choices=["movable","fixed","zoom_control","quant_control","combined"])
    ap.add_argument("--H_world", type=int, default=32)
    ap.add_argument("--W_world", type=int, default=32)
    ap.add_argument("--H_view", type=int, default=8)
    ap.add_argument("--W_view", type=int, default=8)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num_placements", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(**vars(args))
