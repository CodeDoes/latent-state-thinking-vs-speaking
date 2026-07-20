#!/usr/bin/env python3
"""
Train BlackGoose channel-mix (single Linear) to match original FFN output.

Trains offline on pre-generated (ln2, ffn_output) pairs — no frozen backbone needed.
Usage:
    python src/train_blackgoose_offline.py --data experiments/blackgoose_data/layer_0.pt --steps 500
"""

import sys, os, time, argparse
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class BlackGooseChannelMix(nn.Module):
    """Single linear layer replacing the entire RWKV-7 channel-mix FFN."""
    def __init__(self, dim: int):
        super().__init__()
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.value(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to .pt file with {'inputs', 'targets'}")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval", action="store_true",
                        help="Eval mode: load trained model and measure MSE")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load checkpoint for eval or resume")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    inputs = data["inputs"]  # (N, dim)
    targets = data["targets"]
    dim = inputs.shape[-1]
    N = inputs.shape[0]
    print(f"Data: {N} samples, dim={dim}")

    # Split
    split = int(N * 0.9)
    train_inp, val_inp = inputs[:split], inputs[split:]
    train_tgt, val_tgt = targets[:split], targets[split:]
    print(f"Train: {train_inp.shape[0]}, Val: {val_inp.shape[0]}")

    # ── Model ─────────────────────────────────────────────────────────
    dtype = inputs.dtype
    model = BlackGooseChannelMix(dim).to(dtype=dtype)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.eval:
        model.eval()
        with torch.no_grad():
            pred = model(val_inp)
            mse = (pred - val_tgt).pow(2).mean().item()
            cos = nn.functional.cosine_similarity(pred, val_tgt, dim=-1).mean().item()
        print(f"Eval: MSE={mse:.8f}, CosSim={cos:.6f}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    train_inp, train_tgt = train_inp.to(device), train_tgt.to(device)
    val_inp, val_tgt = val_inp.to(device), val_tgt.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_batches = max(1, train_inp.shape[0] // args.batch)

    t0 = time.time()
    for step in range(args.steps):
        # Random batch
        idx = torch.randint(0, train_inp.shape[0], (args.batch,))
        inp = train_inp[idx]
        tgt = train_tgt[idx]

        pred = model(inp)
        loss = (pred - tgt).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        dt = time.time() - t0
        rate = (step + 1) / dt if dt > 0 else 0

        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                val_pred = model(val_inp)
                val_mse = (val_pred - val_tgt).pow(2).mean().item()
                cos_sim = nn.functional.cosine_similarity(val_pred, val_tgt, dim=-1).mean().item()
            print(
                f"step {step:5d}/{args.steps}  "
                f"train_loss={loss.item():.8f}  "
                f"val_mse={val_mse:.8f}  "
                f"cos_sim={cos_sim:.6f}  "
                f"{dt:.0f}s  {rate:.0f}step/s"
            )

    # Final eval
    model.eval()
    with torch.no_grad():
        pred = model(val_inp)
        mse = (pred - val_tgt).pow(2).mean().item()
        cos = nn.functional.cosine_similarity(pred, val_tgt, dim=-1).mean().item()
    print(f"\nFinal: val_mse={mse:.8f}, cos_sim={cos:.6f}")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Save
    out_path = Path(args.data).with_suffix("")  # strip .pt
    save_name = f"{out_path.name}_trained.pt"
    torch.save(model.state_dict(), save_name)
    print(f"Saved: {save_name}")


if __name__ == "__main__":
    main()
