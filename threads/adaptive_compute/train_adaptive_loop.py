"""Training script for AdaptiveLoopModel.

Usage:
    python -m src.train_adaptive_loop --steps 500 --dim 64 --patch-size 4

Same setup as other experiments: TinyStories byte-level, small model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
import time
import argparse

from threads.adaptive_compute.adaptive_loop_model import AdaptiveLoopModel
from domains.byte.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, UNK_ID


def load_text(path: Path) -> list[int]:
    text = path.read_bytes().decode("utf-8", errors="replace")
    return [BYTE_TO_ID.get(ord(c), UNK_ID) for c in text]


def make_batches(stream: list[int], max_len: int, batch_size: int):
    n = (len(stream) - 1) // max_len * max_len
    stream = stream[: n + 1]
    while True:
        for start in range(0, len(stream) - max_len - 1, batch_size * max_len):
            rows = []
            for i in range(batch_size):
                chunk = stream[start + i * max_len : start + (i + 1) * max_len + 1]
                if len(chunk) < max_len + 1:
                    continue
                rows.append(chunk)
            if not rows:
                continue
            for r in rows:
                while len(r) < max_len + 1:
                    r.append(PAD_ID)
            batch = torch.tensor(rows, dtype=torch.long)
            yield batch[:, :-1], batch[:, 1:].contiguous()


def train(
    steps: int = 500,
    batch_size: int = 8,
    max_len: int = 128,
    lr: float = 3e-4,
    dim: int = 64,
    patch_size: int = 4,
    enc_layers: int = 2,
    core_layers: int = 2,
    dec_layers: int = 2,
    enc_max_loops: int = 3,
    core_depth_loops: int = 2,
    dec_max_loops: int = 3,
    log_every: int = 50,
    dynamic_patch: bool = False,
    patch_threshold: float = 0.7,
    min_patch: int = 2,
    max_patch_val: int = 16,
    text_path: str = "threads/g1g_frontend/experiments/byte_ts_001/text.txt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stream = load_text(Path(text_path))
    print(f"Loaded {len(stream):,} bytes")

    model = AdaptiveLoopModel(
        dim=dim,
        patch_size=patch_size,
        enc_layers=enc_layers,
        core_layers=core_layers,
        dec_layers=dec_layers,
        enc_max_loops=enc_max_loops,
        core_depth_loops=core_depth_loops,
        dec_max_loops=dec_max_loops,
        dynamic_patch=dynamic_patch,
        patch_threshold=patch_threshold,
        min_patch=min_patch,
        max_patch=max_patch_val,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"  encoder:  {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  core:     {sum(p.numel() for p in model.core.parameters()):,}")
    print(f"  decoder:  {sum(p.numel() for p in model.decoder.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batch_iter = make_batches(stream, max_len, batch_size)

    model.train()
    t0 = time.time()

    for step in range(steps):
        input_ids, targets = next(batch_iter)
        input_ids, targets = input_ids.to(device), targets.to(device)

        logits, info = model(input_ids)

        # Reconstruction loss
        recon_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=PAD_ID,
        )

        # Adaptive exit entropy loss (encourages exploration of loop depths)
        ent_loss = torch.tensor(0.0, device=device)
        for lam_list, name in [
            (info["core"]["exit_lambdas"], "core"),
            (info["decoder"]["exit_lambdas"], "dec"),
        ]:
            if lam_list:
                lams = torch.stack(lam_list)  # [R, B, T]
                # Entropy: -∑ π_r log π_r (maximize → subtract)
                # π_r = λ_r * ∏_{k<r} (1-λ_k)
                cum_survive = torch.ones_like(lams[0])
                entropy = torch.zeros_like(lams[0])
                for r in range(lams.shape[0]):
                    pi_r = lams[r] * cum_survive
                    entropy = entropy - pi_r * (pi_r + 1e-7).log()
                    cum_survive = cum_survive * (1 - lams[r])
                ent_loss = ent_loss - entropy.mean()  # maximize entropy

        loss = recon_loss + 0.01 * ent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - t0
            stats = model.get_exit_stats(info)
            loops_str = (
                f"enc={stats['enc_loops']} "
                f"core={stats['core_depth_loops']} "
                f"dec={stats['dec_loops']}"
            )
            print(
                f"step {step:4d} | loss {recon_loss.item():.4f} | "
                f"{loops_str} | latents {stats['n_latents']} | "
                f"ρ={stats['compression_ratio']:.1f} | {elapsed:.1f}s"
            )

    print(f"\nDone. Final loss: {recon_loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--core-layers", type=int, default=2)
    parser.add_argument("--dec-layers", type=int, default=2)
    parser.add_argument("--enc-max-loops", type=int, default=3)
    parser.add_argument("--core-depth-loops", type=int, default=2)
    parser.add_argument("--dec-max-loops", type=int, default=3)
    parser.add_argument("--dynamic-patch", action="store_true")
    parser.add_argument("--patch-threshold", type=float, default=0.7)
    parser.add_argument("--min-patch", type=int, default=2)
    parser.add_argument("--max-patch", type=int, default=16, dest="max_patch_val")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--text-path", type=str, default="threads/g1g_frontend/experiments/byte_ts_001/text.txt")
    args = parser.parse_args()
    train(**vars(args))
