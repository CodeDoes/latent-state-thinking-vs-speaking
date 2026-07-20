#!/usr/bin/env python3
"""Train BlackGoose channel-mix (single Linear) on frozen g1g 2.9B NF4.

Usage: python src/train_g1g_blackgoose.py --layers 0 --steps 20 --lr 3e-4
"""

import sys, os, time, argparse
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from domains.g1g.g1g_blackgoose_nf4 import G1GBlackGooseNF4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0", help="comma-sep layer indices")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    model = G1GBlackGooseNF4(layers_to_replace=layers, device="cuda")
    model.train()
    opt = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr)
    print(f"Trainable: {sum(p.numel() for p in model.get_trainable_params())/1e6:.2f}M")

    # Tiny text corpus: Alice in Wonderland excerpt
    text = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, "
        "and of having nothing to do. once or twice she had peeped into the book her "
        "sister was reading, but it had no pictures or conversations in it, "
        "and where is the use of a book, thought Alice without pictures or conversation? "
        "So she was considering in her own mind (as well as she could, for the hot day "
        "made her feel very sleepy and stupid), whether the pleasure of making a "
        "daisy-chain would be worth the trouble of getting up and picking the daisies, "
        "when suddenly a White Rabbit with pink eyes ran close by her."
    )
    raw = text.encode("utf-8")
    raw = bytes(b for b in raw if b >= 2)  # drop reserved bytes
    seqlen = 16
    stride = 8
    xs, ys = [], []
    for i in range(0, len(raw) - seqlen, stride):
        xs.append(torch.tensor(list(raw[i:i+seqlen]), dtype=torch.long))
        ys.append(torch.tensor(list(raw[i+1:i+seqlen+1]), dtype=torch.long))
    xs = torch.stack(xs)  # (N, 16)
    ys = torch.stack(ys)
    print(f"Data: {xs.shape[0]} sequences of {seqlen} bytes")

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, xs.shape[0], ())
        x = xs[idx:idx+1].cuda()
        y = ys[idx:idx+1].cuda()

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 258), y.view(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
        opt.step()

        mem = torch.cuda.memory_allocated() / 1e9
        dt = time.time() - t0
        rate = (step + 1) / dt if dt > 0 else 0
        print(f"step {step:3d}/{args.steps}  loss={loss.item():.4f}  GPU={mem:.2f}GB  {dt:.0f}s  {rate:.3f}step/s")

    final = loss.item()
    print(f"\nDone! Final loss={final:.4f}")


if __name__ == "__main__":
    main()
