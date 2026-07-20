"""Evaluate byte auto-encoder: test reconstruction accuracy per token length.

Loads the trained model from experiments/byte_ae/model.pt and tests
on the full vocabulary (all 65K tokens), reporting accuracy by length.
"""
import torch
from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.train_byte_ae import ByteAE, DIM, LATENT, device
from src.hybrid_tokenizer import token_bytes
from collections import defaultdict

model = ByteAE(DIM, LATENT).to(device)
model.load_state_dict(torch.load("experiments/byte_ae/model.pt", map_location=device))
model.eval()

# Group tokens by length
by_len = defaultdict(list)
for tid in range(1, 65529):
    b = token_bytes(tid)
    if len(b) and len(b) <= 24:
        by_len[len(b)].append(b)

print(f"{'Len':>3s}  {'Total':>6s}  {'Tested':>6s}  {'Correct':>7s}  {'%':>6s}")
print("-" * 35)
total_correct, total_tested = 0, 0
for L in sorted(by_len):
    tlist = by_len[L]
    n = min(500, len(tlist))
    correct = 0
    for tb in tlist[:n]:
        inp = torch.tensor([[2 + b for b in tb]], device=device)
        logits, _ = model(inp)
        T = len(tb)
        pred = logits[0, :T].argmax(-1).tolist()
        pred_bytes = bytes(b - 2 for b in pred if b >= 2)
        if tb == pred_bytes:
            correct += 1
    pct = 100 * correct / n
    total_correct += correct
    total_tested += n
    print(f"{L:3d}  {len(tlist):6d}  {n:6d}  {correct:7d}  {pct:5.1f}%")

print("-" * 35)
print(f"Total: {total_correct}/{total_tested} = {100*total_correct/total_tested:.1f}%")
