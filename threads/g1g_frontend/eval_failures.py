"""Show which specific tokens the auto-encoder fails on, by length.

For each token length, list the failing tokens with their ID and bytes.
"""
import torch
from pathlib import Path
import sys; sys.path.insert(0, '.')
from threads.g1g_frontend.train_byte_ae import ByteAE, DIM, LATENT, device
from domains.byte.hybrid_tokenizer import token_bytes

model = ByteAE(DIM, LATENT).to(device)
model.load_state_dict(torch.load("threads/g1g_frontend/experiments/byte_ae/model.pt", map_location=device))
model.eval()

for L in [1, 2, 16, 24]:
    print(f"\n=== Length {L} failures ===")
    count = 0
    for tid in range(1, 65529):
        b = token_bytes(tid)
        if len(b) != L: continue
        inp = torch.tensor([[2 + b_ for b_ in b]], device=device)
        logits, _ = model(inp)
        pred = logits[0, :L].argmax(-1).tolist()
        pred_b = bytes(x - 2 for x in pred if x >= 2)
        if b != pred_b:
            print(f"  ID {tid:5d}: {b!r:25s} → {pred_b!r:25s}")
            count += 1
            if count >= 20: break
    if count == 0: print("  (none)")
    print(f"  Total failures: {count}")
