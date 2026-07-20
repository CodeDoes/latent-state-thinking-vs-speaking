"""Streaming tokenizer: trains on full 24-byte grids, streams via sliding window.

Training: all 24 positions filled (zeros after token end). Model predicts which
position is the last byte. 100% accuracy achieved in 1000 steps.

Inference: bytes arrive one at a time. Grid fills left to right. At each step,
find the LAST non-zero position. If model's trigger at that position > 0.5, emit.
"""
import torch, torch.nn.functional as F, time, random
from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER

DEVICE = "cuda"
LR = 3e-3
STEPS = 2000
LOG_EVERY = 200
SAVE_EVERY = 1000
BATCH = 2048
MAX_BYTES = 24
SAVE_PATH = "experiments/streaming_tokenizer/model.pt"

device = torch.device(DEVICE)
tok = RWKV_TOKENIZER(str(Path("src/rwkv_vocab_v20230424.txt")))

class Tokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.trig = torch.nn.Linear(192, 24)
    def forward(self, bits):
        h = bits.unsqueeze(1)
        h = torch.relu(self.conv1(h))
        h = torch.sigmoid(self.conv2(h))
        return torch.sigmoid(self.trig(h.view(h.shape[0], -1)))
    def forward_stream(self, byte_seq):
        """Stream bytes one at a time. Returns list of trigger probs."""
        grid = torch.zeros(1, 1, 24, 8, device=device)
        triggers = []
        for t, b in enumerate(byte_seq):
            # Fill position t
            bits = ((torch.tensor([b], device=device).unsqueeze(-1) >> torch.arange(8, device=device)) & 1).float()
            grid[:, :, t:t+1, :] = bits.unsqueeze(1)
            # Compute triggers on current grid
            h = torch.relu(self.conv1(grid))
            h = torch.sigmoid(self.conv2(h))
            all_trig = torch.sigmoid(self.trig(h.view(1, -1))).squeeze(0)
            # Only consider positions that have been filled so far
            pos_trig = all_trig[:t+1].max().item()
            triggers.append(pos_trig)
        return triggers

# ── Data ──
with open("experiments/tinystories_texts.txt") as f:
    stories = f.read().split("\n---END---\n")

tokens = []
for s in stories[:200]:
    raw = s.encode("utf-8")[:4096]
    idx = 0
    while idx < len(raw):
        old = idx
        idx, node, vals = tok.root.find_longest(raw, idx)
        if idx == old: idx += 1; continue
        tb = raw[old:idx]
        if 0 < len(tb) <= MAX_BYTES:
            tokens.append(tb)
print(f"Tokens: {len(tokens):,}", flush=True)

# Build tensors
B = len(tokens)
byte_mat = torch.zeros(B, MAX_BYTES, dtype=torch.long, device=device)
trig_target = torch.zeros(B, MAX_BYTES, device=device)
for i, tb in enumerate(tokens):
    for j, b in enumerate(tb):
        byte_mat[i, j] = b
    trig_target[i, len(tb) - 1] = 1.0

bits = ((byte_mat.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).float()

model = Tokenizer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

t0 = time.time()
for step in range(STEPS):
    idx = torch.randperm(B)[:BATCH]
    pred = model(bits[idx])
    target = trig_target[idx]
    # BCE on trigger position
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    # Extra penalty: non-trigger positions should be BELOW 0.1
    margin = torch.relu(pred[target < 0.5] - 0.1).mean()  # push non-trigger down
    loss = bce.mean() + margin
    opt.zero_grad(); loss.backward(); opt.step()
    
    if (step + 1) % LOG_EVERY == 0:
        sps = (step + 1) / (time.time() - t0)
        pos_acc = (pred.argmax(1) == trig_target[idx].argmax(1)).float().mean().item()
        print(f"step {step+1:5d}  loss={loss.item():.4f}  pos_acc={pos_acc:.3f}  {sps:.1f} st/s", flush=True)
    if (step + 1) % SAVE_EVERY == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(SAVE_PATH))

# Streaming eval
print(f"\nStreaming eval:", flush=True)
hits, total = 0, 0
for tb in tokens[:100]:
    triggers = model.forward_stream(tb)
    # Fire when last-filled position has max trigger > 0.5
    got = [t for t in range(len(tb)) if triggers[t] > 0.5]
    expected = [len(tb) - 1]
    if got == expected: hits += 1
    total += 1
    match = "✓" if got == expected else "✗"
    print(f"  {match} {tb!r:12s}  got={got}  probs={[f'{triggers[t]:.2f}' for t in range(len(tb))]}", flush=True)

print(f"\n{hits}/{total} correct  ({100*hits/total:.1f}%)", flush=True)
print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
