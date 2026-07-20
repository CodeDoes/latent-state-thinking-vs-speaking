"""Streaming tokenizer: bytes fill from RIGHT, trigger at position 23.

Training: tokens placed right-aligned in 24-byte grid. Position 23 = last byte.
The conv learns: does position 23 look like a token boundary?

Streaming: new byte → shift grid left → insert at position 23 → check trigger.
If trigger[23] > 0.5, emit the accumulated token and reset.
"""
import torch, torch.nn.functional as F, time, random
from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER

DEVICE = "cuda"; LR = 3e-3; STEPS = 3000; LOG_EVERY = 500
BATCH = 2048; MAX_BYTES = 24
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
    def trig_at_23(self, bits):
        """Return trigger probability at position 23 only."""
        return self.forward(bits)[:, 23]

# ── Data: tokens right-aligned + prefix negatives ──
with open("experiments/tinystories_texts.txt") as f:
    stories = f.read().split("\n---END---\n")

items = []  # (byte_mat_row, label)  label=1 for complete token, 0 for prefix
for s in stories[:100]:
    raw = s.encode("utf-8")[:4096]
    idx = 0
    while idx < len(raw):
        old = idx
        idx, node, vals = tok.root.find_longest(raw, idx)
        if idx == old: idx += 1; continue
        tb = raw[old:idx]
        L = len(tb)
        if 0 < L <= MAX_BYTES:
            # Positive: full token right-aligned
            row = [0] * (MAX_BYTES - L) + list(tb)
            items.append((row, 1))
            # Negatives: each prefix of the token (right-aligned, shorter)
            for prefix_len in range(1, L):
                prefix = tb[-prefix_len:]  # take last prefix_len bytes
                prefix_row = [0] * (MAX_BYTES - prefix_len) + list(prefix)
                items.append((prefix_row, 0))

print(f"Items: {len(items):,} (pos: {sum(1 for _,l in items if l==1)}, neg: {sum(1 for _,l in items if l==0)})", flush=True)

B = len(items)
byte_mat = torch.tensor([row for row, _ in items], dtype=torch.uint8)
trig_target = torch.tensor([l for _, l in items], dtype=torch.float).unsqueeze(1)
byte_mat = byte_mat.to(device)
trig_target = trig_target.to(device)
bits = ((byte_mat.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).float()

model = Tokenizer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

t0 = time.time()
for step in range(STEPS):
    idx = torch.randperm(B)[:BATCH]
    pred = model.forward(bits[idx])  # (B, 24)
    trig_23 = pred[:, 23]  # trigger at position 23
    
    # BCE: position 23 should fire for complete tokens
    loss = F.binary_cross_entropy(trig_23, trig_target[idx].squeeze())
    
    opt.zero_grad(); loss.backward(); opt.step()
    
    if (step + 1) % LOG_EVERY == 0:
        sps = (step + 1) / (time.time() - t0)
        acc = (trig_23.round() == trig_target[idx].squeeze()).float().mean().item()
        print(f"step {step+1:5d}  loss={loss.item():.4f}  acc={acc:.3f}  {sps:.1f} st/s", flush=True)
    if (step + 1) % 2000 == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(SAVE_PATH))

# ── Streaming inference ──
print(f"\n{'='*50}", flush=True)
print("Streaming inference:", flush=True)
test_bytes = b"Hello World! This is a test of the streaming tokenizer."

buf = bytearray()
tokens_out = []
for b in test_bytes:
    buf.append(b)  # grows leftward; we'll right-align for the model
    L = len(buf)
    if L > 24:  # shouldn't happen for valid tokens
        tokens_out.append(bytes(buf)); buf.clear(); continue
    
    # Build right-aligned grid: token at positions (24-L)..23
    inp = torch.zeros(1, 24, dtype=torch.long, device=device)
    start = 24 - L
    for j, x in enumerate(buf): inp[0, start + j] = x
    
    bits_inp = ((inp.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).float()
    trig_23 = model.trig_at_23(bits_inp).item()
    
    if trig_23 > 0.5:
        tokens_out.append(bytes(buf)); buf.clear()

if buf: tokens_out.append(bytes(buf))

for t in tokens_out:
    print(f"  {t!r}", flush=True)

# Compare with real
real_t = tok.encodeBytes(test_bytes)
real_str = [tok.decodeBytes([t]).decode('utf-8', errors='replace') for t in real_t]
print(f"\nReal ({len(real_t)} tok): {real_str}", flush=True)
print(f"Our model ({len(tokens_out)} tok)", flush=True)
print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
