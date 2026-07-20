"""Sliding-window tokenizer: 24-byte windows, predict first-token boundary.

Training: for each 24-byte window aligned to token starts, the model
predicts which position ends the first token. Window then slides past
the predicted boundary (like inference).
"""
import torch, torch.nn.functional as F, time, random, pickle
from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER

DEVICE = "cuda"; LR = 3e-3; STEPS = 5000; LOG_EVERY = 500
BATCH = 2048; MAX_BYTES = 24
SAVE_PATH = "experiments/streaming_tokenizer/sliding_model.pt"

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

# ── Build sliding-window training data ──
with open("experiments/tinystories_texts.txt") as f:
    stories = f.read().split("\n---END---\n")

cache_path = "experiments/streaming_tokenizer/windows_cache.pkl"
if Path(cache_path).exists():
    print("Loading cached windows...", flush=True)
    with open(cache_path, "rb") as f:
        windows = pickle.load(f)
    print(f"Loaded {len(windows):,} windows", flush=True)
else:
    windows = []
    for si, s in enumerate(stories[:100]):
        if si % 50 == 0: print(f"  story {si}/100", flush=True)
        raw = s.encode("utf-8")[:4096]
        if len(raw) < 3: continue
        boundaries = set()
        idx = 0
        while idx < len(raw):
            old = idx
            idx, node, vals = tok.root.find_longest(raw, idx)
            if idx == old: idx += 1; continue
            boundaries.add(idx - 1)
        for start in range(max(0, len(raw) - MAX_BYTES + 1)):
            win = raw[start:start + MAX_BYTES]
            bps = [p for p in range(MAX_BYTES) if (start + p) in boundaries]
            if bps:
                windows.append((win, bps[0]))
    print(f"Windows: {len(windows):,}", flush=True)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(windows, f)

B = len(windows)
print(f"Building tensors ({B:,} windows)...", flush=True)
byte_mat = torch.tensor([[b for b in win] for win, _ in windows], dtype=torch.long, device='cpu')
trig_target = torch.zeros(B, MAX_BYTES)
for i, (_, pos) in enumerate(windows):
    trig_target[i, pos] = 1.0
byte_mat = byte_mat.to(device)
trig_target = trig_target.to(device)
bits = ((byte_mat.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).float()
print(f"Ready: {byte_mat.shape}, {bits.shape}", flush=True)

model = Tokenizer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

t0 = time.time()
for step in range(STEPS):
    idx = torch.randperm(B)[:BATCH]
    pred = model(bits[idx])
    target = trig_target[idx]
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    margin = torch.relu(pred[target < 0.5] - 0.1).mean()
    loss = bce.mean() + margin
    
    opt.zero_grad(); loss.backward(); opt.step()
    
    if (step + 1) % LOG_EVERY == 0:
        sps = (step + 1) / (time.time() - t0)
        pos_acc = (pred.argmax(1) == target.argmax(1)).float().mean().item()
        print(f"step {step+1:5d}  loss={loss.item():.4f}  pos_acc={pos_acc:.3f}  {sps:.1f} st/s", flush=True)
    if (step + 1) % 2000 == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(SAVE_PATH))

# ── Inference test ──
print(f"\n{'='*50}", flush=True)
print("Inference:", flush=True)
test_bytes = b"Hello World! This is a test of sliding window tokenization."
print(f"Input: {test_bytes.decode()}", flush=True)

buf = bytearray()
tokens = []
for b in test_bytes:
    buf.append(b)
    if len(buf) >= 24:
        inp = torch.zeros(1, 24, dtype=torch.long, device=device)
        for j, x in enumerate(buf[:24]): inp[0, j] = x
        bits_inp = ((inp.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).float()
        boundary = model(bits_inp).squeeze(0).argmax().item()
        token_end = boundary + 1
        if token_end <= len(buf) and token_end <= 24:
            tokens.append(bytes(buf[:token_end]))
            buf = buf[token_end:]
if buf: tokens.append(bytes(buf))

for t in tokens:
    print(f"  {t!r}", flush=True)

# Compare with real
real_tokens = tok.encodeBytes(test_bytes)
print(f"\\nReal tokenizer: {len(real_tokens)} tokens", flush=True)
print(f"Our model: {len(tokens)} tokens", flush=True)
print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
