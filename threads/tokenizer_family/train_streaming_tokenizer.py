"""Uint8 propagation tokenizer: 600 params, byte addition with overflow.

Each byte flows leftward: adds to its own position AND the position to its left.
uint8 overflow handles non-linearity. Only learned part: Linear(24, 24) trigger head.
"""
import torch, torch.nn.functional as F, time, pickle
from pathlib import Path
import sys; sys.path.insert(0, '.')
from domains.rwkv.hf_rwkv_tokenizer import RWKV_TOKENIZER

DEVICE = "cuda"; LR = 1e-2; STEPS = 10000; LOG_EVERY = 1000
BATCH = 4096; MAX_BYTES = 24
SAVE_PATH = "experiments/streaming_tokenizer/uint8_model.pt"
CACHE = "experiments/streaming_tokenizer/ramp_cache.pkl"

device = torch.device(DEVICE)
tok = RWKV_TOKENIZER(str(Path("domains/rwkv/rwkv_vocab_v20230424.txt")))

class PerPosTokenizer(torch.nn.Module):
    """Per-position byte lookup: each (position, byte_value) has a learned weight."""
    def __init__(self):
        super().__init__()
        self.lookup = torch.nn.EmbeddingBag(24 * 256, 2, mode='sum')
        self.head = torch.nn.Linear(2, 24)
    def forward(self, byte_ids):
        B, T = byte_ids.shape
        pos = torch.arange(T, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        idx = pos * 256 + byte_ids
        h = self.lookup(idx)
        return torch.sigmoid(self.head(h))

# ── Load data ──
with open(CACHE, "rb") as f:
    byte_mat, ramp_targets = pickle.load(f)
byte_mat = byte_mat.to(device)
ramp_targets = ramp_targets.to(device)
print(f"Data: {len(byte_mat)} windows", flush=True)

model = PerPosTokenizer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

t0 = time.time()
for step in range(STEPS):
    idx = torch.randperm(len(byte_mat))[:BATCH]
    pred = model(byte_mat[idx])
    loss = F.mse_loss(pred, ramp_targets[idx])
    
    opt.zero_grad(); loss.backward(); opt.step()
    
    if (step + 1) % LOG_EVERY == 0:
        sps = (step + 1) / (time.time() - t0)
        boundary = ramp_targets[idx].argmax(1)
        pos_acc = (pred.argmax(1) == boundary).float().mean().item()
        print(f"step {step+1:5d}  loss={loss.item():.6f}  pos_acc={pos_acc:.3f}  {sps:.1f} st/s", flush=True)
    if (step + 1) % 2000 == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(SAVE_PATH))

# ── Inference ──
print(f"\n{'='*50}", flush=True)

def tokenize_stream(model, data):
    buf = bytearray()
    i = 0
    while i < len(data) or len(buf) >= 24:
        while len(buf) < 24 and i < len(data): buf.append(data[i]); i += 1
        if len(buf) < 24: break
        inp = torch.tensor([list(buf[:24])], device=device)
        boundary = model(inp).squeeze(0).argmax().item()
        token_len = boundary + 1
        if token_len <= len(buf):
            yield bytes(buf[:token_len]); buf = buf[token_len:]
        else: break
    if buf: yield bytes(buf)

test = b"Hello World! This is a test of the streaming tokenizer."
print(f"Input: {test!r}", flush=True)
tokens = list(tokenize_stream(model, test))
for t in tokens: print(f"  {t!r}", flush=True)
real_t = tok.encodeBytes(test)
print(f"\nOur: {len(tokens)} tok  Real: {len(real_t)} tok", flush=True)
print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
