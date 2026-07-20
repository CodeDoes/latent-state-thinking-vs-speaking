"""Train byte encoder to match layer 0's time-mix output (tm_out).

Data: experiments/byte_time_mix/training_data/story_*.pt
Each sample: bytes (24 byte IDs) → target = tm_out (2560-dim) from real g1g layer 0.

Architecture: bytes → 8-bit embed → minGRU → project → tm_out.
"""
import torch, torch.nn.functional as F, time
from pathlib import Path
import sys; sys.path.insert(0, '.')
from minGRU_pytorch import minGRU
import torch.nn as nn

torch.set_float32_matmul_precision('high')

DEVICE = "cuda"
DIM = 256
LR = 3e-4
STEPS = 20000
BATCH = 256
MAX_BYTES = 24
SAVE_PATH = "experiments/byte_time_mix/encoder.pt"
DATA_DIR = Path("experiments/byte_time_mix/training_data")

device = torch.device(DEVICE)

# ── Encoder ──
class ByteTimeMixEncoder(nn.Module):
    def __init__(self, dim=256, max_bytes=24):
        super().__init__()
        self.embed = nn.Embedding(258, dim)
        self.pos = nn.Embedding(max_bytes, dim)
        self.gru = minGRU(dim)
        self.out = nn.Linear(dim, 2560)
    def forward(self, byte_ids):
        B, T = byte_ids.shape
        pos = torch.arange(T, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        return self.out(self.gru(self.embed(byte_ids) + self.pos(pos))[:, -1])

# ── Load data ──
files = sorted(DATA_DIR.glob("story_*.pt"))
print(f"Found {len(files)} data files", flush=True)

all_bytes, all_targets, all_xx, all_mats = [], [], [], []
for f in files:
    try:
        chunk = torch.load(f, map_location='cpu', weights_only=True)
        for s in chunk:
            all_bytes.append(s['bytes'])
            all_targets.append(s['tm_out'].float())
            all_xx.append(s['xx'].float())
            all_mats.append(s['mat'].float())
    except Exception as e:
        print(f"  Skipping {f.name}: {e}", flush=True)

if len(all_bytes) == 0:
    print("No training data!", flush=True)
    sys.exit(1)

bytes_t = torch.tensor(all_bytes, dtype=torch.long)
targets_t = torch.stack(all_targets)
xx_t = torch.stack(all_xx)
mats_t = torch.stack(all_mats)
print(f"Data: {len(all_bytes)} samples", flush=True)
print(f"  bytes: {bytes_t.shape}, tm_out: {targets_t.shape}", flush=True)
print(f"  tm_out mean={targets_t.mean():.3f} std={targets_t.std():.3f}", flush=True)

# ── Model ──
enc = ByteTimeMixEncoder().to(device)
opt = torch.optim.AdamW(enc.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in enc.parameters()):,}", flush=True)

# ── Train ──
t0 = time.time()
for step in range(STEPS):
    idx = torch.randperm(len(all_bytes), device='cpu')[:BATCH]
    inp = bytes_t[idx].to(device)
    target = targets_t[idx].to(device)
    
    pred = enc(inp)
    loss = F.mse_loss(pred, target)
    
    opt.zero_grad(); loss.backward(); opt.step()
    
    if (step+1) % 1000 == 0:
        sps = (step+1) / (time.time() - t0)
        cos = F.cosine_similarity(pred, target).mean().item()
        print(f"step {step+1:5d}  loss={loss.item():.4f}  cos={cos:.4f}  {sps:.1f} st/s", flush=True)
    if (step+1) % 5000 == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(enc.state_dict(), Path(SAVE_PATH))

Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
torch.save(enc.state_dict(), Path(SAVE_PATH))
print(f"\nSaved to {SAVE_PATH}", flush=True)
print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
