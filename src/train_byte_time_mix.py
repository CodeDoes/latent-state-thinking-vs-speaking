"""Train byte encoder to match layer 0's time-mix output (tm_out).

Data: experiments/byte_time_mix/training_data/story_*.pt
Each sample: bytes (24 byte IDs) → target = tm_out (2560-dim) from real g1g layer 0.

Architecture: raw uint8[24] → scalar-proj → minGRU → project → tm_out.
No 256-way embedding table: each byte is its raw uint8 value (0..255),
normalized, projected to dim. Padding is just value 0 like any byte
(do not fixate on 0/1).

Output is a tuple (state, mask):
  state: (B, 2560) time-mix state (tm_out)
  mask:  (B, 24) per-position logit. At inference, split before the first
         position where sigmoid(mask) > threshold (see find_split) — i.e.
         the token boundary is found by thresholding, not by fixed bins.
         Trained toward a step function at num_bytes (1 before, 0 after)."""
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
        # Raw uint8[24]: each byte is a scalar in 0..255 -> dim.
        self.byte_proj = nn.Linear(1, dim)
        self.pos = nn.Embedding(max_bytes, dim)
        self.gru = minGRU(dim)
        self.out = nn.Linear(dim, 2560)          # state head (from last pos)
        self.mask_head = nn.Linear(dim, 1)       # per-position mask logit (B, T)
    def forward(self, byte_ids):
        # byte_ids: (B, T) with values (2 + b); recover true uint8, normalize.
        B, T = byte_ids.shape
        raw = (byte_ids.float() - 2).clamp(0, 255) / 255.0  # (B, T) in [0,1]
        x = self.byte_proj(raw.unsqueeze(-1))               # (B, T, dim)
        pos = torch.arange(T, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos(pos)
        h = self.gru(x)                                     # (B, T, dim)
        state = self.out(h[:, -1])                          # (B, 2560)
        mask = self.mask_head(h).squeeze(-1)                # (B, T) per-position logit
        return state, mask


def find_split(mask_logits, threshold=0.5):
    """Split position = first index where sigmoid(mask) > threshold.

    Returns (B,) long tensor of split points (clamped to T if none exceed).
    The token's bytes are byte_ids[:, :split]; the boundary byte is the one
    at `split` (i.e. split *before* the first position exceeding threshold).
    """
    B, T = mask_logits.shape
    active = torch.sigmoid(mask_logits) > threshold          # (B, T) bool
    # first True per row; if none, default to T (whole window)
    idx = torch.full((B,), T, dtype=torch.long, device=mask_logits.device)
    first = active.float().argmax(dim=1)                    # first 1 (or 0 if none)
    has = active.any(dim=1)
    idx[has] = first[has]
    return idx

# ── Load data ──
files = sorted(DATA_DIR.glob("story_*.pt"))
print(f"Found {len(files)} data files", flush=True)

all_bytes, all_targets, all_xx, all_mats, all_nbytes = [], [], [], [], []
for f in files:
    try:
        chunk = torch.load(f, map_location='cpu', weights_only=True)
        for s in chunk:
            all_bytes.append(s['bytes'])
            all_targets.append(s['tm_out'].float())
            all_xx.append(s['xx'].float())
            all_mats.append(s['mat'].float())
            all_nbytes.append(s['num_bytes'])
    except Exception as e:
        print(f"  Skipping {f.name}: {e}", flush=True)

if len(all_bytes) == 0:
    print("No training data!", flush=True)
    sys.exit(1)

bytes_t = torch.tensor(all_bytes, dtype=torch.long)
targets_t = torch.stack(all_targets)
xx_t = torch.stack(all_xx)
mats_t = torch.stack(all_mats)
nbytes_t = torch.tensor(all_nbytes, dtype=torch.long)
MAX_BYTES = bytes_t.shape[1]  # 24
print(f"Data: {len(all_bytes)} samples", flush=True)
print(f"  bytes: {bytes_t.shape}, tm_out: {targets_t.shape}", flush=True)
print(f"  tm_out mean={targets_t.mean():.3f} std={targets_t.std():.3f}", flush=True)

# ── Model ──
enc = ByteTimeMixEncoder().to(device)
opt = torch.optim.AdamW(enc.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in enc.parameters()):,}", flush=True)

# ── Train ──
MASK_THRESHOLD = 0.5
MASK_LAMBDA = 0.1
t0 = time.time()
for step in range(STEPS):
    idx = torch.randperm(len(all_bytes), device='cpu')[:BATCH]
    inp = bytes_t[idx].to(device)
    target = targets_t[idx].to(device)

    state, mask = enc(inp)                       # mask: (B, 24) per-position logit
    state_loss = F.mse_loss(state, target)

    # mask target: rising edge at num_bytes — ~0 before the boundary byte,
    # ~1 at/after it, so sigmoid(mask) exceeds threshold exactly at the split.
    nb = nbytes_t[idx].to(device)                                  # (B,)
    pos_idx = torch.arange(MAX_BYTES, device=device).unsqueeze(0).float()  # (1, T)
    mask_tgt = torch.sigmoid(4.0 * (pos_idx - nb.unsqueeze(1)))    # (B, 24) ramp
    mask_loss = F.binary_cross_entropy_with_logits(mask, mask_tgt)

    loss = state_loss + MASK_LAMBDA * mask_loss

    opt.zero_grad(); loss.backward(); opt.step()

    if (step+1) % 1000 == 0:
        sps = (step+1) / (time.time() - t0)
        cos = F.cosine_similarity(state, target).mean().item()
        # split accuracy vs num_bytes
        split = find_split(mask, MASK_THRESHOLD)
        acc = (split == nb).float().mean().item()
        print(f"step {step+1:5d}  loss={loss.item():.4f}  state={state_loss.item():.4f}  "
              f"mask={mask_loss.item():.4f}  cos={cos:.4f}  split_acc={acc:.3f}  {sps:.1f} st/s", flush=True)
    if (step+1) % 5000 == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(enc.state_dict(), Path(SAVE_PATH))

Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
torch.save(enc.state_dict(), Path(SAVE_PATH))
print(f"\nSaved to {SAVE_PATH}", flush=True)
print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
