"""Train loopy xx predictor: byte → byte_embed → minGRU → predict xx.

xx is the time-mix state input (previous time-mix output, 2560-dim).
At each byte, our model predicts what xx WILL BE after processing this byte.
The trigger fires at token boundaries — the accumulated xx feeds into layer 1.

Replaces byte_embed + layer 0's time-mix entirely.
"""
import torch, torch.nn.functional as F, time
from pathlib import Path
import sys; sys.path.insert(0, '.')
from minGRU_pytorch import minGRU
import torch.nn as nn

DEVICE = "cuda"
DIM = 256
LR = 3e-4
STEPS = 3000
BATCH = 128
SAVE_PATH = "experiments/loopy_timemix/model.pt"

device = torch.device(DEVICE)

# ── Load g1g components ──
print("Loading g1g...", flush=True)
ckpt = torch.load(Path.home() / "Documents/models/rwkv7-g1g-byte-iface/model.pth", map_location='cpu', weights_only=True)

byte_embed = nn.Embedding(258, 2560)
byte_embed.weight.data = ckpt['byte_embed.weight'].float()
byte_embed = byte_embed.to(device).eval()
for p in byte_embed.parameters(): p.requires_grad = False

# Time-mix weights (for generating training targets: xx = time-mix output)
ln1_w = ckpt['blocks.0.ln1.weight'].float().to(device)
ln1_b = ckpt['blocks.0.ln1.bias'].float().to(device)
key_w = ckpt['blocks.0.att.key.weight'].float().to(device)
value_w = ckpt['blocks.0.att.value.weight'].float().to(device)
recep_w = ckpt['blocks.0.att.receptance.weight'].float().to(device)
output_w = ckpt['blocks.0.att.output.weight'].float().to(device)

def time_mix_xx(x):
    """Compute xx (time-mix output) for a byte embedding x."""
    xn = F.layer_norm(x, [2560], ln1_w, ln1_b)
    return F.linear(F.linear(xn, recep_w) * F.linear(xn, key_w) * F.linear(xn, value_w), output_w)

# ── Loopy model: byte → predict xx ──
class LoopyXX(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.proj = nn.Linear(2560, dim)
        self.gru = minGRU(dim)
        self.xx_head = nn.Linear(dim, 2560)  # predicts xx
        self.trig = nn.Linear(dim, 1)        # predicts trigger

    def forward(self, emb):
        """emb: (B, 2560) — single byte embedding. Returns predicted_xx, trigger."""
        h = self.gru(self.proj(emb).unsqueeze(1))
        h = h.squeeze(1)
        return self.xx_head(h), torch.sigmoid(self.trig(h))

model = LoopyXX().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# ── Train on all 256 byte values ──
all_byte_ids = torch.tensor([[2 + b for b in range(256)]], device=device)
all_embs = byte_embed(all_byte_ids)  # (1, 256, 2560)
all_xx = torch.stack([time_mix_xx(all_embs[0, b].unsqueeze(0)) for b in range(256)]).squeeze(1)

t0 = time.time()
for step in range(STEPS):
    idx = torch.randperm(256, device=device)[:BATCH]
    pred, _ = model(all_embs[0, idx])
    # Load balancing loss: encourage uniform routing
    load_balance = -router_weights * torch.log(router_weights + 1e-8)  # entropy
    load_balance = load_balance.sum(-1).mean()
    
    loss = F.mse_loss(pred, all_xx[idx]) - 0.01 * load_balance  # maximize entropy initially

    opt.zero_grad(); loss.backward(); opt.step()

    if (step + 1) % 500 == 0:
        cos = F.cosine_similarity(pred, all_xx[idx]).mean().item()
        sps = (step + 1) / (time.time() - t0)
        print(f"step {step+1:5d}  loss={loss.item():.6f}  cos={cos:.4f}  {sps:.1f} st/s", flush=True)

    if (step + 1) % 2000 == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(SAVE_PATH))

# Save
Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), Path(SAVE_PATH))
print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)
