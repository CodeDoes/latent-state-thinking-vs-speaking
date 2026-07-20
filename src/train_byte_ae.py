"""Byte-level auto-encoder: bytes → latent → bytes with trigger.

Encoder: stacked 2-layer minGRU reads bytes, produces latent per position.
Decoder: minGRU reconstructs bytes from latents step by step (teacher forcing).
Trigger: fires at the last byte of each token (token boundary).

The latent at trigger position is the compact representation usable by g1g.
"""
import torch, torch.nn.functional as F, time, random
from pathlib import Path
import sys; sys.path.insert(0, '.')
from minGRU_pytorch import minGRU
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER

DEVICE = "cuda"; DIM = 128; LATENT = 256; LR = 3e-3
STEPS = 1500; BATCH = 32; MAX_BYTES = 24; LOG_EVERY = 300
SAVE_PATH = "experiments/byte_ae/model.pt"

device = torch.device(DEVICE)
tok = RWKV_TOKENIZER(str(Path("src/rwkv_vocab_v20230424.txt")))


class ByteAE(torch.nn.Module):
    def __init__(self, dim=DIM, latent=LATENT):
        super().__init__()
        self.embed = torch.nn.Embedding(258, dim)
        self.pos = torch.nn.Embedding(MAX_BYTES, dim)
        self.enc1 = minGRU(dim)
        self.enc2 = minGRU(dim)
        self.to_latent = torch.nn.Linear(dim, latent)
        self.dec_proj = torch.nn.Linear(latent, dim)
        self.dec_embed = torch.nn.Embedding(258, dim)
        self.dec_gru = minGRU(dim)
        self.byte_head = torch.nn.Linear(dim, 258)
        self.trig_head = torch.nn.Linear(dim, 1)

    def encode(self, byte_ids):
        """byte_ids: (B, T). Returns latent (B, T, latent)."""
        B, T = byte_ids.shape
        pos = torch.arange(T, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(byte_ids) + self.pos(pos)
        h = self.enc1(x)
        h = self.enc2(h)
        return self.to_latent(h)

    def forward(self, byte_ids):
        """Full auto-encode: encode → decode with teacher forcing.
        Returns (byte_logits, triggers) each (B, T, *) ."""
        B, T = byte_ids.shape
        latent = self.encode(byte_ids)  # (B, T, latent)
        logits_l, trig_l = [], []
        prev = torch.zeros(B, 1, dtype=torch.long, device=byte_ids.device)
        h = None
        for t in range(T):
            l = latent[:, t:t + 1]
            x = self.dec_proj(l) + self.dec_embed(prev)
            h = self.dec_gru(x, h)
            logits_l.append(self.byte_head(h.squeeze(1)))
            trig_l.append(torch.sigmoid(self.trig_head(h.squeeze(1))))
            prev = byte_ids[:, t:t + 1]
        return torch.stack(logits_l, dim=1), torch.stack(trig_l, dim=1)


# ── Build dataset ──
items = []
with open("experiments/tinystories_texts.txt") as f:
    stories = f.read().split("\n---END---\n")
for s in stories[:50]:
    raw = s.encode("utf-8")[:4096]
    for tid in tok.encodeBytes(raw)[:30]:
        b = tok.idx2token.get(tid, b"")
        if len(b) and len(b) <= MAX_BYTES:
            items.append(b)
print(f"Items: {len(items)}", flush=True)

model = ByteAE().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# Warmup
print("Warming up...", flush=True)
with torch.no_grad():
    _ = model(torch.zeros(1, 3, dtype=torch.long, device=device))
print("Ready.", flush=True)

t0 = time.time()
for step in range(STEPS):
    batch = random.sample(items, min(BATCH, len(items)))
    # Build padded batch (B, max_T)
    lengths = [len(tb) for tb in batch]
    max_T = max(lengths)
    inp = torch.zeros(BATCH, max_T, dtype=torch.long, device=device)
    for i, tb in enumerate(batch):
        for j, b in enumerate(tb):
            inp[i, j] = 2 + b
    
    logits, trig = model(inp)  # (B, T, 258), (B, T)
    
    # Byte loss: masked to non-pad positions
    mask = (inp > 0).float()
    byte_loss = F.cross_entropy(logits.reshape(-1, 258), inp.reshape(-1), reduction='none')
    byte_loss = (byte_loss.view_as(mask) * mask).sum() / (mask.sum() + 1e-8)
    
    # Trigger loss: 1 at last real byte of each token
    tgt = torch.zeros(BATCH, max_T, device=device)
    for i, L in enumerate(lengths):
        tgt[i, L - 1] = 1.0
    trig_loss = F.binary_cross_entropy(trig.squeeze(-1) * mask, tgt * mask)
    
    loss = byte_loss + 0.5 * trig_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

    if (step + 1) % LOG_EVERY == 0:
        sps = (step + 1) / (time.time() - t0)
        print(f"step {step+1:5d}  loss={loss.item():.4f}  {sps:.1f} st/s", flush=True)
    if (step + 1) % 2000 == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(SAVE_PATH))

Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), Path(SAVE_PATH))

# ── Eval ──
print(f"\n{'='*50}", flush=True)
correct = 0
for tb in items[:50]:
    inp = torch.tensor([[2 + b for b in tb]], device=device)
    logits, trig = model(inp)
    T = len(tb)
    pred = logits[0, :T].argmax(-1).tolist()
    pred_b = bytes(b - 2 for b in pred if b >= 2)
    trig_fire = [t for t in range(T) if trig[0, t].item() > 0.5]
    ok = "✓" if tb == pred_b else "✗"
    if ok == "✓":
        correct += 1
    print(f"  {ok} {tb!r:15s} → {pred_b!r:15s}  trig_at={trig_fire}", flush=True)

print(f"\n{correct}/50 correct ({100*correct/50:.0f}%)", flush=True)
print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
