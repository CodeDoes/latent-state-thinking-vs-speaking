"""Train byte encoder to match g1g embedding. Batched for speed.
"""
import torch, torch.nn.functional as F, time, random
from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER
from minGRU_pytorch import minGRU

DEVICE = "cuda"
DIM = 256
LR = 3e-4
STEPS = 5000
BATCH = 256
MAX_BYTES = 24
SAVE_PATH = "experiments/byte_time_mix/encoder.pt"

device = torch.device(DEVICE)
tok = RWKV_TOKENIZER(str(Path("src/rwkv_vocab_v20230424.txt")))

class ByteEncoder(torch.nn.Module):
    def __init__(self, dim=256, max_bytes=24):
        super().__init__()
        self.embed = torch.nn.Embedding(258, dim)
        self.pos = torch.nn.Embedding(max_bytes, dim)
        self.gru = minGRU(dim)
        self.out = torch.nn.Linear(dim, 2560)
    def forward(self, byte_ids):
        B, T = byte_ids.shape
        pos = torch.arange(T, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        return self.out(self.gru(self.embed(byte_ids) + self.pos(pos))[:, -1])

# Build batched dataset
with open("experiments/tinystories_texts.txt") as f:
    stories = f.read().split("\n---END---\n")

byte_batch = []
tid_batch = []
for s in stories[:200]:
    raw = s.encode("utf-8")[:4096]
    for tid in tok.encodeBytes(raw)[:100]:
        if tid >= 65529: continue
        b = tok.idx2token.get(tid, b'')
        if len(b) == 0 or len(b) > MAX_BYTES: continue
        row = [0]*MAX_BYTES
        for j, x in enumerate(b): row[j] = x
        byte_batch.append(row)
        tid_batch.append(tid)

byte_tensor = torch.tensor(byte_batch, dtype=torch.long, device=device)
tid_tensor = torch.tensor(tid_batch, device=device)
print(f"Data: {len(byte_batch)} items", flush=True)

enc = ByteEncoder().to(device)
emb = torch.nn.Embedding(65529, 2560).to(device)
emb.requires_grad_(False)
opt = torch.optim.AdamW(enc.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in enc.parameters()):,}", flush=True)

t0 = time.time()
for step in range(STEPS):
    idx = torch.randperm(len(byte_batch), device=device)[:BATCH]
    pred = enc(byte_tensor[idx])
    target = emb(tid_tensor[idx])
    loss = F.mse_loss(pred, target)
    
    opt.zero_grad(); loss.backward(); opt.step()
    
    if (step+1) % 500 == 0:
        sps = (step+1) / (time.time() - t0)
        cos = F.cosine_similarity(pred, target).mean().item()
        print(f"step {step+1:5d}  loss={loss.item():.4f}  cos={cos:.4f}  {sps:.1f} st/s", flush=True)
    if (step+1) % 2000 == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(enc.state_dict(), Path(SAVE_PATH))

print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)
