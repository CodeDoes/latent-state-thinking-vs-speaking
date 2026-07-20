"""Byte-level auto-encoder: bytes → latent → bytes with trigger.
"""
import torch, torch.nn.functional as F, time, random
from pathlib import Path
import sys; sys.path.insert(0, '.')
from minGRU_pytorch import minGRU
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER

DEVICE = "cuda"; DIM = 128; LATENT = 256; LR = 3e-3
STEPS = 5000; BATCH = 128; MAX_BYTES = 24; LOG_EVERY = 500
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
        B, T = byte_ids.shape
        pos = torch.arange(T, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(byte_ids) + self.pos(pos)
        h = self.enc1(x)
        h = self.enc2(h)
        return self.to_latent(h)

    def forward(self, byte_ids):
        B, T = byte_ids.shape
        latent = self.encode(byte_ids)
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


if __name__ == "__main__":
    # Build dataset from full vocabulary
    items = []
    for tid in range(1, 65529):
        b = tok.idx2token.get(tid, b"")
        if len(b) and len(b) <= MAX_BYTES:
            items.append(b)
    print(f"Items: {len(items):,}", flush=True)

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
        lengths = [len(tb) for tb in batch]
        max_T = max(lengths)
        inp = torch.zeros(BATCH, max_T, dtype=torch.long, device=device)
        for i, tb in enumerate(batch):
            for j, b in enumerate(tb):
                inp[i, j] = 2 + b

        logits, trig = model(inp)
        mask = (inp > 0).float()
        byte_loss = F.cross_entropy(logits.reshape(-1, 258), inp.reshape(-1), reduction='none')
        byte_loss = (byte_loss.view_as(mask) * mask).sum() / (mask.sum() + 1e-8)

        tgt = torch.zeros(BATCH, max_T, device=device)
        for i, L in enumerate(lengths):
            tgt[i, L - 1] = 1.0
        trig_loss = F.binary_cross_entropy(trig.squeeze(-1) * mask, tgt * mask)

        loss = byte_loss + 0.5 * trig_loss
        
        # Contrastive loss for single-byte tokens: push their latents apart
        if lengths[0] == 1 and all(l == 1 for l in lengths):
            # Get latents for this batch of single-byte tokens
            with torch.no_grad():
                lats = model.encode(inp)  # (B, 1, latent)
            # Cosine similarity matrix: encourage diversity
            lats = lats.squeeze(1)  # (B, latent)
            sim = (lats @ lats.T) / (lats.norm(dim=1, keepdim=True) @ lats.norm(dim=1, keepdim=True).T + 1e-8)
            off_diag = sim - torch.eye(BATCH, device=inp.device) * 2  # zero out diagonal, negative for others
            # Push all off-diagonal toward 0
            cont_loss = (off_diag ** 2).mean()
            loss = loss + 0.01 * cont_loss
        
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
    print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
