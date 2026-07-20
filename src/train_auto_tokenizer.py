"""Train per-token byte auto-tokenizer with smooth trigger ramp.

Encoder reads a fixed-width window (max token length = 80 bytes).
Trigger target: linear ramp 0→1 from first to last byte of each token.
Decoder reconstructs bytes from the latent at the trigger position.

After training: encoder produces latents at token boundaries,
decoder reconstructs tokens from those latents.
"""
import torch, torch.nn.functional as F, time, random
from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER
from minGRU_pytorch import minGRU
import torch.nn as nn

# ════════════ CONFIG ════════════
DEVICE = "cuda"
DIM = 128
LATENT_DIM = 256
LR = 3e-3
BATCH = 128
MAX_TOKEN_LEN = 80
LOG_EVERY = 50
SAVE_EVERY = 500
EPOCHS = 30
SAVE_PATH = "experiments/auto_tokenizer/per_token_model.pt"
# ════════════════════════════════

BYTE_VOCAB = 258
BYTE_PAD = 0

tok = RWKV_TOKENIZER(str(Path("src/rwkv_vocab_v20230424.txt")))

with open("experiments/tinystories_texts.txt") as f:
    stories = f.read().split("\n---END---\n")
stories = [s.strip() for s in stories if s.strip()]
random.shuffle(stories)
print(f"Stories: {len(stories):,}", flush=True)

device = torch.device(DEVICE)

# ── Extract all tokens from stories ──
# Group by length, pad to MAX_TOKEN_LEN
tokens_by_len = {}
for si, s in enumerate(stories[:500]):
    if si % 100 == 0: print(f"  processing story {si}/{500}", flush=True)
    raw = s.encode("utf-8")[:4096]
    idx = 0
    while idx < len(raw):
        old = idx
        idx, node, values = tok.root.find_longest(raw, idx)
        if idx == old: idx += 1; continue
        _, tid = next(iter(values))
        L = idx - old
        if L > MAX_TOKEN_LEN: continue
        tokens_by_len.setdefault(L, []).append(bytes(raw[old:idx]))

print(f"Token lengths: {sorted(tokens_by_len.keys())}", flush=True)
total = sum(len(v) for v in tokens_by_len.values())
print(f"Total tokens: {total:,}", flush=True)

# Build padded tensors per length
items_by_len = {}
for L, token_list in tokens_by_len.items():
    N = len(token_list)
    byte_mat = torch.zeros(N, L, dtype=torch.long, device=device)
    trig_target = torch.zeros(N, L, device=device)
    for i, token_bytes in enumerate(token_list):
        for j, b in enumerate(token_bytes):
            byte_mat[i, j] = 2 + b
        # Smooth trigger ramp: 0→1 from first to last byte
        for j in range(L):
            trig_target[i, j] = (j + 1) / L  # linear from ~0 to 1
    items_by_len[L] = (byte_mat, trig_target)

print(f"Groups: {len(items_by_len)}", flush=True)

# ── Architecture ──

class Encoder(nn.Module):
    """Fixed-width byte window → latent at each position."""
    def __init__(self, dim, latent_dim, max_len):
        super().__init__()
        self.embed = nn.Embedding(BYTE_VOCAB, dim, padding_idx=BYTE_PAD)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.gru = minGRU(dim)
        self.latent_head = nn.Linear(dim, latent_dim)
        self.trigger_head = nn.Linear(dim, 1)
    
    def forward(self, byte_ids):
        B, T = byte_ids.shape
        pos = torch.arange(T, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(byte_ids) + self.pos_embed(pos)
        h = self.gru(x)  # (B, T, dim)
        latent = self.latent_head(h)  # (B, T, latent_dim)
        trigger_logits = self.trigger_head(h).squeeze(-1)  # (B, T)
        return latent, trigger_logits


class Decoder(nn.Module):
    """Latent → bytes, step by step."""
    def __init__(self, dim, latent_dim):
        super().__init__()
        self.project = nn.Linear(latent_dim, dim)
        self.embed = nn.Embedding(BYTE_VOCAB, dim, padding_idx=BYTE_PAD)
        self.gru = minGRU(dim)
        self.byte_head = nn.Linear(dim, BYTE_VOCAB)
    
    def forward(self, latent, target_bytes):
        """Teacher-forced: latent (B, 1, LD), target_bytes (B, T) → logits."""
        B, T = target_bytes.shape
        # Prepare decoder input embeddings (teacher forcing)
        prev = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=target_bytes.device),
                          target_bytes[:, :-1]], dim=1)
        x = self.embed(prev) + self.project(latent)  # (B, T, dim)
        h = self.gru(x)
        logits = self.byte_head(h)  # (B, T, 258)
        return logits


enc = Encoder(DIM, LATENT_DIM, MAX_TOKEN_LEN).to(device)
dec = Decoder(DIM, LATENT_DIM).to(device)
params = list(enc.parameters()) + list(dec.parameters())
opt = torch.optim.AdamW(params, lr=LR)
n = sum(p.numel() for p in params)
print(f"Params: {n:,}", flush=True)

t0 = time.time()
step = 0

for epoch in range(EPOCHS):
    lengths = list(items_by_len.keys())
    random.shuffle(lengths)
    
    for L in lengths:
        byte_mat, trig_target = items_by_len[L]
        N = byte_mat.shape[0]
        
        for bstart in range(0, N, BATCH):
            batch_bytes = byte_mat[bstart:bstart + BATCH]
            batch_trig = trig_target[bstart:bstart + BATCH]
            bb = batch_bytes.shape[0]
            
            # Encode
            latent, trigger_logits = enc(batch_bytes)  # (B, T, LD), (B, T)
            
            # Trigger loss: smooth ramp
            trig_loss = F.binary_cross_entropy_with_logits(trigger_logits, batch_trig)
            
            # Decode: teacher-forced byte prediction
            # Use latent at trigger position (last byte = best context)
            latent_last = latent[:, -1:]  # (B, 1, LD)
            byte_logits = dec(latent_last, batch_bytes)  # (B, T, 258)
            
            byte_loss = F.cross_entropy(byte_logits.view(-1, BYTE_VOCAB), batch_bytes.view(-1))
            
            loss = byte_loss + trig_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            
            if step % LOG_EVERY == 0:
                sps = step / (time.time() - t0)
                with torch.no_grad():
                    acc = (byte_logits.argmax(-1) == batch_bytes).float().mean().item()
                avg_trig = torch.sigmoid(trigger_logits[:, -1]).mean().item()
                print(f"step {step:5d}  e{epoch}  L={L:2d}  byte={byte_loss.item():.3f}  "
                      f"trig={trig_loss.item():.3f}  acc={acc:.3f}  last_trig={avg_trig:.2f}  {sps:.1f} st/s",
                      flush=True)
            
            if step % SAVE_EVERY == 0:
                Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
                torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()}, Path(SAVE_PATH))
                print(f"  saved", flush=True)

torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()}, Path(SAVE_PATH))
print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)

# Eval: test reconstruction for a few tokens
print("\n--- Eval ---", flush=True)
for L in sorted(items_by_len.keys())[:3]:
    byte_batch, _ = items_by_len[L]
    if byte_batch.shape[0] == 0: continue
    batch = byte_batch[:5]
    latent, trigger_logits = enc(batch)
    latent_last = latent[:, -1:]
    byte_logits = dec(latent_last, batch)
    pred = byte_logits.argmax(-1)
    trig_last = torch.sigmoid(trigger_logits[:, -1]).mean().item()
    for i in range(min(3, batch.shape[0])):
        orig = bytes(b-2 for b in batch[i].tolist() if b >= 2)
        pred_str = bytes(b-2 for b in pred[i].tolist() if b >= 2)
        ok = "✓" if orig == pred_str else "✗"
        print(f"  {ok} {orig!r} → {pred_str!r}  trig_last={trig_last:.2f}", flush=True)
