"""Curriculum auto-tokenizer: train one length at a time, freeze when done.

L=1 first → freeze pos_0 → L=2 → freeze pos_1 → ... 
Each length learns only its new position. OOD examples added:
random byte sequences get trigger_target=0 everywhere (don't fire on garbage).
"""
import torch, torch.nn.functional as F, time, random
from pathlib import Path
import sys; sys.path.insert(0, '.')
from domains.rwkv.hf_rwkv_tokenizer import RWKV_TOKENIZER
from minGRU_pytorch import minGRU
import torch.nn as nn

DEVICE = "cuda"
DIM = 128
LATENT_DIM = 256
LR = 3e-3
BATCH = 64
MAX_TOKEN_LEN = 80
TRIGGER_WEIGHT = 1.0
ACC_THRESH = 0.95
SAVE_PATH = "experiments/auto_tokenizer/curriculum_model.pt"
CACHE_PATH = "experiments/auto_tokenizer/dataset.pt"

BYTE_VOCAB = 258
tok = RWKV_TOKENIZER(str(Path("domains/rwkv/rwkv_vocab_v20230424.txt")))
device = torch.device(DEVICE)

if not Path(CACHE_PATH).exists():
    print("Building dataset cache...", flush=True)
    from domains.rwkv.hf_rwkv_tokenizer import RWKV_TOKENIZER
    with open("experiments/tinystories_texts.txt") as f:
        raw = f.read()
    stories = raw.split("\n---END---\n")
    stories = [s.strip() for s in stories if s.strip()]
    random.shuffle(stories)
    
    tokens_by_len = {}
    for si, s in enumerate(stories[:500]):
        if si % 100 == 0: print(f"  story {si}/500", flush=True)
        raw_bytes = s.encode("utf-8")[:4096]
        idx = 0
        while idx < len(raw_bytes):
            old = idx
            idx, node, values = tok.root.find_longest(raw_bytes, idx)
            if idx == old: idx += 1; continue
            L = idx - old
            if L > MAX_TOKEN_LEN: continue
            tokens_by_len.setdefault(L, []).append(bytes(raw_bytes[old:idx]))
    
    items_by_len = {}
    for L, token_list in tokens_by_len.items():
        N = len(token_list)
        byte_mat = torch.zeros(N, MAX_TOKEN_LEN, dtype=torch.long)
        trig_target = torch.zeros(N, MAX_TOKEN_LEN)
        for i, tb in enumerate(token_list):
            for j, b in enumerate(tb):
                byte_mat[i, j] = 2 + b
            # Each column fires for ALL tokens that reach it
            # Position 0 fires for any length >= 1 (all tokens)
            # Position j fires for length >= j+1
            trig_target[i, :L] = 1.0
        items_by_len[L] = (byte_mat, trig_target)
    
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(items_by_len, CACHE_PATH)
    print(f"Cached {sum(v[0].shape[0] for v in items_by_len.values()):,} tokens", flush=True)

items_by_len = torch.load(CACHE_PATH, map_location='cpu')
items_by_len = {k: (v[0].to(device), v[1].to(device)) for k, v in items_by_len.items()}
lengths = sorted(items_by_len.keys())
print(f"Lengths: {lengths}", flush=True)

class Encoder(nn.Module):
    def __init__(self, dim, latent_dim, max_len):
        super().__init__()
        self.embed = nn.Embedding(BYTE_VOCAB, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.gru = minGRU(dim)
        self.latent_head = nn.Linear(dim, latent_dim)
        self.trigger_head = nn.Linear(dim, 1)
    def forward(self, byte_ids):
        B, T = byte_ids.shape
        pos = torch.arange(T, device=byte_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(byte_ids) + self.pos_embed(pos)
        h = self.gru(x)
        return self.latent_head(h), self.trigger_head(h).squeeze(-1)

class Decoder(nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()
        self.project = nn.Linear(latent_dim, dim)
        self.embed = nn.Embedding(BYTE_VOCAB, dim)
        self.gru = minGRU(dim)
        self.byte_head = nn.Linear(dim, BYTE_VOCAB)
    def forward(self, latent, target_bytes):
        B, T = target_bytes.shape
        prev = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=target_bytes.device),
                          target_bytes[:, :-1]], dim=1)
        x = self.embed(prev) + self.project(latent)
        return self.byte_head(self.gru(x))

enc = Encoder(DIM, LATENT_DIM, MAX_TOKEN_LEN).to(device)
dec = Decoder(DIM, LATENT_DIM).to(device)
params = list(enc.parameters()) + list(dec.parameters())
opt = torch.optim.AdamW(params, lr=LR)
n = sum(p.numel() for p in params)
print(f"Params: {n:,}", flush=True)

# Frozen positions tracker (in pos_embed)
pos_frozen = torch.zeros(MAX_TOKEN_LEN, dtype=torch.bool, device=device)
acc_by_len = {}
t0 = time.time()
step = 0

def make_ood_batch(L, n):
    """Generate random byte sequences as OOD examples. All get trigger=0."""
    noise = torch.randint(2, 258, (n, MAX_TOKEN_LEN), device=device)
    mask = torch.zeros(n, MAX_TOKEN_LEN, device=device)
    mask[:, :L] = 1.0  # only first L bytes are "active"
    trig_target = torch.zeros(n, MAX_TOKEN_LEN, device=device)  # never trigger
    return noise, noise.clone(), trig_target, mask

for L in lengths:
    byte_mat, trig_target = items_by_len[L]
    N = byte_mat.shape[0]
    
    print(f"\n{'='*50}\nL={L} ({N:,} tok)  frozen_pos: {pos_frozen.sum().item()}", flush=True)
    
    for epoch in range(3):
        idxs = list(range(N))
        random.shuffle(idxs)
        total_acc, n_batches = 0.0, 0
        
        for bstart in range(0, N, BATCH):
            batch_idx = idxs[bstart:bstart + BATCH]
            batch_bytes = byte_mat[batch_idx]
            batch_trig = trig_target[batch_idx]
            bb = batch_bytes.shape[0]
            mask = (batch_bytes > 0).float()
            
            latent, trigger_logits = enc(batch_bytes)
            
            # Use per-position latents (first L positions)
            byte_logits = dec(latent[:, :L], batch_bytes[:, :L])  # (B, L, 258)
            
            # Byte loss only over actual token positions
            b_loss = F.cross_entropy(byte_logits.view(-1, BYTE_VOCAB), batch_bytes[:, :L].contiguous().view(-1))
            
            # Trigger loss over actual token positions
            t_loss = F.binary_cross_entropy_with_logits(
                trigger_logits[:, :L].contiguous().view(-1),
                batch_trig[:, :L].contiguous().view(-1)
            )
            
            loss = b_loss + TRIGGER_WEIGHT * t_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            # Freeze positions learned in PREVIOUS lengths (not current one)
            if pos_frozen.any():
                enc.pos_embed.weight.grad[pos_frozen] = 0
            
            opt.step(); opt.zero_grad()
            step += 1
            
            with torch.no_grad():
                acc = (byte_logits.argmax(-1) == batch_bytes[:, :L]).float().mean().item()
            total_acc += acc; n_batches += 1
            
            if step % 100 == 0:
                sps = step / (time.time() - t0)
                lr = opt.param_groups[0]['lr']
                print(f"  step {step:5d}  L={L}  e{epoch}  byte={b_loss.item():.4f}  trig={t_loss.item():.4f}  acc={acc:.4f}  {sps:.1f} st/s  lr={lr:.2e}", flush=True)
        
        avg_acc = total_acc / n_batches
        print(f"  → L={L} e{epoch} avg_acc={avg_acc:.4f}", flush=True)
        
        if avg_acc > ACC_THRESH:
            break  # Stop this length, move to next
    
    # After done training this length, freeze its positions
    pos_frozen[:L] = True
    print(f"  ★ Frozen positions 0..{L-1}", flush=True)
    
    acc_by_len[L] = avg_acc
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict(),
                'pos_frozen': pos_frozen.cpu(), 'acc_by_len': acc_by_len}, Path(SAVE_PATH))

print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)

# OOD test
print("\n--- OOD Test ---", flush=True)
for L in [1, 3, 5]:
    noise, _, _, _ = make_ood_batch(L, 20)
    _, trigger_logits = enc(noise)
    trig_probs = torch.sigmoid(trigger_logits)
    max_trig = trig_probs.max().item()
    mean_trig = trig_probs.mean().item()
    print(f"  L={L} OOD: max_trigger={max_trig:.4f}  mean_trigger={mean_trig:.4f}  (should be ~0)", flush=True)

# Reconstruction test
print("\n--- Reconstruction ---", flush=True)
for L in lengths[:5]:
    batch = items_by_len[L][0][:5]  # (B, 80)
    latent, _ = enc(batch)  # (B, 80, LD)
    # Use per-position latents for only the actual token length
    batch_slice = batch[:, :L]
    latent_slice = latent[:, :L]
    logits = dec(latent_slice, batch_slice)  # (B, L, 258)
    pred = logits.argmax(-1)  # (B, L)
    for i in range(min(3, batch.shape[0])):
        orig = bytes(b-2 for b in batch[i, :L].tolist() if b >= 2)
        got = bytes(b-2 for b in pred[i].tolist() if b >= 2)
        print(f"  L={L} {'✓' if orig==got else '✗'} {orig!r} → {got!r}", flush=True)
