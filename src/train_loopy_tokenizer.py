"""Train loopy tokenizer on full vocab using ranking loss + real text.

Each byte position predicts the correct token ID via dot-product with
learned token embeddings. Loss: push correct token logit above a random
subset of negatives (sampled softmax / NCE).
"""
import torch, torch.nn.functional as F, time, random
from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.loopy_tokenizer import LoopyTokenizerWithPos
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER

# ════════════ CONFIG ════════════
DEVICE = "cuda"
DIM = 8
LR = 1e-1
BATCH = 256
EPOCHS = 100
LOG_EVERY = 10
SAVE_EVERY = 200
NEG_SAMPLES = 512   # negatives per positive for ranking loss
SAVE_PATH = "experiments/loopy_tokenizer/full_model.pt"
# ════════════════════════════════

tok = RWKV_TOKENIZER(str(Path("src/rwkv_vocab_v20230424.txt")))
V = len(tok.token2idx)
device = torch.device(DEVICE)

# Build dataset from vocab file itself — each token as an example
token_ids = sorted(t for t in range(V) if t in tok.idx2token)
by_len = {}
for tid in token_ids:
    b = tok.idx2token[tid]
    L = len(b)
    if L > 10: continue
    by_len.setdefault(L, []).append(tid)

# Pre-encode
items_by_len = {}
for L, tids in by_len.items():
    byte_mat = torch.zeros(len(tids), L, dtype=torch.long, device=device)
    for i, tid in enumerate(tids):
        b = tok.idx2token[tid]
        byte_mat[i] = torch.tensor([2 + byte for byte in b], device=device)
    targets = torch.tensor(tids, device=device)
    items_by_len[L] = (byte_mat, targets)

total_tokens = sum(v[0].shape[0] for v in items_by_len.values())
print(f"Tokens: {total_tokens}, lengths: {list(items_by_len.keys())}", flush=True)

loopy = LoopyTokenizerWithPos(DIM, V).to(device)
opt = torch.optim.SGD(loopy.parameters(), lr=LR, momentum=0.9)
n = sum(p.numel() for p in loopy.parameters())
print(f"Params: {n:,}", flush=True)

t0 = time.time()
step = 0

for epoch in range(EPOCHS):
    lengths = list(items_by_len.keys())
    random.shuffle(lengths)
    
    for L in lengths:
        byte_mat, targets = items_by_len[L]
        n_items = byte_mat.shape[0]
        idxs = list(range(n_items))
        random.shuffle(idxs)
        
        for bstart in range(0, n_items, BATCH):
            batch_idx = idxs[bstart:bstart + BATCH]
            batch_bytes = byte_mat[batch_idx]  # (B', L)
            batch_targets = targets[batch_idx]  # (B',)
            bb = len(batch_idx)
            
            token_logits, trigger_logits = loopy.forward_bytes(batch_bytes)
            last_logits = token_logits[:, -1]  # (B', V)
            
            # Ranking loss: for each target, sample negatives
            # Push target logit above mean of negatives
            negs = torch.randint(0, V, (bb, NEG_SAMPLES), device=device)
            # Ensure negatives don't include the target
            for i in range(bb):
                mask = negs[i] == batch_targets[i]
                negs[i, mask] = (negs[i, mask] + 1) % V
            
            target_logits = last_logits.gather(1, batch_targets.unsqueeze(1))  # (B', 1)
            neg_logits = last_logits.gather(1, negs)  # (B', NEG)
            
            # Hinge loss: max(0, neg_logit - target_logit + margin)
            margin = 1.0
            loss = F.relu(neg_logits - target_logits + margin).mean()
            
            # Trigger loss
            trig_target = torch.zeros(bb, L, device=device)
            trig_target[:, -1] = 1.0
            trig_loss = F.binary_cross_entropy_with_logits(trigger_logits[:bb], trig_target)
            
            total_loss = loss + 2.0 * trig_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(loopy.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            
            if step % LOG_EVERY == 0:
                sps = step / (time.time() - t0)
                # Compute accuracy on this batch
                preds = last_logits.argmax(dim=-1)
                acc = (preds == batch_targets).float().mean().item()
                print(f"step {step:5d}  e{epoch}  len={L}  rank_loss={loss.item():.4f}  trig={trig_loss.item():.4f}  acc={acc:.3f}  {sps:.1f} st/s", flush=True)
            
            if step % SAVE_EVERY == 0:
                torch.save(loopy.state_dict(), Path(SAVE_PATH))
                print(f"  saved", flush=True)

torch.save(loopy.state_dict(), Path(SAVE_PATH))
print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)

# Eval: test first 500 tokens
correct = 0
for tid in token_ids[:500]:
    b = tok.idx2token[tid]
    byte_ids = torch.tensor([[2 + byte for byte in b]], device=device)
    token_logits, trigger_logits = loopy.forward_bytes(byte_ids)
    pred = token_logits[0, -1].argmax().item()
    trig_prob = torch.sigmoid(trigger_logits[0, -1]).item()
    if pred == tid and trig_prob > 0.5:
        correct += 1
print(f"\n{correct}/500 correct ({100*correct/500:.1f}%)", flush=True)
