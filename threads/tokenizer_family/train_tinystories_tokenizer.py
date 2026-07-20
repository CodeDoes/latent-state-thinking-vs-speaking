"""Train loopy tokenizer on TinyStories — real text with token boundaries.

Architecture: bytes + position → minGRU → 128D bottleneck → 21K token head
Loss: token prediction at every byte + trigger at token boundaries.
"""
import torch, torch.nn.functional as F, time, json, random
from pathlib import Path
import sys; sys.path.insert(0, '.')
from threads.tokenizer_family.loopy_tokenizer import LoopyTokenizerWithPos
from domains.rwkv.hf_rwkv_tokenizer import RWKV_TOKENIZER

# ════════════ CONFIG ════════════
DEVICE = "cuda"
DIM = 8
LR = 3e-2
BATCH = 4          # stories per batch
MAX_BYTES = 1024   # truncate stories to this many bytes
LOG_EVERY = 20
SAVE_EVERY = 200
TRIGGER_WEIGHT = 2.0
EPOCHS = 5
SAVE_PATH = "experiments/loopy_tokenizer/tinystories_model.pt"
# ════════════════════════════════

tok = RWKV_TOKENIZER(str(Path("domains/rwkv/rwkv_vocab_v20230424.txt")))
V = len(tok.token2idx)

# Load stories and build vocab
with open("experiments/tinystories_texts.txt") as f:
    raw = f.read()
stories = raw.split("\n---END---\n")
stories = [s.strip() for s in stories if s.strip()]
random.shuffle(stories)
stories = stories[:200]  # quick test
print(f"Stories: {len(stories):,}", flush=True)

with open("experiments/tinystories_tokens.json") as f:
    used_tokens = json.load(f)
V_SUB = len(used_tokens)
TOKEN_MAP = {tid: i for i, tid in enumerate(used_tokens)}  # full → sub
REV_MAP = {i: tid for i, tid in enumerate(used_tokens)}   # sub → full
print(f"Token vocab: {V_SUB} (from {V})", flush=True)

device = torch.device(DEVICE)

# Pre-compute byte sequences + targets for all stories
items = []
skipped = 0
print("Pre-computing...", flush=True)
for i, s in enumerate(stories):
    raw_bytes = s.encode("utf-8")
    if len(raw_bytes) > MAX_BYTES:
        raw_bytes = raw_bytes[:MAX_BYTES]
    if len(raw_bytes) < 2:
        skipped += 1
        continue
    
    byte_ids = torch.tensor([[2 + b for b in raw_bytes]], device=device)
    T = len(raw_bytes)
    
    # Tokenize to get spans
    tokens, spans = [], []
    idx = 0
    while idx < len(raw_bytes):
        old = idx
        idx, node, values = tok.root.find_longest(raw_bytes, idx)
        if idx == old:
            idx += 1
            continue
        _, tid = next(iter(values))
        tokens.append(tid)
        spans.append((old, idx))
    
    # Trigger target: 1 at last byte of each token
    trig_target = torch.zeros(1, T, device=device)
    for _, (s, e) in enumerate(spans):
        if e - 1 < T:
            trig_target[0, e - 1] = 1.0
    
    # Token target at each byte: the token ID it belongs to
    tok_target = torch.zeros(1, T, dtype=torch.long, device=device)
    span_mask = torch.zeros(1, T, device=device)
    for ti, (s, e) in enumerate(spans):
        sub_id = TOKEN_MAP.get(tokens[ti], 0)
        for bp in range(s, min(e, T)):
            tok_target[0, bp] = sub_id
            span_mask[0, bp] = 1.0
    
    items.append((byte_ids, trig_target, tok_target, span_mask, len(spans)))

print(f"Items: {len(items)} (skipped {skipped})", flush=True)

# Build model
loopy = LoopyTokenizerWithPos(DIM, V_SUB).to(device)
opt = torch.optim.AdamW(loopy.parameters(), lr=LR)
n = sum(p.numel() for p in loopy.parameters())
print(f"Params: {n:,}", flush=True)

t0 = time.time()
step = 0

for epoch in range(EPOCHS):
    random.shuffle(items)
    
    for bstart in range(0, len(items), BATCH):
        batch = items[bstart:bstart + BATCH]
        total_loss = 0.0
        
        for byte_ids, trig_target, tok_target, span_mask, n_toks in batch:
            T = byte_ids.shape[1]
            token_logits, trigger_logits = loopy.forward_bytes(byte_ids)
            
            # Token loss: cross-entropy at ALL byte positions (masked)
            losses = F.cross_entropy(
                token_logits.view(-1, V_SUB), tok_target.view(-1), reduction="none"
            ).view_as(span_mask)
            tok_loss = (losses * span_mask).sum() / (span_mask.sum() + 1e-8)
            
            # Trigger loss
            trig_loss = F.binary_cross_entropy_with_logits(trigger_logits, trig_target)
            
            loss = tok_loss + TRIGGER_WEIGHT * trig_loss
            total_loss += loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(loopy.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        
        if step % LOG_EVERY == 0:
            sps = step / (time.time() - t0)
            avg_loss = total_loss.item() / len(batch)
            print(f"step {step:6d}  e{epoch}  loss={avg_loss:.4f}  {sps:.1f} st/s", flush=True)
        
        if step % SAVE_EVERY == 0:
            torch.save({
                'model': loopy.state_dict(),
                'rev_map': REV_MAP,
                'tok_map': TOKEN_MAP,
                'dim': DIM,
                'vocab_size': V_SUB,
            }, Path(SAVE_PATH))
            print(f"  saved to {SAVE_PATH}", flush=True)

# Final save
torch.save({
    'model': loopy.state_dict(),
    'rev_map': REV_MAP,
    'tok_map': TOKEN_MAP,
    'dim': DIM,
    'vocab_size': V_SUB,
}, Path(SAVE_PATH))
print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)

# Quick eval: decode a few stories
print("\n--- Eval ---", flush=True)
for s in stories[:3]:
    raw = s.encode("utf-8")[:MAX_BYTES]
    byte_ids = torch.tensor([[2 + b for b in raw]], device=device)
    token_logits, trigger_logits = loopy.forward_bytes(byte_ids)
    trig_probs = torch.sigmoid(trigger_logits)
    
    # Get expected tokens
    tokens, spans = [], []
    idx = 0
    while idx < len(raw):
        old = idx
        idx, node, values = tok.root.find_longest(raw, idx)
        if idx == old: idx += 1; continue
        _, tid = next(iter(values))
        tokens.append(tid)
        spans.append((old, idx))
    
    # Emit tokens at trigger positions
    emitted = []
    for t in range(len(raw)):
        if trig_probs[0, t] > 0.5:
            pred = REV_MAP[token_logits[0, t].argmax().item()]
            emitted.append(pred)
    
    corr = sum(1 for a, b in zip(emitted, tokens) if a == b)
    print(f"  {corr}/{len(tokens)} tokens correct", flush=True)
    if corr > 0:
        # Decode what we got
        try:
            decoded = tok.decodeBytes(emitted).decode("utf-8", errors="replace")
            print(f"  decoded: {decoded[:100]!r}", flush=True)
        except:
            pass
