"""Train the loopy tokenizer (byte→world-token) supervised by TRIE.

Architecture:
    bytes + byte-position → AccumulatorCell (minGRU) → token_logits + trigger

Loss:
    - token_loss: cross-entropy predicting world token ID at each byte position
    - trigger_loss: BCE on trigger flag (1 at last byte of each TRIE token)

Easy-to-tweak constants at the top of the file.
"""
import torch, torch.nn.functional as F, time
from pathlib import Path
import sys; sys.path.insert(0, '.')

from src.loopy_tokenizer import LoopyTokenizerWithPos
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER

# ══════════════════════════ CONFIG — TWEAK ME ══════════════════════════
DEVICE = "cuda"
DIM = 8
LR = 3e-2
GRAD_CLIP = 1.0
TRIGGER_WEIGHT = 2.0
LOG_EVERY = 10
SAVE_EVERY = 100
SAVE_PATH = "experiments/loopy_tokenizer/model.pt"
# ═══════════════════════════════════════════════════════════════════════

# ── Data ──
TEXTS = [
    "Hello World!", "Once upon a time", "The quick brown fox",
    "Machine learning is fun.", "Testing one two three.",
    "The cat sat on the mat.", "She sells sea shells.",
    "How much wood would a woodchuck chuck?",
    "Peter Piper picked a peck of pickled peppers.",
    "abcdefghijklmnopqrstuvwxyz", "1234567890",
    "Foo bar baz", "A quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
]

# ── Tokenizer ──
tok = RWKV_TOKENIZER(str(Path("src/rwkv_vocab_v20230424.txt")))
FULL_VOCAB = len(tok.token2idx)

def find_used_tokens(texts):
    """Return sorted list of token IDs that appear in the training texts."""
    used = set()
    for text in texts:
        raw = text.encode("utf-8")
        idx = 0
        while idx < len(raw):
            old_idx = idx
            idx, node, values = tok.root.find_longest(raw, idx)
            if idx == old_idx: break
            _, token_id = next(iter(values))
            used.add(token_id)
    return sorted(used)

# Map full vocab → trimmed vocab
USED_TOKENS = find_used_tokens(TEXTS)
WORLD_VOCAB = len(USED_TOKENS)  # trimmed vocab size
TOKEN_MAP = {full_id: i for i, full_id in enumerate(USED_TOKENS)}  # full → trimmed

print(f"Full vocab: {FULL_VOCAB:,}, used tokens: {WORLD_VOCAB}", flush=True)

def precompute(texts):
    items = []
    for text in texts:
        raw = text.encode("utf-8")
        byte_ids_list = [2 + b for b in raw]
        tokens, spans = [], []
        idx = 0
        while idx < len(raw):
            old_idx = idx
            idx, node, values = tok.root.find_longest(raw, idx)
            if idx == old_idx: break
            _, token_id = next(iter(values))
            tokens.append(token_id)
            spans.append((old_idx, idx))
        T = len(byte_ids_list)
        byte_ids = torch.tensor([byte_ids_list], device=DEVICE)
        trig_target = torch.zeros(1, T, device=DEVICE)
        for _, (s, e) in enumerate(spans):
            if e - 1 < T: trig_target[0, e - 1] = 1.0
        # Map full token IDs to trimmed indices
        trimmed_tokens = [TOKEN_MAP[t] for t in tokens]
        tok_target = torch.zeros(1, T, dtype=torch.long, device=DEVICE)
        span_mask = torch.zeros(1, T, device=DEVICE)
        for ti, (s, e) in enumerate(spans):
            for bp in range(s, min(e, T)):
                tok_target[0, bp] = trimmed_tokens[ti]
                span_mask[0, bp] = 1.0
        items.append((byte_ids, trimmed_tokens, trig_target, tok_target, span_mask))
    return items

# ── Build (resume from checkpoint if exists) ──
device = torch.device(DEVICE)
loopy = LoopyTokenizerWithPos(DIM, WORLD_VOCAB, trigger_bias=-2.0).to(device)
opt = torch.optim.AdamW(loopy.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
items = precompute(TEXTS)

n_params = sum(p.numel() for p in loopy.parameters())
save_p = Path(SAVE_PATH)
start_step = 0
if save_p.exists():
    loopy.load_state_dict(torch.load(save_p, map_location=device))
    start_step = int(save_p.with_suffix('.step').read_text().strip())
    print(f"Resumed from {save_p} (step {start_step})", flush=True)

print(f"Loopy tokenizer params: {n_params:,}", flush=True)
print(f"dim={DIM}, world_vocab={WORLD_VOCAB}, texts={len(TEXTS)}", flush=True)
print(f"lr={LR}, trigger_weight={TRIGGER_WEIGHT}", flush=True)
print(f"log every {LOG_EVERY}, save every {SAVE_EVERY}", flush=True)

# ── Train ──
t0 = time.time()
step = start_step
while True:
    total_loss = 0.0
    n = 0

    for byte_ids, tokens, trig_target, tok_target, span_mask in items:
        token_logits, triggers = loopy.forward_bytes(byte_ids)

        losses = F.cross_entropy(
            token_logits.view(-1, WORLD_VOCAB), tok_target.view(-1), reduction="none"
        ).view_as(span_mask)
        tok_loss = (losses * span_mask).sum() / (span_mask.sum() + 1e-8)

        trig_loss = F.binary_cross_entropy_with_logits(triggers, trig_target)
        loss = tok_loss + TRIGGER_WEIGHT * trig_loss
        total_loss += loss
        n += 1

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(loopy.parameters(), GRAD_CLIP)
    opt.step()
    opt.zero_grad()

    step += 1
    avg_loss = total_loss.item() / n

    if step % LOG_EVERY == 0:
        sps = step / (time.time() - t0)
        print(f"step {step:5d}  loss={avg_loss:.4f}  {sps:.1f} st/s", flush=True)

    if step % SAVE_EVERY == 0:
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(loopy.state_dict(), save_p)
        save_p.with_suffix('.step').write_text(str(step))
        print(f"  saved to {save_p} (step {step})", flush=True)

    sched.step()

    if avg_loss < 0.01:
        print(f"\nLow loss ({avg_loss:.4f}) — stopping.", flush=True)
        break
