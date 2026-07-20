"""Auto-tokenizer with minGRU: bytes → latent → bytes.

Encoder: minGRU reads bytes in parallel, produces compact latent + trigger
Decoder: minGRU reads latent + trigger, reconstructs bytes

The latent vector is the "token" that feeds into the frozen g1g model.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from minGRU_pytorch import minGRU
from domains.rwkv.hf_rwkv_tokenizer import RWKV_TOKENIZER

BYTE_VOCAB = 258
BYTE_PAD = 0
BYTE_TO_ID = {b: 2 + b for b in range(256)}
ID_TO_BYTE = {v: k for k, v in BYTE_TO_ID.items()}

tok = RWKV_TOKENIZER(str(Path(__file__).parent / "rwkv_vocab_v20230424.txt"))
WORLD_VOCAB = len(tok.token2idx)

def encode(text: str, max_len: int = 512) -> list[int]:
    raw = text.encode("utf-8")
    ids = [BYTE_TO_ID[b] for b in raw]
    return (ids[:max_len] if len(ids) > max_len else ids + [BYTE_PAD] * (max_len - len(ids)))

def decode(ids) -> str:
    return bytes(ID_TO_BYTE.get(i, 0) for i in ids if i >= 2).decode("utf-8", errors="replace")

# ── Encoder: bytes → latent ──
class TriEncoder(nn.Module):
    def __init__(self, dim: int = 128, latent_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(BYTE_VOCAB, dim, padding_idx=BYTE_PAD)
        self.gru = minGRU(dim)
        # Produce latent vector (feeds into frozen model)
        self.latent_head = nn.Linear(dim, latent_dim, bias=False)
        self.byte_head = nn.Linear(dim, BYTE_VOCAB, bias=False)
        self.trigger_head = nn.Linear(dim, 1)

    def forward(self, byte_ids: torch.Tensor, prev_hidden=None):
        x = self.embed(byte_ids)  # (B, T, dim)
        h = self.gru(x, prev_hidden)  # (B, T, dim)
        byte_logits = self.byte_head(h)  # (B, T, 258)
        latent = self.latent_head(h)  # (B, T, latent_dim)
        triggers = self.trigger_head(h).squeeze(-1)  # (B, T)
        final_hidden = h[:, -1:]  # (B, 1, dim)
        return byte_logits, latent, triggers, final_hidden

# ── Decoder: latent → bytes ──
class TriDecoder(nn.Module):
    def __init__(self, dim: int = 128, latent_dim: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim + 1, dim, bias=False)
        self.gru = minGRU(dim)
        self.byte_head = nn.Linear(dim, BYTE_VOCAB, bias=False)
        self.trigger_head = nn.Linear(dim, 1)

    def forward(self, latent: torch.Tensor, triggers: torch.Tensor, prev_hidden=None):
        trig_probs = torch.sigmoid(triggers).unsqueeze(-1)
        x = self.input_proj(torch.cat([latent, trig_probs], dim=-1))
        h = self.gru(x, prev_hidden)
        byte_logits = self.byte_head(h)
        out_triggers = self.trigger_head(h).squeeze(-1)
        final_hidden = h[:, -1:]
        return byte_logits, out_triggers, final_hidden

# ── Full auto-tokenizer ──
class AutoTokenizer(nn.Module):
    def __init__(self, dim: int = 128, latent_dim: int = 256):
        super().__init__()
        self.encoder = TriEncoder(dim, latent_dim)
        self.decoder = TriDecoder(dim, latent_dim)

    def encode(self, byte_ids: torch.Tensor):
        """Bytes → latent vectors (for feeding into frozen model)."""
        byte_logits, latent, triggers, hidden = self.encoder(byte_ids)
        return latent, triggers, hidden

    def decode(self, latent: torch.Tensor, triggers: torch.Tensor):
        """Latent → reconstructed bytes."""
        dec_logits, dec_triggers, hidden = self.decoder(latent, triggers)
        return dec_logits, dec_triggers

    def forward(self, byte_ids: torch.Tensor):
        byte_logits, latent, triggers, enc_hidden = self.encoder(byte_ids)
        dec_logits, dec_triggers, dec_hidden = self.decoder(latent.detach(), triggers.detach())
        return byte_logits, dec_logits, triggers, dec_triggers, latent

# ── Train ──
if __name__ == "__main__":
    device = torch.device("cuda")
    model = AutoTokenizer(dim=128, latent_dim=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    texts = [
        "Hello World!", "Once upon a time", "The quick brown fox",
        "Machine learning", "Testing one two three", "Foo bar baz",
        "abcdefghijklmnop", "1234567890", "The cat sat",
        "She sells sea shells", "How much wood", "Peter Piper",
    ]

    import time
    t0 = time.time()
    for step in range(1000):
        total_loss = 0
        for text in texts:
            raw = text.encode("utf-8")
            ids = torch.tensor([encode(text, max_len=48)], device=device)
            byte_logits, dec_logits, triggers, dec_triggers, latent = model(ids)

            # Encoder loss: next byte prediction
            e_loss = F.cross_entropy(byte_logits.view(-1, BYTE_VOCAB), ids.view(-1))
            # Decoder loss: reconstruct bytes
            d_loss = F.cross_entropy(dec_logits.view(-1, BYTE_VOCAB), ids.view(-1))
            loss = e_loss + d_loss
            total_loss = total_loss + loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if (step + 1) % 200 == 0:
            sps = (step + 1) / (time.time() - t0)
            print(f"step {step+1:4d}  loss={total_loss.item()/len(texts):.4f}  {sps:.1f} st/s", flush=True)

    print(f"\n--- Test ---", flush=True)
    for text in texts[:5]:
        raw = text.encode("utf-8")
        ids = torch.tensor([encode(text, max_len=48)], device=device)
        byte_logits, dec_logits, triggers, dec_triggers, latent = model(ids)
        pred = decode(dec_logits.argmax(dim=-1)[0].tolist()[:len(raw)])
        match = "✓" if text == pred else "✗"
        print(f"  {match} {text!r} -> {pred!r}", flush=True)
