"""Train loopy time-mix: byte_embed → minGRU → layer 0 time-mix output.

For each byte, the real layer 0 time-mix produces a state.
Our minGRU learns to predict that state from the byte embedding.
Trigger fires at the last byte of each token (suppress real time-mix).
"""
import torch, torch.nn.functional as F, time
from pathlib import Path
import sys; sys.path.insert(0, '.')
from minGRU_pytorch import minGRU
import torch.nn as nn

DEVICE = "cuda"
DIM = 256
LR = 3e-4
STEPS = 5000
BATCH = 64
MAX_BYTES = 24
SAVE_PATH = "experiments/loopy_timemix/model.pt"

device = torch.device(DEVICE)

# ── Load g1g components ──
print("Loading g1g...", flush=True)
model_path = Path.home() / "Documents/models/rwkv7-g1g-byte-iface/model.pth"
ckpt = torch.load(model_path, map_location='cpu', weights_only=True)

# Byte embedding
byte_embed = nn.Embedding(258, 2560)
byte_embed.weight.data = ckpt['byte_embed.weight'].float()
byte_embed = byte_embed.to(device).eval()
for p in byte_embed.parameters(): p.requires_grad = False

# Layer 0 attention (time-mix) weights — extract from checkpoint
# We'll build a simplified time-mix that just does the WKV recurrence
# with the actual weights

# Load the key time-mix parameters
att_keys = ['key', 'receptance', 'value', 'output', 'w1', 'w2', 'a1', 'a2', 'g1']
att_params = {}
for k in att_keys:
    w = ckpt[f'blocks.0.att.{k}.weight'].float().to(device)
    att_params[k] = w

# Also load layernorms
ln1_w = ckpt['blocks.0.ln1.weight'].float().to(device)
ln1_b = ckpt['blocks.0.ln1.bias'].float().to(device)

print(f"Loaded layer 0 attention weights", flush=True)

# ── Simplified time-mix forward ──
def time_mix_step(x, state_xx=None, state_mat=None):
    """One step of layer 0's time-mix.
    x: (1, 2560) — byte embedding
    state_xx: (1, 2560) — previous output
    state_mat: (1, 40, 64, 64) — recurrent matrix state
    Returns: output (1, 2560), new_xx, new_mat
    """
    B, C = x.shape
    H, N = 40, 64
    
    if state_xx is None: state_xx = torch.zeros(1, C, device=x.device)
    if state_mat is None: state_mat = torch.zeros(1, H, N, N, device=x.device)
    
    # Layer norm
    xn = F.layer_norm(x, [C], weight=ln1_w, bias=ln1_b)
    
    # Time-mix with previous output
    xx_diff = state_xx - xn
    
    # Per-head mixing (simplified — skip per-head time_mix_x params)
    r = F.linear(xn, att_params['receptance'])  # (1, 2560)
    k = F.linear(xn, att_params['key'])
    v = F.linear(xn, att_params['value'])
    
    # Reshape to heads
    r_h = r.view(B, H, N)
    k_h = k.view(B, H, N)
    v_h = v.view(B, H, N)
    
    # Simplified WKV: v @ k^T update
    vk = v_h.unsqueeze(-1) @ k_h.unsqueeze(-2)
    
    # Decay (simplified)
    w = torch.sigmoid(F.linear(xn, att_params['w1']) @ att_params['w2'].T).view(B, H, N)
    
    # State update
    state_mat = state_mat * w.unsqueeze(-2).float() + vk.float()
    
    # Output
    out = (state_mat @ r_h.unsqueeze(-1).float()).squeeze(-1).to(x.dtype)
    out = out.reshape(B, C)
    
    # Output projection
    out = F.linear(out, att_params['output'])
    
    return out, xn, state_mat

# ── Loopy time-mix predictor ──
class LoopyTimeMix(nn.Module):
    """minGRU that predicts layer 0 time-mix output from byte embeddings."""
    def __init__(self, dim=256):
        super().__init__()
        self.input_proj = nn.Linear(2560, dim)  # compress byte_embed → dim
        self.gru = minGRU(dim)
        self.output_proj = nn.Linear(dim, 2560)  # predict time-mix output
        self.trigger_head = nn.Linear(dim, 1)
        self.state_xx_proj = nn.Linear(dim, 2560)  # predict xx state
        self.state_mat_proj = nn.Linear(dim, 40 * 64 * 64)  # predict mat state
        
    def forward(self, byte_embs):
        """byte_embs: (B, T, 2560) — sequence of byte embeddings.
        Returns: outputs (B, T, 2560), triggers (B, T), states
        """
        B, T, C = byte_embs.shape
        h = self.input_proj(byte_embs)  # (B, T, dim)
        h = self.gru(h)  # (B, T, dim)
        
        outputs = self.output_proj(h)  # (B, T, 2560)
        triggers = torch.sigmoid(self.trigger_head(h)).squeeze(-1)  # (B, T)
        xx_preds = self.state_xx_proj(h)  # (B, T, 2560)
        mat_preds = self.state_mat_proj(h).view(B, T, 40, 64, 64)  # (B, T, 40, 64, 64)
        
        return outputs, triggers, xx_preds, mat_preds

model = LoopyTimeMix().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# ── Build training data ──
# For each token in TinyStories: run bytes through byte_embed + time_mix
# Capture time-mix output at each byte position

from src.hf_rwkv_tokenizer import RWKV_TOKENIZER
tok = RWKV_TOKENIZER(str(Path("src/rwkv_vocab_v20230424.txt")))

items = []
with open("experiments/tinystories_texts.txt") as f:
    stories = f.read().split("\n---END---\n")

for s in stories[:50]:
    raw = s.encode("utf-8")[:2048]
    tokens = tok.encodeBytes(raw)
    for tid in tokens[:30]:
        if tid >= 65529: continue
        b = tok.idx2token.get(tid, b'')
        if len(b) < 1 or len(b) > MAX_BYTES: continue
        
        # Run bytes through byte_embed + time_mix
        byte_ids = torch.tensor([[2 + byte for byte in b]], device=device)
        embs = byte_embed(byte_ids)  # (1, T, 2560)
        
        # Step through time-mix
        tm_outputs = []
        xx_state, mat_state = None, None
        for t in range(len(b)):
            emb_t = embs[:, t, :]
            out, new_xx, new_mat = time_mix_step(emb_t, xx_state, mat_state)
            tm_outputs.append(out)
            xx_state, mat_state = new_xx, new_mat
        
        tm_out_tensor = torch.stack(tm_outputs, dim=1)  # (1, T, 2560)
        
        # Trigger target: 1 at last byte, 0 elsewhere
        trig_target = torch.zeros(1, len(b), device=device)
        trig_target[0, -1] = 1.0
        
        items.append((embs.detach(), tm_out_tensor.detach(), trig_target))

print(f"Items: {len(items)}", flush=True)

t0 = time.time()
for step in range(STEPS):
    idx = torch.randperm(len(items), device='cpu')[:BATCH]
    batch_embs = torch.cat([items[i][0] for i in idx], dim=0)  # (B, T, 2560)
    batch_targets = torch.cat([items[i][1] for i in idx], dim=0)  # (B, T, 2560)
    batch_trig = torch.cat([items[i][2] for i in idx], dim=0)  # (B, T)
    
    pred_out, pred_trig, _, _ = model(batch_embs)
    
    # Loss: MSE on time-mix output + BCE on trigger
    out_loss = F.mse_loss(pred_out, batch_targets)
    trig_loss = F.binary_cross_entropy(pred_trig, batch_trig)
    loss = out_loss + trig_loss
    
    opt.zero_grad(); loss.backward(); opt.step()
    
    if (step+1) % 500 == 0:
        sps = (step+1) / (time.time() - t0)
        cos = F.cosine_similarity(pred_out.view(-1, 2560), batch_targets.view(-1, 2560)).mean().item()
        trig_acc = ((pred_trig > 0.5) == (batch_trig > 0)).float().mean().item()
        print(f"step {step+1:5d}  out_loss={out_loss.item():.4f}  trig={trig_loss.item():.4f}  cos={cos:.4f}  trig_acc={trig_acc:.3f}  {sps:.1f} st/s", flush=True)
    
    if (step+1) % 2000 == 0:
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(SAVE_PATH))

print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)
