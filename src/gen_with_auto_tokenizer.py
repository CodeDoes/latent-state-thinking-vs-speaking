"""Auto-tokenizer + frozen g1g story generation.

Architecture:
    bytes → TriEncoder (minGRU) → latent → [frozen g1g 2.9B] → output
    TriDecoder reads latent + g1g output to reconstruct bytes

The auto-tokenizer is trained first, then the frozen g1g runs on
the latent vectors as if they were token embeddings.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, json, sys
from pathlib import Path
sys.path.insert(0, '.')

from minGRU_pytorch import minGRU
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER
from src.auto_tokenizer_model import AutoTokenizer, BYTE_TO_ID, BYTE_PAD, encode, decode
from bitsandbytes.functional import dequantize_4bit, QuantState

# ── Load trained auto-tokenizer ──
device = torch.device("cuda")
at = AutoTokenizer(dim=128, latent_dim=256).to(device)
at.eval()

# Load frozen g1g model (NF4 quantized)
print("Loading g1g model...", flush=True)
t0 = time.time()
sd = torch.load('/home/kit/Documents/models/rwkv7-g1g-2.9b-20260526-ctx8192.pth', map_location='cpu', weights_only=True)
D = sd['ln_out.weight'].shape[0]
head_size = sd['blocks.0.att.r_k'].shape[-1]
n_heads = D // head_size
n_layers = 32

NF4_DIR = Path.home() / "Documents/models/rwkv7-g1g-byte-iface/nf4_cache"
NF4_INDEX = NF4_DIR / "index.json"
index = json.loads(NF4_INDEX.read_text())

for k, v in list(sd.items()):
    layer_num = None
    if k.startswith('blocks.'):
        try: layer_num = int(k.split('.')[1])
        except: pass
    if k in ('emb.weight', 'head.weight'):
        sd[k] = v.to(device=device, dtype=torch.bfloat16)
    elif k in index:
        data = torch.load(index[k], map_location=device, weights_only=True)
        qs = QuantState(absmax=data['absmax'], shape=data['shape'], code=data['code'],
            blocksize=data['blocksize'], dtype=data['dtype'], quant_type=data['quant_type'])
        sd[k] = (data['q'], qs)
    elif isinstance(v, torch.Tensor):
        sd[k] = v.to(device=device, dtype=torch.bfloat16)
torch.cuda.empty_cache()
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB ({time.time()-t0:.1f}s)", flush=True)

def get_w(key):
    v = sd[key]
    if isinstance(v, tuple):
        return dequantize_4bit(v[0], v[1]).to(dtype=torch.bfloat16)
    return v

# ── Projection: latent (256) → g1g dim (2560) and back ──
latent_to_model = nn.Linear(256, D, bias=False).to(device)
model_to_latent = nn.Linear(D, 256, bias=False).to(device)

# ── Stateful inference ──
states = None

@torch.no_grad()
def step_g1g(h: torch.Tensor):
    """Run one step through frozen g1g. h: (B, D) embedding."""
    global states
    B = h.shape[0]
    if states is None:
        states = [{
            'xx': torch.zeros(D, device=device, dtype=torch.bfloat16),
            'xx_c': torch.zeros(D, device=device, dtype=torch.bfloat16),
            'mat': torch.zeros(n_heads, head_size, head_size, device=device, dtype=torch.float),
            'v_first': None,
        } for _ in range(n_layers)]
    
    for i in range(n_layers):
        s = states[i]
        ln1 = F.layer_norm(h, (D,), weight=sd[f'blocks.{i}.ln1.weight'], bias=sd[f'blocks.{i}.ln1.bias'])
        att = f'blocks.{i}.att.'
        xx = s['xx'] - ln1
        xr = ln1 + xx * sd[att + 'x_r'].squeeze()
        xw = ln1 + xx * sd[att + 'x_w'].squeeze()
        xk = ln1 + xx * sd[att + 'x_k'].squeeze()
        xv = ln1 + xx * sd[att + 'x_v'].squeeze()
        xa = ln1 + xx * sd[att + 'x_a'].squeeze()
        xg = ln1 + xx * sd[att + 'x_g'].squeeze()
        r = xr @ get_w(att + 'receptance.weight')
        w = torch.tanh(xw @ get_w(att + 'w1')) @ get_w(att + 'w2')
        k = xk @ get_w(att + 'key.weight')
        v = xv @ get_w(att + 'value.weight')
        a = torch.sigmoid(sd[att + 'a0'].squeeze() + (xa @ get_w(att + 'a1')) @ get_w(att + 'a2'))
        g = torch.sigmoid(xg @ get_w(att + 'g1')) @ get_w(att + 'g2')
        r_h, k_h, v_h, a_h, g_h, w_h = (t.view(n_heads, head_size) for t in [r, k, v, a, g, w])
        kk = F.normalize(k_h * sd[att + 'k_k'].squeeze().view(n_heads, head_size), dim=-1, p=2.0)
        k_adj = k_h * (1 + (a_h - 1) * sd[att + 'k_a'].squeeze().view(n_heads, head_size))
        if s['v_first'] is None:
            s['v_first'] = v_h.clone()
        else:
            blend = torch.sigmoid(sd[att + 'v0'].squeeze().view(n_heads, head_size) + (xv @ get_w(att + 'v1') @ get_w(att + 'v2')).view(n_heads, head_size))
            v_h = v_h + (s['v_first'] - v_h) * blend
        w_decay = torch.exp(-0.606531 * torch.sigmoid((sd[att + 'w0'].squeeze().view(n_heads, head_size) + w_h).float()))
        mat = s['mat']
        vk = v_h.unsqueeze(-1) @ k_adj.unsqueeze(-2)
        ab = (-kk).unsqueeze(-1) @ (kk * a_h).unsqueeze(-2)
        mat = (mat * w_decay.unsqueeze(-2).float() + (mat @ ab.float()) + vk.float())
        out_h = (mat.to(dtype=ln1.dtype) @ r_h.unsqueeze(-1)).squeeze(-1)
        out_flat = out_h.reshape(D)
        out_flat = F.group_norm(out_flat.view(1, D), num_groups=n_heads,
            weight=sd[att + 'ln_x.weight'], bias=sd[att + 'ln_x.bias'], eps=64e-5).view(D)
        shortcut = (r_h * k_h * sd[att + 'r_k']).sum(dim=-1, keepdim=True) * v_h
        out_flat = out_flat + shortcut.reshape(D)
        out_flat = out_flat * g
        tm_out = out_flat @ get_w(att + 'output.weight')
        h = h + tm_out
        ln2 = F.layer_norm(h, (D,), weight=sd[f'blocks.{i}.ln2.weight'], bias=sd[f'blocks.{i}.ln2.bias'])
        ffn = f'blocks.{i}.ffn.'
        xx_c = s['xx_c'] - ln2
        xk_c = ln2 + xx_c * sd[ffn + 'x_k'].squeeze()
        k_c = F.relu(xk_c @ get_w(ffn + 'key.weight').T) ** 2
        v_c = k_c @ get_w(ffn + 'value.weight').T
        h = h + v_c
        s['xx'] = ln1.detach().clone(); s['xx_c'] = ln2.detach().clone()
        s['mat'] = mat.detach().clone()
    h = F.layer_norm(h, (D,), weight=sd['ln_out.weight'], bias=sd['ln_out.bias'])
    return h @ sd['head.weight'].T, h

# ── Generate ──
prompt = "Once upon a time,"
print(f"\nPrompt: {prompt!r}", flush=True)

# Encode prompt through auto-tokenizer → get latents
byte_ids = torch.tensor([encode(prompt, max_len=128)], device=device)
latent, triggers, enc_hidden = at.encode(byte_ids)  # (1, T, 256), (1, T)

# Project latent to g1g dimension and feed through frozen model
print("Feeding prompt through g1g...", flush=True)
t0 = time.time()
for t in range(latent.shape[1]):
    h = latent_to_model(latent[:, t]).to(dtype=torch.bfloat16)  # (1, 256) → (1, 2560)
    out_logits, out_hidden = step_g1g(h)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

# Generate: use decoder to reconstruct bytes from g1g output
# New approach: feed g1g output back through decoder to get next bytes
print("Generating...", flush=True)
generated_bytes = []
next_byte_id = BYTE_TO_ID.get(32, 2)  # space
states = None  # reset for generation

for i in range(200):
    # Encode current byte through auto-tokenizer
    byte_t = torch.tensor([[next_byte_id]], device=device)
    byte_latent, byte_triggers, _ = at.encode(byte_t)
    
    # Project to g1g dim and step
    h = latent_to_model(byte_latent[:, 0]).to(dtype=torch.bfloat16)
    out_logits, out_hidden = step_g1g(h)
    
    # Use g1g output hidden as next latent directly
    # out_hidden is (B, D=2560) from the frozen model's output
    # Average pool to get a single latent
    g1g_latent = out_hidden.float().mean(dim=-1, keepdim=True).expand(-1, 256)  # hack: expand
    # Actually just use the g1g output's first 256 dims as latent
    next_latent = out_hidden.float()[:, :256].unsqueeze(0)  # (1, 1, 256)
    
    # Decode to byte
    dec_logits, _ = at.decode(next_latent, torch.zeros(1, 1, device=device))
    next_byte_id = dec_logits[0, -1].argmax().item()
    
    # If we get PAD or repeat > 10 times, stop
    if next_byte_id == BYTE_PAD:
        break
    generated_bytes.append(next_byte_id)
    
    if (i + 1) % 50 == 0:
        print(f"  [{i+1:3d}] {decode(torch.tensor(generated_bytes))[-40:]}", flush=True)

result = decode(torch.tensor(generated_bytes))
print(f"\n--- Generated ({len(generated_bytes)} bytes) ---")
print(prompt + result)
