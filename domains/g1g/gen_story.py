"""Generate a story using the frozen g1g model via world token IDs.

Bytes → TRIE tokenizer → world token IDs → frozen g1g → world logits → text

No learned front-end, no byte embed swap. Uses the original world tokenizer
and the original model as-is, just with our own forward pass.
"""
import time, sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from pathlib import Path
import json

from domains.rwkv.hf_rwkv_tokenizer import RWKV_TOKENIZER
from domains.g1g.auto_tokenizer import ByteG1GInference  # reuse the stateful step pattern

# Load world tokenizer
vocab_path = Path('domains/rwkv/rwkv_vocab_v20230424.txt')
tok = RWKV_TOKENIZER(str(vocab_path))
VOCAB_SIZE = len(tok.token2idx)
WORLD_PAD = 0

# Load original model (not byte-iface)
MODEL_PATH = '/home/kit/Documents/models/rwkv7-g1g-2.9b-20260526-ctx8192.pth'

# Need to load with world vocab (65536) not byte vocab (258)
# The .pth has 'emb.weight' and 'head.weight' of shape (65536, 2560)
print("Loading original state dict...")
t0 = time.time()
sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
print(f"  {time.time()-t0:.1f}s, {len(sd)} keys")

D = sd['ln_out.weight'].shape[0]
head_size = sd['blocks.0.att.r_k'].shape[-1]
n_heads = D // head_size
n_layers = sum(1 for k in sd if k.startswith('blocks.') and k.endswith('.ln1.weight'))
print(f"  {n_layers} layers, {D} dim, {n_heads}×{head_size} heads")

# Move to GPU with NF4 quantization
import bitsandbytes as bnb
from bitsandbytes.functional import quantize_4bit, dequantize_4bit, QuantState

dev = torch.device('cuda')
print(f"Hybrid quantization (FP16: layers 0-2, 29-31; NF4: rest)...")
t0 = time.time()

NF4_DIR = Path.home() / "Documents/models/rwkv7-g1g-byte-iface/nf4_cache"
NF4_INDEX = NF4_DIR / "index.json"
index = json.loads(NF4_INDEX.read_text()) if NF4_INDEX.exists() else {}

FP16_LAYERS = {0, 1, 2, 29, 30, 31}

for k, v in list(sd.items()):
    layer_num = None
    if k.startswith('blocks.'):
        try:
            layer_num = int(k.split('.')[1])
        except (IndexError, ValueError):
            pass
    
    if k in ('emb.weight', 'head.weight'):
        sd[k] = v.to(device=dev, dtype=torch.bfloat16)
    elif layer_num is not None and layer_num in FP16_LAYERS:
        sd[k] = v.to(device=dev, dtype=torch.bfloat16)
    elif k in index:
        # FFN weights from NF4 cache
        data = torch.load(index[k], map_location=dev, weights_only=True)
        qs = QuantState(
            absmax=data['absmax'], shape=data['shape'],
            code=data['code'], blocksize=data['blocksize'],
            dtype=data['dtype'], quant_type=data['quant_type'],
        )
        sd[k] = (data['q'], qs)
    elif isinstance(v, torch.Tensor):
        sd[k] = v.to(device=dev, dtype=torch.bfloat16)
    # Middle layers use NF4 cache
    elif k in index:
        data = torch.load(index[k], map_location=dev, weights_only=True)
        qs = QuantState(
            absmax=data['absmax'], shape=data['shape'],
            code=data['code'], blocksize=data['blocksize'],
            dtype=data['dtype'], quant_type=data['quant_type'],
        )
        sd[k] = (data['q'], qs)
    elif isinstance(v, torch.Tensor):
        sd[k] = v.to(device=dev, dtype=torch.bfloat16)

torch.cuda.empty_cache()
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB ({time.time()-t0:.1f}s)")

# Stateful inference
states = None

def get_w(key):
    v = sd[key]
    if isinstance(v, tuple):
        q, qs = v
        return dequantize_4bit(q.to(dev), qs).to(dtype=torch.bfloat16)
    return v

@torch.no_grad()
def step(token_id):
    global states
    B = 1
    h = F.embedding(torch.tensor([token_id], device=dev), get_w('emb.weight')).squeeze(0)
    
    if states is None:
        states = [{
            'xx': torch.zeros(D, device=dev, dtype=torch.bfloat16),
            'xx_c': torch.zeros(D, device=dev, dtype=torch.bfloat16),
            'mat': torch.zeros(n_heads, head_size, head_size, device=dev, dtype=torch.float),
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
    logits = h @ sd['head.weight'].T
    if 'head.bias' in sd:
        logits = logits + sd['head.bias'].to(dtype=torch.bfloat16)
    return logits

# Generate
prompt = "Once upon a time,"
print(f"\nTokenizing prompt: {prompt!r}")
tokens = tok.encodeBytes(prompt.encode('utf-8'))
print(f"  {len(tokens)} tokens: {tokens}")

print("Feeding prompt...")
for tid in tokens:
    step(tid)

print("Generating...")
generated_tokens = []
next_tid = tokens[-1]
t0 = time.time()

for i in range(200):
    logits = step(next_tid)
    probs = F.softmax(logits / 0.8, dim=-1)
    # top-p
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    sorted_probs[cumsum - sorted_probs > 0.9] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    idx = int(torch.multinomial(sorted_probs, 1).item())
    next_tid = int(sorted_indices[idx].item())
    if next_tid == WORLD_PAD:
        break
    generated_tokens.append(next_tid)
    if (i + 1) % 25 == 0:
        partial = tok.decodeBytes([next_tid]).decode('utf-8', errors='replace')
        print(f"  [{i+1:3d}] token={next_tid} char={partial!r}")

elapsed = time.time() - t0
out = tok.decodeBytes(generated_tokens).decode('utf-8', errors='replace')
print(f"\n--- Generated in {elapsed:.1f}s ({len(generated_tokens)} tokens) ---")
print(prompt + out)
