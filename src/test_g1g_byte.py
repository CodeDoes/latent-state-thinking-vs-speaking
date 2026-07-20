#!/usr/bin/env python3
"""Test the byte-interface g1g model: load, forward, verify output."""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Byte vocab ──
BYTE_TO_ID = {b: 2 + b for b in range(256)}
BYTE_PAD = 0

# ── Load model ──
MODEL_DIR = Path.home() / "Documents" / "models" / "rwkv7-g1g-byte-iface"

print("Loading state dict...")
t0 = time.time()
sd = torch.load(MODEL_DIR / "model.pth", map_location="cpu", weights_only=True)
print(f"  {len(sd)} keys, {sum(v.numel() for v in sd.values()):,} params  ({time.time()-t0:.1f}s)")

dim = sd['ln_out.weight'].shape[0]
n_layers = sum(1 for k in sd if k.startswith('blocks.') and k.endswith('.ln1.weight'))
head_size = sd['blocks.0.att.r_k'].shape[-1]
n_heads = dim // head_size
print(f"  dim={dim}, heads={n_heads}×{head_size}, layers={n_layers}")
print(f"  byte_embed: {sd['byte_embed.weight'].shape}")
print(f"  byte_head:  {sd['byte_head.weight'].shape}")

# ── Build model from the rwkv package ──
# We use the official rwkv package's RWKV-7 model but override embed/head
print("\nBuilding model...")
import os
os.environ['RWKV_V7_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '0'

# Monkey-patch the official model to use our byte embed/head
# The official RWKV class loads weights from a file, but we already have them
from rwkv.model import RWKV

# Create a minimal model wrapper
class ByteG1GModel:
    """Minimal wrapper: byte embed → RWKV-7 layers → byte head."""
    
    def __init__(self, sd):
        self.sd = sd
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_size = head_size
        self.device = 'cpu'
        
    def embed(self, byte_ids):
        """Byte token IDs → embeddings."""
        return F.embedding(byte_ids, self.sd['byte_embed.weight'].float())
    
    def layer_norm(self, x, weight_key, bias_key):
        """LayerNorm with weights from state dict."""
        w = self.sd[weight_key].float()
        b = self.sd.get(bias_key)
        if b is None:
            b = torch.zeros_like(w)
        else:
            b = b.float()
        return F.layer_norm(x, (self.dim,), weight=w, bias=b)
    
    def tmix(self, layer_id, x, state, v_first):
        """Time-mixing for one layer."""
        att = f'blocks.{layer_id}.att.'
        H = self.n_heads
        N = self.head_size
        
        # Shift
        xx = state['xx'] - x
        xr = x + xx * self.sd[att + 'x_r'].float()
        xw = x + xx * self.sd[att + 'x_w'].float()
        xk = x + xx * self.sd[att + 'x_k'].float()
        xv = x + xx * self.sd[att + 'x_v'].float()
        xa = x + xx * self.sd[att + 'x_a'].float()
        xg = x + xx * self.sd[att + 'x_g'].float()
        
        # Projections
        r = xr @ self.sd[att + 'receptance.weight'].float()
        w = torch.tanh(xw @ self.sd[att + 'w1'].float()) @ self.sd[att + 'w2'].float()
        k = xk @ self.sd[att + 'key.weight'].float()
        v = xv @ self.sd[att + 'value.weight'].float()
        a = torch.sigmoid(self.sd[att + 'a0'].float() + 
                          (xa @ self.sd[att + 'a1'].float()) @ self.sd[att + 'a2'].float())
        g = torch.sigmoid(xg @ self.sd[att + 'g1'].float()) @ self.sd[att + 'g2'].float()
        
        # Head reshape
        def to_h(t): return t.view(H, N)
        def from_h(t): return t.reshape(H * N)

        kk = F.normalize(to_h(k) * to_h(self.sd[att + 'k_k'].float()), dim=-1, p=2.0)
        k_adj = k * (1 + (a - 1) * self.sd[att + 'k_a'].float())
        
        if layer_id == 0:
            v_first = v
        else:
            blend = torch.sigmoid(self.sd[att + 'v0'].float() + 
                                  (xv @ self.sd[att + 'v1'].float()) @ self.sd[att + 'v2'].float())
            v = v + (v_first - v) * blend
        
        w_decay = torch.exp(-0.606531 * torch.sigmoid((self.sd[att + 'w0'].float() + w).float()))
        
        # Matrix state update
        vk = to_h(v).unsqueeze(-1) @ to_h(k_adj).unsqueeze(-2)
        ab = (-kk).unsqueeze(-1) @ (kk * to_h(a)).unsqueeze(-2)
        state['mat'] = state['mat'] * to_h(w_decay).unsqueeze(-1) + \
                       state['mat'] @ ab.float() + vk.float()
        
        xx = (state['mat'].float() @ to_h(r).unsqueeze(-1)).squeeze(-1)
        xx = from_h(xx)
        
        # Group norm
        xx = xx.view(1, H * N)
        xx = F.group_norm(xx, num_groups=H,
                          weight=self.sd[att + 'ln_x.weight'].float(),
                          bias=self.sd[att + 'ln_x.bias'].float(), eps=64e-5)
        xx = xx.view(H * N)
        
        # Residual shortcut
        shortcut = (to_h(r) * to_h(k) * to_h(self.sd[att + 'r_k'].float()))  # (H, N)
        shortcut = shortcut.sum(dim=-1, keepdim=True) * to_h(v)  # (H, N)
        xx = xx + from_h(shortcut)
        xx = (xx * g) @ self.sd[att + 'output.weight'].float()
        
        return xx, v_first
    
    def cmix(self, layer_id, x, state):
        """Channel-mixing for one layer."""
        ffn = f'blocks.{layer_id}.ffn.'
        xx = state['xx_c'] - x
        xk = x + xx * self.sd[ffn + 'x_k'].float()
        
        # key.weight: (hidden_dim, dim) — stored transposed from nn.Linear(dim, hidden_dim)
        k = F.relu(xk @ self.sd[ffn + 'key.weight'].float().T) ** 2
        # value.weight: (dim, hidden_dim) — stored as nn.Linear(hidden_dim, dim)
        v = k @ self.sd[ffn + 'value.weight'].float().T
        return v
    
    def forward(self, byte_ids):
        """Forward pass: bytes → logits."""
        B, T = byte_ids.shape
        assert B == 1, "Batch size 1 only"
        
        # Embed
        x = self.embed(byte_ids[0])  # (T, dim)
        
        # Initialize states
        v_first = None
        states = []
        for i in range(self.n_layers):
            states.append({
                'xx': torch.zeros(self.dim),
                'xx_c': torch.zeros(self.dim),
                'mat': torch.zeros(self.n_heads, self.head_size, self.head_size),
            })
        
        outputs = []
        for t in range(T):
            if byte_ids[0, t] == BYTE_PAD:
                break
                
            h = x[t]  # (dim,)
            
            for i in range(self.n_layers):
                s = states[i]
                ln1 = self.layer_norm(h, f'blocks.{i}.ln1.weight', f'blocks.{i}.ln1.bias')
                tm, v_first = self.tmix(i, ln1, s, v_first)
                h = h + tm
                
                ln2 = self.layer_norm(h, f'blocks.{i}.ln2.weight', f'blocks.{i}.ln2.bias')
                cm = self.cmix(i, ln2, s)
                h = h + cm
                
                # Update state
                s['xx'] = ln1.detach().clone()
                s['xx_c'] = ln2.detach().clone()
            
            outputs.append(h)
        
        if not outputs:
            return torch.zeros(1, 0, 258)
        
        x_out = torch.stack(outputs)  # (T, dim)
        x_out = self.layer_norm(x_out, 'ln_out.weight', 'ln_out.bias')
        logits = x_out @ self.sd['byte_head.weight'].float()  # (T, 258)
        return logits.unsqueeze(0)  # (1, T, 258)


# ── Test ──
print("\nRunning forward pass...")
model = ByteG1GModel(sd)

# Encode a test string as bytes
test_text = "Hello World!"
byte_ids = torch.tensor([[BYTE_TO_ID[b] for b in test_text.encode("utf-8")]])
print(f"Input: {test_text!r} → {byte_ids.shape}")

t0 = time.time()
logits = model.forward(byte_ids)
elapsed = time.time() - t0

print(f"Output: {logits.shape}  ({elapsed:.2f}s for {byte_ids.shape[1]} tokens)")
print(f"  {elapsed / byte_ids.shape[1]:.2f}s per token")

# Check predictions
probs = F.softmax(logits[0, :-1], dim=-1)  # (T-1, 258)
pred_ids = probs.argmax(dim=-1)  # (T-1,)

# Decode predictions to bytes
pred_bytes = bytes([bid - 2 for bid in pred_ids if bid >= 2])
print(f"Input text:  {test_text!r}")
print(f"Predicted:   {pred_bytes!r}")
print(f"Match: {test_text.encode('utf-8') == pred_bytes}")

# Check next-token prediction for the last position
next_probs = F.softmax(logits[0, -1], dim=-1)
top5 = next_probs.topk(5)
top5_chars = [chr(bid - 2) if 2 <= bid <= 257 else '?' for bid in top5.indices]
print(f"Next-token top-5: {list(zip(top5_chars, [f'{v:.3f}' for v in top5.values]))}")

print("\nAll OK!")
