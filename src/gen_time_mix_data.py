"""Fast GPU-based training data generation for time-mix targets.

Uses the NF4-quantized byte-interface model on GPU.
For each token from TinyStories:
1. Get its embedding from the ORIGINAL model's emb.weight
2. Process through layer 0's time-mix (NF4 quantized weights)
3. Capture tm_out + state

This is ~1000× faster than CPU.
"""
import torch, torch.nn.functional as F, time, json, gc
from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.hybrid_tokenizer import encode, token_bytes
from bitsandbytes.functional import dequantize_4bit, QuantState

DEVICE = torch.device('cuda')
DTYPE = torch.bfloat16
MAX_STORIES = 2000
TOKENS_PER_STORY = 100
MAX_BYTES = 24
DATA_DIR = Path("experiments/byte_time_mix/training_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORY_KEY = DATA_DIR / "_stories_done.txt"

# ── Load model weights ──
print("Loading model...", flush=True)
MODEL_PATH = Path.home() / "Documents/models/rwkv7-g1g-2.9b-20260526-ctx8192.pth"
sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)

# Weights for layer 0 time-mix (loaded from original model)
D, H, N = 2560, 40, 64

# Original word embedding
emb = sd['emb.weight'].to(device=DEVICE, dtype=DTYPE)

# Layer 0 weights (from original model, NO quantization for these)
l0 = {}
for k, v in sd.items():
    if k.startswith('blocks.0.'):
        l0[k] = v.to(device=DEVICE, dtype=DTYPE)
del sd  # free CPU memory
gc.collect()

print(f"Layer 0 weights: {len(l0)} tensors on GPU", flush=True)

# ── Load TinyStories ──
print("Loading TinyStories...", flush=True)
with open("experiments/tinystories_texts.txt") as f:
    stories = f.read().split("\n---END---\n")
print(f"  {len(stories)} stories", flush=True)

# ── Resume ──
stories_done = set()
if STORY_KEY.exists():
    for line in STORY_KEY.read_text().strip().splitlines():
        if line.strip():
            stories_done.add(int(line.strip()))
    print(f"  {len(stories_done)} stories already done", flush=True)

# ── GPU time-mix step ──
def time_mix_step_gpu(h, state):
    """One token through layer 0 time-mix on GPU."""
    if state is None:
        xx = torch.zeros(D, device=DEVICE, dtype=DTYPE)
        mat = torch.zeros(H, N, N, device=DEVICE, dtype=torch.float32)
    else:
        xx, mat = state

    ln1 = F.layer_norm(h, (D,),
        weight=l0['blocks.0.ln1.weight'], bias=l0['blocks.0.ln1.bias'])
    xx_diff = xx - ln1
    xr = ln1 + xx_diff * l0['blocks.0.att.x_r'].squeeze()
    xw = ln1 + xx_diff * l0['blocks.0.att.x_w'].squeeze()
    xk = ln1 + xx_diff * l0['blocks.0.att.x_k'].squeeze()
    xv = ln1 + xx_diff * l0['blocks.0.att.x_v'].squeeze()
    xa = ln1 + xx_diff * l0['blocks.0.att.x_a'].squeeze()
    xg = ln1 + xx_diff * l0['blocks.0.att.x_g'].squeeze()

    r = xr @ l0['blocks.0.att.receptance.weight']
    w_tmp = torch.tanh(xw @ l0['blocks.0.att.w1']) @ l0['blocks.0.att.w2']
    k = xk @ l0['blocks.0.att.key.weight']
    v = xv @ l0['blocks.0.att.value.weight']
    a = torch.sigmoid(l0['blocks.0.att.a0'].squeeze()
                       + (xa @ l0['blocks.0.att.a1']) @ l0['blocks.0.att.a2'])
    g = torch.sigmoid(xg @ l0['blocks.0.att.g1']) @ l0['blocks.0.att.g2']

    def to_h(t): return t.view(H, N)
    def from_h(t): return t.reshape(D)
    r_h, k_h, v_h, a_h, g_h, w_h = map(to_h, [r, w_tmp, k, v, a, g])

    kk = F.normalize(k_h * to_h(l0['blocks.0.att.k_k'].squeeze()), dim=-1, p=2.0)
    k_adj = k_h * (1 + (a_h - 1) * to_h(l0['blocks.0.att.k_a'].squeeze()))

    w0 = l0['blocks.0.att.w0'].squeeze()
    w_decay = torch.exp(-0.606531 * torch.sigmoid((to_h(w0) + w_h).float()))

    vk = v_h.unsqueeze(-1) @ k_adj.unsqueeze(-2)
    ab = (-kk).unsqueeze(-1) @ (kk * a_h).unsqueeze(-2)
    mat = (mat * w_decay.unsqueeze(-2).float()
           + (mat @ ab.float())
           + vk.float())

    out_h = (mat.to(dtype=DTYPE) @ r_h.unsqueeze(-1)).squeeze(-1)
    out_flat = from_h(out_h)
    out_flat = F.group_norm(out_flat.view(1, D), num_groups=H,
        weight=l0['blocks.0.att.ln_x.weight'],
        bias=l0['blocks.0.att.ln_x.bias'], eps=64e-5).view(D)
    shortcut = (r_h * k_h * l0['blocks.0.att.r_k']).sum(dim=-1, keepdim=True) * v_h
    out_flat = out_flat + from_h(shortcut)
    out_flat = out_flat * g
    tm_out = out_flat @ l0['blocks.0.att.output.weight']

    return tm_out, (ln1.detach().clone(), mat.detach().clone())

# ── Process ──
t0 = time.time()
stats = {'done': len(stories_done), 'samples': 0}

for si in range(len(stories)):
    if si >= MAX_STORIES:
        break
    if si in stories_done:
        continue

    story = stories[si]
    raw = story.encode('utf-8')[:8192]
    token_ids = encode(raw.decode('utf-8', errors='replace'))

    story_samples = []
    state = None
    for pi, tid in enumerate(token_ids[:TOKENS_PER_STORY]):
        if tid >= 65529:
            continue
        bs = token_bytes(tid)
        if len(bs) == 0 or len(bs) > MAX_BYTES:
            continue

        h = emb[tid]
        tm_out, state = time_mix_step_gpu(h, state)

        byte_values = [2 + b for b in bs]
        padded = byte_values + [0] * (MAX_BYTES - len(byte_values))
        story_samples.append({
            'bytes': padded,
            'num_bytes': len(bs),
            'tm_out': tm_out.cpu().clone(),
            'xx': state[0].cpu().clone(),
            'mat': state[1].cpu().clone(),
        })

    if story_samples:
        torch.save(story_samples, DATA_DIR / f"story_{si:05d}.pt")
        with open(STORY_KEY, 'a') as f:
            f.write(f"{si}\n")
        stories_done.add(si)

    stats['done'] += 1
    stats['samples'] += len(story_samples)

    if stats['done'] % 100 == 0:
        dt = time.time() - t0
        sps = stats['done'] / dt * 60
        print(f"  {stats['done']:4d} stories, {stats['samples']:6d} samples, "
              f"{dt:.0f}s ({sps:.1f} st/min)", flush=True)

print(f"\nDone: {stats['done']} stories, {stats['samples']} samples, "
      f"{(time.time()-t0)/60:.1f} min", flush=True)
