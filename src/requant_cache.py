"""Re-quantize NF4 cache on GPU (one layer at a time)."""
import torch, json
from pathlib import Path
from bitsandbytes.functional import quantize_4bit

CACHE_DIR = Path.home() / "Documents/models/rwkv7-g1g-byte-iface/nf4_cache"
INDEX_PATH = CACHE_DIR / "index.json"
MODEL_PATH = Path.home() / "Documents/models/rwkv7-g1g-byte-iface/model.pth"
index = json.loads(INDEX_PATH.read_text())

print("Loading full state dict on CPU...")
sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)

dev = torch.device('cuda')

for i, k in enumerate(index.keys()):
    cache_path = CACHE_DIR / f"q_{i:04d}.pt"
    data = torch.load(cache_path, map_location='cpu', weights_only=True)
    if 'quant_type' in data:
        print(f"  [{i}/192] ✓ already done")
        continue

    print(f"  [{i}/192] quantizing {k} ...", end=' ', flush=True)
    v_gpu = sd[k].to(device=dev, dtype=torch.float16)
    q, qs = quantize_4bit(v_gpu, compress_statistics=False)

    torch.save({
        'q': q.cpu(),
        'absmax': qs.absmax.cpu(),
        'code': qs.code.cpu(),
        'blocksize': qs.blocksize,
        'shape': qs.shape,
        'dtype': qs.dtype,
        'quant_type': qs.quant_type,
    }, cache_path)

    del v_gpu, q, qs
    torch.cuda.empty_cache()
    print("done")

print("All done")
