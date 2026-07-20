"""Build a bf16 on-disk cache for the g1g weights that are NOT NF4-quantized.

The NF4 cache (nf4_cache/) covers the big att/ffn linears of layers 4-27.
Everything else (norms, scalars, embeddings, first/last layers) is kept in
bf16. This script writes those 872 tensors to disk as bf16 so the loader
never has to torch.load the full 5.5GB fp32 model.pth.

Output: <model_dir>/bf16_cache/index.json + q_XXXX.pt (each a bf16 tensor).

Usage:
    python3 src/build_bf16_cache.py
"""
import json
import torch
from pathlib import Path

MODEL_DIR = Path.home() / "Documents/models/rwkv7-g1g-byte-iface"
MODEL_PATH = MODEL_DIR / "model.pth"
NF4_INDEX = MODEL_DIR / "nf4_cache" / "index.json"
BF16_DIR = MODEL_DIR / "bf16_cache"
BF16_INDEX = BF16_DIR / "index.json"


def main():
    BF16_DIR.mkdir(exist_ok=True)
    nf4_keys = set(json.loads(NF4_INDEX.read_text()).keys())

    print("Loading full fp32 state dict on CPU (one-time)...")
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

    keep = [k for k in sd.keys() if k not in nf4_keys]
    print(f"Writing {len(keep)} bf16 tensors to {BF16_DIR} ...")

    index = {}
    for i, k in enumerate(keep):
        v = sd[k]
        if isinstance(v, torch.Tensor):
            t = v.to(dtype=torch.bfloat16)
            cache_path = BF16_DIR / f"q_{i:04d}.pt"
            torch.save(t, cache_path)
            index[k] = str(cache_path.relative_to(MODEL_DIR))
            del t, v
        else:
            index[k] = v  # non-tensor metadata (none expected, but safe)

    BF16_INDEX.write_text(json.dumps(index, indent=2))
    del sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Done. bf16 cache index -> {BF16_INDEX}")


if __name__ == "__main__":
    main()
