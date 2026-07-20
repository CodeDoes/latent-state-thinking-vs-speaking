#!/usr/bin/env python3
"""Convert RWKV-7 g1g to byte-interface version.

Creates a new model file with:
- Byte embed (258 × 2560) instead of 65K embed
- Byte head (258 × 2560) instead of 65K head
- All 32 layers frozen, copied verbatim from original

Usage:
    PYTHONPATH=. python src/convert_g1g_to_byte.py

Output: ~/Documents/models/rwkv7-g1g-byte-iface/
"""

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from domains.rwkv.rwkv_nano import RWKV7Block

MODEL_PATH = Path.home() / "Documents" / "models" / "rwkv7-g1g-2.9b-20260526-ctx8192.pth"
OUTPUT_DIR = Path.home() / "Documents" / "models" / "rwkv7-g1g-byte-iface"


def main():
    print(f"Loading original model from {MODEL_PATH}...")
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    print(f"  {len(sd)} keys, {sum(v.numel() for v in sd.values()):,} total params")

    # Verify structure
    dim = sd['ln_out.weight'].shape[0]
    n_layers = sum(1 for k in sd if k.startswith('blocks.') and k.endswith('.ln1.weight'))
    head_size = sd['blocks.0.att.r_k'].shape[-1]
    n_heads = dim // head_size
    
    print(f"  dim={dim}, heads={n_heads}×{head_size}, layers={n_layers}")
    print(f"  embed: {sd['emb.weight'].shape}  ({sd['emb.weight'].numel():,} params)")
    print(f"  head:  {sd['head.weight'].shape}  ({sd['head.weight'].numel():,} params)")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build byte-interface state dict
    byte_sd = {}
    BYTE_VOCAB = 258
    BYTE_PAD = 0

    # ── Byte embed ──
    byte_embed = nn.Embedding(BYTE_VOCAB, dim, padding_idx=BYTE_PAD)
    with torch.no_grad():
        byte_embed.weight.zero_()
        # Copy first 256 rows from original embed (tokens 0-255 = single bytes)
        # Original vocab: token 0 = empty (""), tokens 1-256 = bytes 0x00-0xFF
        # So byte value bv maps to original token bv+1
        for bv in range(256):
            byte_id = 2 + bv  # byte vocab: PAD=0, UNK=1, bytes 2..257
            orig_token_id = bv + 1  # skip token 0 (empty)
            byte_embed.weight.data[byte_id].copy_(sd['emb.weight'][orig_token_id])
        # PAD and UNK get byte-0 embedding (original token 1)
        byte_embed.weight.data[0].copy_(sd['emb.weight'][1])
        byte_embed.weight.data[1].copy_(sd['emb.weight'][1])
    byte_sd['byte_embed.weight'] = byte_embed.weight

    # ── Byte head (no bias — matches original) ──
    byte_head = nn.Linear(dim, BYTE_VOCAB, bias=False)
    with torch.no_grad():
        byte_head.weight.zero_()
        for bv in range(256):
            byte_id = 2 + bv
            orig_token_id = bv + 1
            byte_head.weight.data[byte_id].copy_(sd['head.weight'][orig_token_id])
        byte_head.weight.data[0].copy_(sd['head.weight'][1])
        byte_head.weight.data[1].copy_(sd['head.weight'][1])
    byte_sd['byte_head.weight'] = byte_head.weight

    # ── Copy all layer weights verbatim ──
    for k, v in sd.items():
        if k.startswith('blocks.') or k.startswith('ln_out.') or k == 'emb.weight' or k == 'head.weight' or k == 'head.bias':
            byte_sd[k] = v

    # ── Config ──
    config = {
        "model": "rwkv7-g1g-byte-iface",
        "original": "rwkv7-g1g-2.9b-20260526-ctx8192",
        "dim": dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_size": head_size,
        "vocab_size": 258,
        "original_vocab_size": 65536,
        "byte_embed_params": byte_embed.weight.numel(),
        "byte_head_params": byte_head.weight.numel(),
        "original_embed_params": sd['emb.weight'].numel(),
        "original_head_params": sd['head.weight'].numel(),
        "frozen_params": sum(v.numel() for k, v in sd.items() 
                           if k.startswith('blocks.') or k.startswith('ln_out.')),
        "total_params": sum(v.numel() for v in byte_sd.values()),
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2))

    # ── Save ──
    model_path = OUTPUT_DIR / "model.pth"
    torch.save(byte_sd, model_path)
    
    # Also save original tokenizer files for reference
    for fname in ["rwkv_vocab_v20230424.txt", "hf_rwkv_tokenizer.py"]:
        src = Path(__file__).parent / fname
        if src.exists():
            shutil.copy2(src, OUTPUT_DIR / fname)

    # Summary
    frozen = config['frozen_params']
    trainable = config['byte_embed_params'] + config['byte_head_params']
    print()
    print(f"Output: {OUTPUT_DIR}/")
    print(f"  model.pth ({model_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  config.json")
    print(f"  rwkv_vocab_v20230424.txt (for reference)")
    print()
    print(f"Params:")
    print(f"  Frozen (32 layers + ln_out): {frozen:>12,}")
    print(f"  Trainable (byte embed+head): {trainable:>12,}")
    print(f"  Total:                      {frozen + trainable:>12,}")
    print(f"  Byte interface: {100 * trainable / (frozen + trainable):.2f}% of total")
    print(f"  Original embed was: {config['original_embed_params']:,} params")
    print(f"  Original head was:  {config['original_head_params']:,} params")


if __name__ == "__main__":
    main()
