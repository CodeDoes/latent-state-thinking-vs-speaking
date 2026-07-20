#!/usr/bin/env python3
"""Byte interface for RWKV-7: swap the 65K embed/head for 258-byte versions.

The simplest possible approach: the original RWKV-7 model is frozen entirely.
We replace only the input embedding (65529→258) and output head (65529→258),
initialized from the first 256/258 rows of the originals. The model sees
identical vectors at both ends — it doesn't know anything changed.

Architecture:
    bytes → embed[258×dim] → [frozen RWKV-7 layers 0..L-1] → head[dim×258] → byte_logits

Usage:
    # Pre-train world model
    PYTHONPATH=. python src/byte_interface.py --mode pretrain --steps 500

    # Train byte interface only
    PYTHONPATH=. python src/byte_interface.py --mode train --steps 500

    # Full run
    PYTHONPATH=. python src/byte_interface.py --mode full --pretrain_steps 300 --train_steps 500
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from domains.rwkv.hf_rwkv_tokenizer import RWKV_TOKENIZER
from domains.rwkv.rwkv_nano import RWKV7Nano, count_params

WORLD_VOCAB_PATH = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
world_tokenizer = RWKV_TOKENIZER(str(WORLD_VOCAB_PATH))
WORLD_VOCAB_SIZE = len(world_tokenizer.token2idx)  # 65529
WORLD_PAD_ID = 0

BYTE_VOCAB_SIZE = 258
BYTE_PAD = 0
BYTE_UNK = 1
# byte 0x00 → id 2, byte 0x01 → id 3, ..., byte 0xFF → id 257
BYTE_TO_ID = {b: 2 + b for b in range(256)}
ID_TO_BYTE = {v: k for k, v in BYTE_TO_ID.items()}


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class ByteRWKV(nn.Module):
    """Frozen RWKV-7 with byte-scaled embed and head.

    The original 65K embed and head are replaced with 258-token versions,
    initialized from the first rows. All RWKV blocks stay frozen.

    Total trainable params: 2 × 258 × dim = 33K (for dim=64).
    """

    def __init__(self, core: RWKV7Nano):
        super().__init__()
        self.dim = core.dim
        self.head_size = core.head_size
        self.n_head = core.n_head

        # ── Byte embed (trainable) ──
        self.byte_embed = nn.Embedding(
            BYTE_VOCAB_SIZE, self.dim, padding_idx=BYTE_PAD
        )

        # ── Frozen core layers ──
        self.blocks = core.blocks
        for p in self.blocks.parameters():
            p.requires_grad = False

        # ── Frozen ln_out ──
        self.ln_out = core.ln_out
        for p in self.ln_out.parameters():
            p.requires_grad = False

        # ── Byte head (trainable) ──
        self.byte_head = nn.Linear(self.dim, BYTE_VOCAB_SIZE, bias=True)

        # ── Init from original weights ──
        self._init_from_core(core)

    def _init_from_core(self, core):
        """Initialize byte embed/head from first rows of original."""
        with torch.no_grad():
            # Byte embed: copy first 256 rows (tokens 0-255 = single bytes)
            self.byte_embed.weight.zero_()
            for bv in range(256):
                byte_id = BYTE_TO_ID[bv]
                self.byte_embed.weight[byte_id].copy_(
                    core.embed.weight[bv]
                )
            # PAD and UNK: copy from token 0
            self.byte_embed.weight[BYTE_PAD].copy_(core.embed.weight[0])
            self.byte_embed.weight[BYTE_UNK].copy_(core.embed.weight[0])

            # Byte head: copy first 258 rows
            self.byte_head.weight.zero_()
            self.byte_head.bias.zero_()
            for bv in range(256):
                byte_id = BYTE_TO_ID[bv]
                self.byte_head.weight[byte_id].copy_(
                    core.head.weight[bv]
                )
                self.byte_head.bias[byte_id].copy_(
                    core.head.bias[bv]
                )
            self.byte_head.weight[BYTE_PAD].copy_(core.head.weight[0])
            self.byte_head.bias[BYTE_PAD].copy_(core.head.bias[0])
            self.byte_head.weight[BYTE_UNK].copy_(core.head.weight[0])
            self.byte_head.bias[BYTE_UNK].copy_(core.head.bias[0])

        emb_trainable = sum(p.numel() for p in self.byte_embed.parameters())
        head_trainable = sum(p.numel() for p in self.byte_head.parameters())
        print(f"  Byte embed: {emb_trainable:,} params "
              f"(was {core.embed.weight.numel():,})")
        print(f"  Byte head:  {head_trainable:,} params "
              f"(was {core.head.weight.numel():,})")
        print(f"  Total trainable: {emb_trainable + head_trainable:,} "
              f"({100*(emb_trainable+head_trainable)/sum(p.numel() for p in core.parameters()):.1f}% of original)")

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: bytes → byte_embed → frozen blocks → byte_head.

        Args:
            byte_ids: (B, T) byte token IDs (0=PAD, 1=UNK, 2..257=0x00..0xFF)
        Returns:
            logits: (B, T, 258) byte logits
        """
        x = self.byte_embed(byte_ids)

        for block in self.blocks:
            x, _, _ = block(x)

        x = self.ln_out(x)
        return self.byte_head(x)


# ── Data ─────────────────────────────────────────────────────────────────

class SyntheticDataset:
    def __init__(self, seed: int = 42):
        self.rng = __import__('random').Random(seed)
        from threads.memory_growth.rule_generator import RULES
        self.rule = RULES["sum_threshold"]

    def get_world_batch(self, batch_size: int, max_len: int = 64):
        input_ids_list, target_list, mask_list = [], [], []
        for _ in range(batch_size):
            self.rule.rng = self.rng
            ex = self.rule.generate()
            text = ex["text"]
            tokens = world_tokenizer.encodeBytes(text.encode("utf-8"))
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [WORLD_PAD_ID] * (max_len - len(tokens))
            ids = torch.tensor(tokens, dtype=torch.long)
            tgt = torch.roll(ids, shifts=-1)
            tgt[-1] = WORLD_PAD_ID
            msk = torch.zeros(max_len, dtype=torch.float)
            msk[min(len(text) - 1, max_len - 2)] = 1.0
            input_ids_list.append(ids)
            target_list.append(tgt)
            mask_list.append(msk)
        return torch.stack(input_ids_list), torch.stack(target_list), torch.stack(mask_list)

    def get_byte_batch(self, batch_size: int, max_len: int = 128):
        input_ids_list, target_list, mask_list = [], [], []
        for _ in range(batch_size):
            self.rule.rng = self.rng
            ex = self.rule.generate()
            text = ex["text"]
            raw = text.encode("utf-8")
            tokens = [BYTE_TO_ID[b] for b in raw]
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [BYTE_PAD] * (max_len - len(tokens))
            ids = torch.tensor(tokens, dtype=torch.long)
            tgt = torch.roll(ids, shifts=-1)
            tgt[-1] = BYTE_PAD
            msk = torch.zeros(max_len, dtype=torch.float)
            msk[min(len(raw) - 1, max_len - 2)] = 1.0
            input_ids_list.append(ids)
            target_list.append(tgt)
            mask_list.append(msk)
        return torch.stack(input_ids_list), torch.stack(target_list), torch.stack(mask_list)


# ── Training ─────────────────────────────────────────────────────────────

def pretrain_world(args, exp_dir):
    """Train a world-tokenizer RWKV-7 from scratch."""
    print("=" * 60)
    print("Pre-training RWKV-7 on world tokenizer")
    print("=" * 60)

    core = RWKV7Nano(
        vocab_size=WORLD_VOCAB_SIZE, dim=args.dim,
        head_size=args.head_size, num_layers=args.layers,
    )
    dataset = SyntheticDataset(args.seed)
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr)
    print(f"  Params: {count_params(core):,}")

    t_start = time.time()
    for step in range(1, args.steps + 1):
        ids, targets, mask = dataset.get_world_batch(args.batch_size, 64)
        logits, _ = core(ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none",
        )
        loss = (loss.view_as(mask) * mask).sum() / (mask.sum() + 1e-8)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  step {step:4d}/{args.steps}  loss={loss.item():.4f}  "
                  f"{step/max(elapsed,1e-6):.1f} st/s")

    torch.save(core.state_dict(), exp_dir / "pretrained_world.pt")
    print(f"  saved: {exp_dir / 'pretrained_world.pt'}")
    return core


def train_byte_interface(args, exp_dir, core):
    """Train only the byte embed and head. Everything else frozen."""
    print("=" * 60)
    print("Training byte interface (embed + head only)")
    print("=" * 60)

    model = ByteRWKV(core)
    dataset = SyntheticDataset(args.seed)

    # Only byte_embed and byte_head are trainable
    optimizer = torch.optim.AdamW([
        {'params': model.byte_embed.parameters(), 'lr': args.lr},
        {'params': model.byte_head.parameters(), 'lr': args.lr},
    ])

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total:,}  Trainable: {trainable:,}  Frozen: {total - trainable:,}")

    t_start = time.time()
    for step in range(1, args.steps + 1):
        ids, targets, mask = dataset.get_byte_batch(args.batch_size, 128)
        logits = model(ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none",
        )
        loss = (loss.view_as(mask) * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
        optimizer.step()

        if step == 1 or step % 50 == 0:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            print(f"  step {step:4d}/{args.steps}  loss={loss.item():.4f}  {sps:.1f} st/s")

    torch.save(model.state_dict(), exp_dir / "byte_interface.pt")
    print(f"  saved: {exp_dir / 'byte_interface.pt'}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Byte interface for RWKV-7")
    ap.add_argument("--exp_id", default="byte_iface_001")
    ap.add_argument("--mode", default="full",
                    choices=["pretrain", "train", "full"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--pretrain_steps", type=int, default=None)
    ap.add_argument("--train_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--head_size", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    p_steps = args.pretrain_steps or args.steps
    t_steps = args.train_steps or args.steps

    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["git_hash"] = get_git_hash()
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Config: {exp_dir / 'config.json'}")

    core = None
    if args.mode in ("pretrain", "full"):
        core = pretrain_world(args, exp_dir)

    if args.mode in ("train", "full"):
        if core is None:
            core_path = exp_dir / "pretrained_world.pt"
            if not core_path.exists():
                print("ERROR: No pretrained model found")
                sys.exit(1)
            core = RWKV7Nano(
                vocab_size=WORLD_VOCAB_SIZE, dim=args.dim,
                head_size=args.head_size, num_layers=args.layers,
            )
            core.load_state_dict(torch.load(core_path, map_location="cpu"))
            print(f"  Loaded pretrained model from {core_path}")

        args.steps = t_steps
        train_byte_interface(args, exp_dir, core)


if __name__ == "__main__":
    main()
