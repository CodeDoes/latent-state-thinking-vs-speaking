#!/usr/bin/env python3
"""Layer-aware tokenizer surgery for RWKV nano.

Thesis: RWKV's first K layers act as a token → state encoder, the middle
layers as a "thinking" core, and the last K layers as a state → token
decoder. Surgery replaces the encoder+decoder with byte-level versions
while keeping the frozen core intact.

Phases:
  1. Pre-train a token-level RWKV on a synthetic rule task using the
     RWKV world tokenizer (65529 vocab).
  2. Surgery: split model into encoder/core/decoder layers.
     Replace encoder (embed + first N layers) with a byte-level encoder.
     Replace decoder (last N layers + head) with a byte-level decoder.
     Core layers stay frozen.
  3. Train only the byte encoder + byte decoder.
  4. (Control) Train same architecture from scratch on bytes.

Usage:
    # Phase 1: pre-train on rwkv world tokenizer
    python src/token_surgery.py --exp_id rwkv7_surgery_001 --phase 1 --steps 500

    # Phase 2+3: surgery + train byte interface
    python src/token_surgery.py --exp_id rwkv7_surgery_001 --phase 23 --steps 500

    # Phase 4: from-scratch control
    python src/token_surgery.py --exp_id rwkv7_surgery_001 --phase 4 --steps 500

    # All phases
    python src/token_surgery.py --exp_id rwkv7_surgery_001 --phase 1234 --steps 300
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── RWKV world tokenizer ──
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER
WORLD_VOCAB_PATH = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
world_tokenizer = RWKV_TOKENIZER(str(WORLD_VOCAB_PATH))
WORLD_VOCAB_SIZE = len(world_tokenizer.token2idx)   # 65529
WORLD_PAD_ID = 0  # byte 0x00

# ── Byte-level vocab (258 tokens: PAD=0, UNK=1, bytes 2..257 = 0x00..0xFF) ──
BYTE_VOCAB_SIZE = 258
BYTE_PAD = 0
# Build byte→id lookup (same as byte_vocab.py)
BYTE_TO_ID = {b: 2 + b for b in range(256)}  # byte 0x00 → id 2, byte 0xFF → id 257

from src.rwkv_nano import RWKVNano, RWKVBlock, count_params


# ── Helpers ───────────────────────────────────────────────────────────────

def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def text_to_world_ids(text: str, max_len: int = 128) -> list[int]:
    """Encode text using the RWKV world tokenizer."""
    tokens = world_tokenizer.encodeBytes(text.encode("utf-8"))
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [WORLD_PAD_ID] * (max_len - len(tokens))
    return tokens


def text_to_byte_ids(text: str, max_len: int = 128) -> list[int]:
    """Encode text as raw bytes using the 258-byte vocabulary."""
    raw = text.encode("utf-8")
    tokens = [BYTE_TO_ID[b] for b in raw]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [BYTE_PAD] * (max_len - len(tokens))
    return tokens


def get_label_mask(text_len: int, total_len: int) -> list[float]:
    """Create a loss mask: 1.0 at the label position."""
    mask = [0.0] * total_len
    label_pos = min(text_len - 1, total_len - 2)
    mask[label_pos] = 1.0
    return mask


# ── Surgery model: byte-level encoder + frozen core + byte-level decoder ──

class ByteLevelRWKV(nn.Module):
    """RWKV with surgically replaced byte-level encoder/decoder.

    Architecture:
        byte_embed → byte_encoder (1 trainable RWKVBlock)
        → [frozen core layers]
        → byte_decoder (1 trainable RWKVBlock) → byte_head
    """

    def __init__(
        self,
        core_model: RWKVNano,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
    ):
        super().__init__()
        self.dim = core_model.dim
        self.n_core_layers = core_model.num_layers - n_encoder_layers - n_decoder_layers
        assert self.n_core_layers >= 0, (
            f"Need at least {n_encoder_layers + n_decoder_layers} layers, "
            f"model has {core_model.num_layers}"
        )

        # ── Byte embedder ──
        self.byte_embed = nn.Embedding(BYTE_VOCAB_SIZE, self.dim, padding_idx=BYTE_PAD)

        # ── Byte encoder (trainable) ──
        self.byte_encoder = nn.ModuleList([
            RWKVBlock(self.dim) for _ in range(n_encoder_layers)
        ])

        # ── Frozen core (copied from pre-trained model) ──
        start = n_encoder_layers
        self.core = nn.ModuleList()
        for i in range(start, start + self.n_core_layers):
            layer = RWKVBlock(self.dim)
            layer.load_state_dict(core_model.blocks[i].state_dict())
            for p in layer.parameters():
                p.requires_grad = False
            self.core.append(layer)

        # ── Byte decoder (trainable) ──
        self.byte_decoder = nn.ModuleList([
            RWKVBlock(self.dim) for _ in range(n_decoder_layers)
        ])

        # ── Output ──
        self.ln_out = nn.LayerNorm(self.dim)
        # Initialize ln_out from the core model's ln_out
        self.ln_out.weight.data.copy_(core_model.ln_out.weight.data)
        self.ln_out.bias.data.copy_(core_model.ln_out.bias.data)

        self.byte_head = nn.Linear(self.dim, BYTE_VOCAB_SIZE, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[dict]]]:
        B, T = input_ids.shape
        x = self.byte_embed(input_ids)  # (B, T, dim)

        # Byte encoder
        for block in self.byte_encoder:
            x, _ = block(x)

        # Frozen core
        for block in self.core:
            x, _ = block(x)

        # Byte decoder
        for block in self.byte_decoder:
            x, _ = block(x)

        x = self.ln_out(x)
        logits = self.byte_head(x)
        return logits, None


def perform_surgery(pretrained_model: RWKVNano, n_encoder: int = 1, n_decoder: int = 1) -> ByteLevelRWKV:
    """Perform layer-aware surgery on a pre-trained RWKV.

    Copies core layer weights, initializes byte embed from the first 258
    rows of the old token embedding, initializes byte head from first 258
    rows of the old head. New encoder/decoder blocks get the old
    encoder/decoder layer weights as initialization.
    """
    model = ByteLevelRWKV(pretrained_model, n_encoder, n_decoder)

    with torch.no_grad():
        # Initialize byte embed from first 258 rows of old embed
        model.byte_embed.weight.zero_()
        # World tokenizer: first 256 tokens are bytes 0x00-0xFF at positions 0-255
        # Our byte vocab: byte 0x00 → id 2, byte 0xFF → id 257; id 0=PAD, id 1=UNK
        # Copy: world_token_id(b) → byte_token_id(b) for bytes 0x00-0xFF
        for byte_val in range(256):
            world_id = byte_val  # world tokenizer tokens 0-255 are bytes
            byte_id = BYTE_TO_ID[byte_val]  # byte_vocab: 2..257
            model.byte_embed.weight[byte_id].copy_(
                pretrained_model.embed.weight[world_id]
            )

        # Initialize byte head from first 258 rows of old head
        model.byte_head.weight.zero_()
        model.byte_head.bias.zero_()
        for byte_val in range(256):
            world_id = byte_val
            byte_id = BYTE_TO_ID[byte_val]
            model.byte_head.weight[byte_id].copy_(
                pretrained_model.head.weight[world_id]
            )
            model.byte_head.bias[byte_id].copy_(
                pretrained_model.head.bias[world_id]
            )

        # Initialize new encoder/decoder blocks from old layer weights
        if n_encoder > 0:
            for new_block, old_block in zip(model.byte_encoder, pretrained_model.blocks[:n_encoder]):
                new_block.load_state_dict(old_block.state_dict())

        if n_decoder > 0:
            for new_block, old_block in zip(
                model.byte_decoder,
                pretrained_model.blocks[-n_decoder:]
            ):
                new_block.load_state_dict(old_block.state_dict())

    return model


# ── Data generation ────────────────────────────────────────────────────────

def char_example_to_tensor(rule, rng_seed: int, max_len: int = 128):
    """One example → world-tokenized (BPE) input."""
    import random
    rng = random.Random(rng_seed)
    rule.rng = rng
    ex = rule.generate()
    text = ex["text"]
    toks = text_to_world_ids(text, max_len)
    input_ids = torch.tensor(toks, dtype=torch.long)
    targets = torch.roll(input_ids, shifts=-1)
    targets[-1] = WORLD_PAD_ID
    mask = torch.tensor(get_label_mask(len(text), max_len), dtype=torch.float)
    return input_ids, targets, mask


def byte_example_to_tensor(rule, rng_seed: int, max_len: int = 256):
    """One example → byte-tokenized input."""
    import random
    rng = random.Random(rng_seed)
    rule.rng = rng
    ex = rule.generate()
    text = ex["text"]
    toks = text_to_byte_ids(text, max_len)
    input_ids = torch.tensor(toks, dtype=torch.long)
    targets = torch.roll(input_ids, shifts=-1)
    targets[-1] = BYTE_PAD
    mask = torch.tensor(get_label_mask(len(text), max_len), dtype=torch.float)
    return input_ids, targets, mask


# ── Phase functions ────────────────────────────────────────────────────────

def phase1_pretrain(args, exp_dir):
    """Phase 1: Pre-train RWKV on world-tokenized data."""
    print("=" * 60)
    print(f"Phase 1: Pre-training on world tokenizer (vocab={WORLD_VOCAB_SIZE})")
    print(f"         rule={args.rule}, dim={args.dim}, layers={args.layers}")
    print("=" * 60)

    from src.rule_generator import RULES as CHAR_RULES
    rule = CHAR_RULES[args.rule]

    model = RWKVNano(
        vocab_size=WORLD_VOCAB_SIZE,
        dim=args.dim,
        num_layers=args.layers,
        pad_token_id=WORLD_PAD_ID,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model params: {count_params(model):,}")

    t_start = time.time()
    for step in range(1, args.steps + 1):
        input_ids_list, target_list, mask_list = [], [], []
        for i in range(args.batch_size):
            seed = step * args.batch_size + i
            inp, tgt, msk = char_example_to_tensor(rule, seed, args.max_len)
            input_ids_list.append(inp)
            target_list.append(tgt)
            mask_list.append(msk)

        input_ids = torch.stack(input_ids_list)
        targets = torch.stack(target_list)
        mask = torch.stack(mask_list)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view_as(mask)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            print(f"  step {step:4d}/{args.steps}  loss={loss.item():.4f}  {sps:.1f} st/s")

    ckpt_path = exp_dir / "pretrained_world.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": WORLD_VOCAB_SIZE,
            "dim": args.dim,
            "num_layers": args.layers,
            "vocab_kind": "rwkv_world",
            "rule": args.rule,
        },
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return ckpt_path


def phase23_surgery(args, exp_dir, pretrained_path: Path):
    """Phase 2+3: Layer-aware surgery + byte-level fine-tune."""
    print("=" * 60)
    print(f"Phase 2-3: Surgery + byte-level fine-tune (rule={args.rule})")
    print("=" * 60)

    ckpt = torch.load(pretrained_path, map_location="cpu")
    cfg = ckpt["config"]
    dim = cfg["dim"]
    layers = cfg["num_layers"]
    n_enc = getattr(args, 'n_encoder_layers', max(1, layers // 3))
    n_dec = getattr(args, 'n_decoder_layers', max(1, layers // 3))
    n_core = layers - n_enc - n_dec

    print(f"  Split: {n_enc} encoder + {n_core} core + {n_dec} decoder")

    old_model = RWKVNano(vocab_size=cfg["vocab_size"], dim=dim, num_layers=layers)
    old_model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded pre-trained model (vocab={cfg['vocab_size']}, {layers} layers)")

    # Perform layer-aware surgery
    model = perform_surgery(old_model, n_enc, n_dec)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params "
          f"(encoder+decoder={trainable:,}, core frozen={total - trainable:,})")

    # Verify initialization
    with torch.no_grad():
        test_ids = torch.tensor([[BYTE_TO_ID[ord('0')]]])
        logits, _ = model(test_ids)
        top5 = logits[0, 0].topk(5)
        print(f"  Init check: top-5 logits for byte '0': "
              f"{[chr(b) if (b:=bid-2) in range(256) else '?' for bid in top5.indices]} "
              f"(values={[round(v.item(),3) for v in top5.values]})")

    from src.rule_generator import RULES as CHAR_RULES
    rule = CHAR_RULES[args.rule]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    t_start = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        input_ids_list, target_list, mask_list = [], [], []
        for i in range(args.batch_size):
            seed = step * args.batch_size + i + 100000
            inp, tgt, msk = byte_example_to_tensor(rule, seed, args.max_len)
            input_ids_list.append(inp)
            target_list.append(tgt)
            mask_list.append(msk)

        input_ids = torch.stack(input_ids_list)
        targets = torch.stack(target_list)
        mask = torch.stack(mask_list)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view_as(mask)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            recent = sum(losses[-min(len(losses), args.log_every):]) / min(len(losses), args.log_every)
            print(f"  step {step:4d}/{args.steps}  loss={recent:.4f}  {sps:.1f} st/s")

    ckpt_path = exp_dir / "surgery_finetuned.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_kind": "byte_surgery",
            "n_encoder": n_enc,
            "n_core": n_core,
            "n_decoder": n_dec,
            "dim": dim,
            "rule": args.rule,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
        },
        "loss_trace": losses,
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return losses


def phase4_from_scratch(args, exp_dir):
    """Phase 4 (control): Train ByteLevelRWKV from scratch on bytes."""
    print("=" * 60)
    print(f"Phase 4 (control): From-scratch byte-level (rule={args.rule})")
    print("=" * 60)

    n_enc = getattr(args, 'n_encoder_layers', 1)
    n_dec = getattr(args, 'n_decoder_layers', 1)

    from src.rule_generator import RULES as CHAR_RULES
    rule = CHAR_RULES[args.rule]

    # Create a scratch "core" to get the architecture
    dummy_core = RWKVNano(
        vocab_size=BYTE_VOCAB_SIZE,
        dim=args.dim,
        num_layers=n_enc + n_dec + 1,  # minimal core of 1
        pad_token_id=BYTE_PAD,
    )
    model = ByteLevelRWKV(dummy_core, n_enc, n_dec)

    # Unfreeze everything for from-scratch training
    for p in model.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total:,} (all trainable)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t_start = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        input_ids_list, target_list, mask_list = [], [], []
        for i in range(args.batch_size):
            seed = step * args.batch_size + i + 200000
            inp, tgt, msk = byte_example_to_tensor(rule, seed, args.max_len)
            input_ids_list.append(inp)
            target_list.append(tgt)
            mask_list.append(msk)

        input_ids = torch.stack(input_ids_list)
        targets = torch.stack(target_list)
        mask = torch.stack(mask_list)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view_as(mask)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            recent = sum(losses[-min(len(losses), args.log_every):]) / min(len(losses), args.log_every)
            print(f"  step {step:4d}/{args.steps}  loss={recent:.4f}  {sps:.1f} st/s")

    ckpt_path = exp_dir / "from_scratch_byte.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_kind": "byte_scratch",
            "dim": args.dim,
            "rule": args.rule,
        },
        "loss_trace": losses,
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return losses


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RWKV world tokenizer surgery")
    ap.add_argument("--exp_id", default="rwkv7_surgery_001")
    ap.add_argument("--phase", default="1234")
    ap.add_argument("--rule", default="sum_threshold",
                    choices=["sum_threshold", "vowel_majority", "endpoint_match",
                             "count_trigger", "parity", "modulo3"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--pretrain_steps", type=int, default=None)
    ap.add_argument("--finetune_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128,
                    help="Max token length for world tokenizer (phase 1)")
    ap.add_argument("--byte_max_len", type=int, default=256,
                    help="Max byte length for byte phases (23, 4)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3,
                    help="Total layers: split as encoder+core+decoder")
    ap.add_argument("--n_encoder_layers", type=int, default=None,
                    help="Layers to replace with byte encoder (default: max(1, layers//3))")
    ap.add_argument("--n_decoder_layers", type=int, default=None,
                    help="Layers to replace with byte decoder (default: max(1, layers//3))")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    p1_steps = args.pretrain_steps or args.steps
    p23_steps = args.finetune_steps or args.steps
    p4_steps = args.finetune_steps or args.steps

    # Default split: 1 encoder, rest-1 decoder, or thirds
    if args.n_encoder_layers is None:
        args.n_encoder_layers = max(1, args.layers // 3)
    if args.n_decoder_layers is None:
        args.n_decoder_layers = max(1, args.layers // 3)

    torch.manual_seed(args.seed)
    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["git_hash"] = get_git_hash()
    config["p1_steps"] = p1_steps
    config["p23_steps"] = p23_steps
    config["p4_steps"] = p4_steps
    config["world_vocab_size"] = WORLD_VOCAB_SIZE
    config["byte_vocab_size"] = BYTE_VOCAB_SIZE
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Config: {exp_dir / 'config.json'}")
    print(f"  git hash: {config['git_hash']}")
    print(f"  rule: {args.rule}")
    print(f"  layers: {args.layers} (enc={args.n_encoder_layers}, "
          f"core={args.layers - args.n_encoder_layers - args.n_decoder_layers}, "
          f"dec={args.n_decoder_layers})")
    print(f"  world vocab: {WORLD_VOCAB_SIZE}, byte vocab: {BYTE_VOCAB_SIZE}")
    print()

    results = {}

    if "1" in args.phase:
        pretrained_path = exp_dir / "pretrained_world.pt"
        if not pretrained_path.exists():
            pretrained_path = phase1_pretrain(
                argparse.Namespace(**{**vars(args), "steps": p1_steps}),
                exp_dir,
            )
        else:
            print(f"Phase 1 already done: {pretrained_path}")
        results["pretrain_path"] = str(pretrained_path)

    if "2" in args.phase or "3" in args.phase:
        pretrained_path = exp_dir / "pretrained_world.pt"
        if not pretrained_path.exists():
            print("ERROR: Phase 1 must complete before Phase 2-3.")
            sys.exit(1)
        surgery_losses = phase23_surgery(
            argparse.Namespace(**{**vars(args), "steps": p23_steps}),
            exp_dir,
            pretrained_path,
        )
        results["surgery_final_loss"] = surgery_losses[-1] if surgery_losses else None
        results["surgery_loss_trace"] = surgery_losses

    if "4" in args.phase:
        scratch_losses = phase4_from_scratch(
            argparse.Namespace(**{**vars(args), "steps": p4_steps}),
            exp_dir,
        )
        results["scratch_final_loss"] = scratch_losses[-1] if scratch_losses else None
        results["scratch_loss_trace"] = scratch_losses

    # Summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    sf = results.get("surgery_final_loss")
    sc = results.get("scratch_final_loss")
    if sf is not None:
        print(f"  Surgery (world→byte): final loss = {sf:.4f}")
    if sc is not None:
        print(f"  Scratch (byte scratch): final loss = {sc:.4f}")
    if sf is not None and sc is not None:
        diff = sc - sf
        print(f"  Δ (scratch − surgery): {diff:+.4f}")
        if diff > 0:
            print(f"  → Surgery transfers knowledge (lower loss)")
        elif diff < 0:
            print(f"  → Surgery HURTS (from-scratch is better)")
        else:
            print(f"  → No measurable transfer")

    # Save metrics
    for k, v in results.items():
        if isinstance(v, list):
            results[k] = [round(x, 4) for x in v[-10:]]
    (exp_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {exp_dir}/")


if __name__ == "__main__":
    main()
