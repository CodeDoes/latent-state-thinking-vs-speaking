#!/usr/bin/env python3
"""Layer-aware tokenizer surgery for RWKV nano — with state distillation.

Thesis: RWKV's first K layers act as a token → state encoder. We replace
them with a byte-level encoder, but first distil the world tokenizer's
state vectors into the byte encoder so the frozen core "understands" it.

Phases:
  0: State distillation — train byte encoder to match world-tokenizer
     states at the encoder→core boundary. Uses MSE on pooled byte states
     aligned via tokenizer byte spans.
  1: Pre-train a token-level RWKV on a synthetic rule task using the
     RWKV world tokenizer (65529 vocab).
  2: Surgery — split model into encoder/core/decoder. Replace encoder
     + decoder with byte versions. Initialize via phase 0 states.
  3: Train byte encoder + decoder on the task.
  4: (Control) Train same architecture from scratch on bytes.

Alternating training (optional):
  After phase 3, you can loop: train byte encoder (phase 0) → train
  core+decoder (phase 3) → repeat. Each loop tightens the alignment.

Usage:
    # Phase 1: pre-train on rwkv world tokenizer
    python src/token_surgery.py --exp_id rwkv7_surgery_002 --phase 1 --steps 500

    # Phase 0: state distillation (align byte encoder→world states)
    python src/token_surgery.py --exp_id rwkv7_surgery_002 --phase 0 --steps 200

    # Phase 2+3: surgery + task fine-tune
    python src/token_surgery.py --exp_id rwkv7_surgery_002 --phase 23 --steps 500

    # Phase 4: from-scratch control
    python src/token_surgery.py --exp_id rwkv7_surgery_002 --phase 4 --steps 500
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── RWKV world tokenizer ──
from src.hf_rwkv_tokenizer import RWKV_TOKENIZER
WORLD_VOCAB_PATH = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
world_tokenizer = RWKV_TOKENIZER(str(WORLD_VOCAB_PATH))
WORLD_VOCAB_SIZE = len(world_tokenizer.token2idx)   # 65529
WORLD_PAD_ID = 0

# ── Byte-level vocab ──
BYTE_VOCAB_SIZE = 258
BYTE_PAD = 0
BYTE_TO_ID = {b: 2 + b for b in range(256)}  # byte 0x00 → id 2, ... 0xFF → id 257

from src.rwkv_nano import RWKVNano, RWKVBlock, count_params


# ── Helpers ───────────────────────────────────────────────────────────────

def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def encode_with_spans(text: str, max_tokens: int = 64) -> tuple[list[int], list[tuple[int, int]]]:
    """Encode text with the world tokenizer, returning (token_ids, byte_spans).

    byte_spans[i] = (start_byte, end_byte) for token i.
    The TRIE is greedy longest-match, so spans are contiguous and cover the
    full byte sequence.
    """
    raw = text.encode("utf-8")
    tokens = []
    spans = []
    idx = 0
    while idx < len(raw):
        old_idx = idx
        idx, node, values = world_tokenizer.root.find_longest(raw, idx)
        assert idx != old_idx, f"Stuck at byte {old_idx} in {raw!r}"
        _, token_id = next(iter(values))
        tokens.append(token_id)
        spans.append((old_idx, idx))
        if len(tokens) >= max_tokens:
            break
    return tokens, spans


def text_to_world_ids(text: str, max_len: int = 128) -> list[int]:
    tokens = world_tokenizer.encodeBytes(text.encode("utf-8"))
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [WORLD_PAD_ID] * (max_len - len(tokens))
    return tokens


def text_to_byte_ids(text: str, max_len: int = 256) -> list[int]:
    raw = text.encode("utf-8")
    tokens = [BYTE_TO_ID[b] for b in raw]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [BYTE_PAD] * (max_len - len(tokens))
    return tokens


def get_label_mask(text_len: int, total_len: int) -> list[float]:
    mask = [0.0] * total_len
    label_pos = min(text_len - 1, total_len - 2)
    mask[label_pos] = 1.0
    return mask


# ── Forward hooks for state extraction ────────────────────────────────────

@torch.no_grad()
def forward_get_hidden(model: RWKVNano, input_ids: torch.Tensor,
                       layer_idx: int) -> torch.Tensor:
    """Run model forward, return hidden state AFTER layer `layer_idx`.

    The returned tensor is shape (B, T, dim) — the output of blocks[layer_idx].
    """
    B, T = input_ids.shape
    x = model.embed(input_ids)
    for i in range(min(layer_idx + 1, len(model.blocks))):
        x, _ = model.blocks[i](x)
    return x  # still pre-ln_out


# ── ByteLevelRWKV — encoder + frozen core + decoder ───────────────────────

class ByteLevelRWKV(nn.Module):
    """RWKV with surgically replaced byte-level encoder/decoder.

    Architecture:
        byte_embed → byte_encoder (N trainable RWKVBlocks)
        → [N_core frozen RWKVBlocks]
        → byte_decoder (M trainable RWKVBlocks) → byte_head
    """

    def __init__(self, core_model: RWKVNano,
                 n_encoder_layers: int = 1, n_decoder_layers: int = 1):
        super().__init__()
        self.dim = core_model.dim
        self.n_core = core_model.num_layers - n_encoder_layers - n_decoder_layers
        assert self.n_core >= 0

        self.byte_embed = nn.Embedding(BYTE_VOCAB_SIZE, self.dim, padding_idx=BYTE_PAD)
        self.byte_encoder = nn.ModuleList([
            RWKVBlock(self.dim) for _ in range(n_encoder_layers)
        ])

        start = n_encoder_layers
        self.core = nn.ModuleList()
        for i in range(start, start + self.n_core):
            layer = RWKVBlock(self.dim)
            layer.load_state_dict(core_model.blocks[i].state_dict())
            for p in layer.parameters():
                p.requires_grad = False
            self.core.append(layer)

        self.byte_decoder = nn.ModuleList([
            RWKVBlock(self.dim) for _ in range(n_decoder_layers)
        ])

        self.ln_out = nn.LayerNorm(self.dim)
        self.ln_out.weight.data.copy_(core_model.ln_out.weight.data)
        self.ln_out.bias.data.copy_(core_model.ln_out.bias.data)
        self.byte_head = nn.Linear(self.dim, BYTE_VOCAB_SIZE, bias=True)

    def forward_encoder(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Byte → encoder states. Returns (B, T_byte, dim)."""
        x = self.byte_embed(input_ids)
        for block in self.byte_encoder:
            x, _ = block(x)
        return x

    def forward_core(self, x: torch.Tensor) -> torch.Tensor:
        """Core states → core outputs."""
        for block in self.core:
            x, _ = block(x)
        return x

    def forward(self, input_ids: torch.Tensor, **kw) -> tuple[torch.Tensor, None]:
        B, T = input_ids.shape
        x = self.byte_embed(input_ids)
        for block in self.byte_encoder:
            x, _ = block(x)
        for block in self.core:
            x, _ = block(x)
        for block in self.byte_decoder:
            x, _ = block(x)
        x = self.ln_out(x)
        logits = self.byte_head(x)
        return logits, None


def create_surgery_model(pretrained_model: RWKVNano,
                         n_encoder: int = 1, n_decoder: int = 1,
                         init_from_world: bool = True) -> ByteLevelRWKV:
    """Create surgery model, init byte embed/head + encoder/decoder from world."""
    model = ByteLevelRWKV(pretrained_model, n_encoder, n_decoder)

    if init_from_world:
        with torch.no_grad():
            # Init byte embed from first 256 rows of world embed
            model.byte_embed.weight.zero_()
            for bv in range(256):
                byte_id = BYTE_TO_ID[bv]
                model.byte_embed.weight[byte_id].copy_(
                    pretrained_model.embed.weight[bv]
                )

            # Init byte head from first 256 rows of world head
            model.byte_head.weight.zero_()
            model.byte_head.bias.zero_()
            for bv in range(256):
                byte_id = BYTE_TO_ID[bv]
                model.byte_head.weight[byte_id].copy_(
                    pretrained_model.head.weight[bv]
                )
                model.byte_head.bias[byte_id].copy_(
                    pretrained_model.head.bias[bv]
                )

            # Init encoder/decoder blocks from world layer weights
            if n_encoder > 0:
                for new_b, old_b in zip(model.byte_encoder,
                                        pretrained_model.blocks[:n_encoder]):
                    new_b.load_state_dict(old_b.state_dict())
            if n_decoder > 0:
                for new_b, old_b in zip(
                    model.byte_decoder,
                    pretrained_model.blocks[-n_decoder:]
                ):
                    new_b.load_state_dict(old_b.state_dict())

    return model


# ── State distillation: match byte-encoder states to world states ──────────

def pool_byte_states(byte_states: torch.Tensor, spans: list[list[tuple[int, int]]],
                     n_tokens: int) -> torch.Tensor:
    """Pool byte-level states into token-level states using byte spans.

    Args:
        byte_states: (B, T_bytes, dim)
        spans: list of B lists of (start, end) byte spans per token
        n_tokens: number of token positions (max across batch)
    Returns:
        (B, n_tokens, dim) — pooled states, 0-padded for short sequences
    """
    B, _, D = byte_states.shape
    device = byte_states.device
    pooled = byte_states.new_zeros(B, n_tokens, D)

    for bi in range(B):
        for ti, (s, e) in enumerate(spans[bi]):
            if ti >= n_tokens:
                break
            if e > byte_states.shape[1]:
                break
            states = byte_states[bi, s:e]  # (span_len, D)
            if states.shape[0] > 0:
                pooled[bi, ti] = states.mean(dim=0)

    return pooled


@torch.no_grad()
def generate_distillation_batch(
    rule,
    batch_size: int,
    world_model: RWKVNano,
    enc_layer_idx: int,
    max_tokens: int,
    max_bytes: int,
    seed_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
    """Generate a batch for state distillation.

    Returns:
        byte_ids: (B, max_bytes) — byte-level input
        world_states: (B, max_tokens, dim) — world-model states at enc_layer_idx
        spans: list of B lists of (s, e) spans
    """
    import random
    B = batch_size
    from src.rule_generator import RULES as CHAR_RULES
    rule = CHAR_RULES[rule] if isinstance(rule, str) else rule

    byte_ids_list = []
    world_states_list = []
    spans_list = []
    max_n_tokens = 0

    for i in range(batch_size):
        rng = random.Random(seed_offset * batch_size + i + 10000)
        rule.rng = rng
        ex = rule.generate()
        text = ex["text"]

        # World tokenize with spans
        tok_ids, spans = encode_with_spans(text, max_tokens)
        max_n_tokens = max(max_n_tokens, len(tok_ids))

        # World forward → get states
        world_ids = torch.tensor(
            tok_ids + [WORLD_PAD_ID] * (max_tokens - len(tok_ids)),
            dtype=torch.long,
        ).unsqueeze(0)

        w_states = forward_get_hidden(world_model, world_ids, enc_layer_idx)
        # w_states: (1, max_tokens, dim) — padded positions have garbage states
        # Zero out padded positions
        if len(tok_ids) < max_tokens:
            w_states[0, len(tok_ids):] = 0.0
        world_states_list.append(w_states)

        # Byte ids
        raw = text.encode("utf-8")
        byte_toks = [BYTE_TO_ID[b] for b in raw]
        if len(byte_toks) > max_bytes:
            byte_toks = byte_toks[:max_bytes]
        else:
            byte_toks = byte_toks + [BYTE_PAD] * (max_bytes - len(byte_toks))
        byte_ids_list.append(torch.tensor(byte_toks, dtype=torch.long))
        spans_list.append(spans)

    byte_ids = torch.stack(byte_ids_list)
    world_states = torch.cat(world_states_list, dim=0)
    return byte_ids, world_states, spans_list, max_n_tokens


def phase0_distillation(args, exp_dir, pretrained_path: Path):
    """Phase 0: State distillation — align byte encoder→world states.

    Training target: MSE between pooled byte-encoder states and world-model
    states (at the encoder→core boundary).
    """
    print("=" * 60)
    print("Phase 0: State distillation (byte encoder → world states)")
    print("=" * 60)

    ckpt = torch.load(pretrained_path, map_location="cpu")
    cfg = ckpt["config"]
    dim = cfg["dim"]
    layers = cfg["num_layers"]
    n_enc = getattr(args, 'n_encoder_layers', max(1, layers // 3))

    # Load world model
    world_model = RWKVNano(vocab_size=cfg["vocab_size"], dim=dim, num_layers=layers)
    world_model.load_state_dict(ckpt["model_state"])
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad = False

    # Create surgery model
    model = create_surgery_model(world_model, n_enc, 0)
    # Only train byte encoder
    for p in model.byte_embed.parameters():
        p.requires_grad = True
    for p in model.byte_encoder.parameters():
        p.requires_grad = True
    for p in model.core.parameters():
        p.requires_grad = False
    for p in model.byte_decoder.parameters():
        p.requires_grad = False
    for p in model.ln_out.parameters():
        p.requires_grad = False
    for p in model.byte_head.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} (only byte encoder)")

    from src.rule_generator import RULES as CHAR_RULES
    rule = CHAR_RULES[args.rule]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    enc_layer_idx = n_enc - 1  # world states after the last encoder layer

    losses = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        byte_ids, world_states, spans, max_nt = generate_distillation_batch(
            rule, args.batch_size, world_model, enc_layer_idx,
            args.distil_max_tokens, args.distil_max_bytes,
            seed_offset=step,
        )

        # Byte encoder forward
        byte_states = model.forward_encoder(byte_ids)  # (B, T_bytes, D)

        # Pool byte states per token span
        pooled = pool_byte_states(byte_states, spans, world_states.shape[1])
        # pooled: (B, max_tokens, D)

        # MSE loss on non-padded positions
        mask = (world_states.abs().sum(dim=-1) > 1e-8).float()  # (B, T)
        diff = (pooled - world_states).pow(2).sum(dim=-1)  # (B, T)
        loss = (diff * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
        optimizer.step()
        losses.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            print(f"  step {step:4d}/{args.steps}  mse={loss.item():.6f}  {sps:.1f} st/s")

    ckpt_path = exp_dir / "distilled_encoder.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_kind": "byte_encoder_distilled",
            "n_encoder": n_enc,
            "dim": dim,
            "rule": args.rule,
        },
        "loss_trace": losses,
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return ckpt_path


# ── Data generation (original phases) ──────────────────────────────────────

def char_example_to_tensor(rule, rng_seed: int, max_len: int = 128):
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


# ── Original phases ────────────────────────────────────────────────────────

def phase1_pretrain(args, exp_dir):
    """Phase 1: Pre-train RWKV on world-tokenized data."""
    print("=" * 60)
    print(f"Phase 1: Pre-training on world tokenizer (vocab={WORLD_VOCAB_SIZE})")
    print(f"         rule={args.rule}, dim={args.dim}, layers={args.layers}")
    print("=" * 60)

    from src.rule_generator import RULES as CHAR_RULES
    rule = CHAR_RULES[args.rule]

    model = RWKVNano(
        vocab_size=WORLD_VOCAB_SIZE, dim=args.dim,
        num_layers=args.layers, pad_token_id=WORLD_PAD_ID,
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
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none",
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
            "vocab_size": WORLD_VOCAB_SIZE, "dim": args.dim,
            "num_layers": args.layers, "vocab_kind": "rwkv_world",
            "rule": args.rule,
        },
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return ckpt_path


def phase23_surgery(args, exp_dir, pretrained_path: Path, distil_path: Optional[Path] = None):
    """Phase 2+3: Surgery + byte-level fine-tune."""
    print("=" * 60)
    print(f"Phase 2-3: Surgery + byte-level fine-tune (rule={args.rule})")
    print("=" * 60)

    ckpt = torch.load(pretrained_path, map_location="cpu")
    cfg = ckpt["config"]
    dim = cfg["dim"]
    layers = cfg["num_layers"]
    n_enc = getattr(args, 'n_encoder_layers', max(1, layers // 3))
    n_dec = getattr(args, 'n_decoder_layers', max(1, layers // 3))

    old_model = RWKVNano(vocab_size=cfg["vocab_size"], dim=dim, num_layers=layers)
    old_model.load_state_dict(ckpt["model_state"])

    # Create surgery model
    model = create_surgery_model(old_model, n_enc, n_dec, init_from_world=True)

    # Load distilled encoder if available
    if distil_path and distil_path.exists():
        distil_ckpt = torch.load(distil_path, map_location="cpu")
        # Copy just the encoder parts
        enc_state = {
            k.replace("byte_embed.", ""): v
            for k, v in distil_ckpt["model_state"].items()
            if k.startswith("byte_embed.")
        }
        enc_state.update({
            k.replace("byte_encoder.", ""): v
            for k, v in distil_ckpt["model_state"].items()
            if k.startswith("byte_encoder.")
        })
        model.byte_embed.load_state_dict(enc_state, strict=False)
        for i, block in enumerate(model.byte_encoder):
            block_state = {
                k: v for k, v in distil_ckpt["model_state"].items()
                if k.startswith(f"byte_encoder.{i}.")
            }
            if block_state:
                block.load_state_dict({
                    k.replace(f"byte_encoder.{i}.", ""): v
                    for k, v in block_state.items()
                })
        print(f"  Loaded distilled encoder from {distil_path}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params")

    from src.rule_generator import RULES as CHAR_RULES
    rule = CHAR_RULES[args.rule]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    losses = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        input_ids_list, target_list, mask_list = [], [], []
        for i in range(args.batch_size):
            seed = step * args.batch_size + i + 100000
            inp, tgt, msk = byte_example_to_tensor(rule, seed, args.byte_max_len or args.max_len)
            input_ids_list.append(inp)
            target_list.append(tgt)
            mask_list.append(msk)

        input_ids = torch.stack(input_ids_list)
        targets = torch.stack(target_list)
        mask = torch.stack(mask_list)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none",
        )
        loss = loss.view_as(mask)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
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
            "n_encoder": n_enc, "n_core": model.n_core, "n_decoder": n_dec,
            "dim": dim, "rule": args.rule,
            "trainable_params": trainable, "frozen_params": total - trainable,
        },
        "loss_trace": losses,
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return losses


def phase4_from_scratch(args, exp_dir):
    """Phase 4 (control): From-scratch byte-level."""
    print("=" * 60)
    print(f"Phase 4 (control): From-scratch byte-level (rule={args.rule})")
    print("=" * 60)

    n_enc = getattr(args, 'n_encoder_layers', 1)
    n_dec = getattr(args, 'n_decoder_layers', 1)

    from src.rule_generator import RULES as CHAR_RULES
    rule = CHAR_RULES[args.rule]

    dummy_core = RWKVNano(
        vocab_size=BYTE_VOCAB_SIZE, dim=args.dim,
        num_layers=n_enc + n_dec + 1, pad_token_id=BYTE_PAD,
    )
    model = ByteLevelRWKV(dummy_core, n_enc, n_dec)
    for p in model.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total:,} (all trainable)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        input_ids_list, target_list, mask_list = [], [], []
        for i in range(args.batch_size):
            seed = step * args.batch_size + i + 200000
            inp, tgt, msk = byte_example_to_tensor(rule, seed, args.byte_max_len or args.max_len)
            input_ids_list.append(inp)
            target_list.append(tgt)
            mask_list.append(msk)

        input_ids = torch.stack(input_ids_list)
        targets = torch.stack(target_list)
        mask = torch.stack(mask_list)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none",
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
        "config": {"vocab_kind": "byte_scratch", "dim": args.dim, "rule": args.rule},
        "loss_trace": losses,
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return losses


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RWKV world tokenizer surgery + distillation")
    ap.add_argument("--exp_id", default="rwkv7_surgery_002")
    ap.add_argument("--phase", default="1234")
    ap.add_argument("--rule", default="sum_threshold",
                    choices=["sum_threshold", "vowel_majority", "endpoint_match",
                             "count_trigger", "parity", "modulo3"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--pretrain_steps", type=int, default=None)
    ap.add_argument("--finetune_steps", type=int, default=None)
    ap.add_argument("--distil_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--byte_max_len", type=int, default=None)
    ap.add_argument("--distil_max_tokens", type=int, default=32)
    ap.add_argument("--distil_max_bytes", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n_encoder_layers", type=int, default=None)
    ap.add_argument("--n_decoder_layers", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    p1_steps = args.pretrain_steps or args.steps
    p23_steps = args.finetune_steps or args.steps
    p0_steps = args.distil_steps or args.steps
    p4_steps = args.finetune_steps or args.steps

    if args.n_encoder_layers is None:
        args.n_encoder_layers = max(1, args.layers // 3)
    if args.n_decoder_layers is None:
        args.n_decoder_layers = max(1, args.layers // 3)
    if args.byte_max_len is None:
        args.byte_max_len = args.max_len * 2

    torch.manual_seed(args.seed)
    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["git_hash"] = get_git_hash()
    config["p0_steps"] = p0_steps
    config["p1_steps"] = p1_steps
    config["p23_steps"] = p23_steps
    config["p4_steps"] = p4_steps
    config["world_vocab_size"] = WORLD_VOCAB_SIZE
    config["byte_vocab_size"] = BYTE_VOCAB_SIZE
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    n_core = args.layers - args.n_encoder_layers - args.n_decoder_layers
    print(f"Config: {exp_dir / 'config.json'}")
    print(f"  layers: {args.layers} (enc={args.n_encoder_layers}, "
          f"core={n_core}, dec={args.n_decoder_layers})")
    print(f"  world vocab: {WORLD_VOCAB_SIZE}, byte vocab: {BYTE_VOCAB_SIZE}")
    print()

    results = {}

    if "1" in args.phase:
        pretrained_path = exp_dir / "pretrained_world.pt"
        if not pretrained_path.exists():
            pretrained_path = phase1_pretrain(
                argparse.Namespace(**{**vars(args), "steps": p1_steps}), exp_dir,
            )
        else:
            print(f"Phase 1 already done: {pretrained_path}")
        results["pretrain_path"] = str(pretrained_path)

    if "0" in args.phase:
        pretrained_path = exp_dir / "pretrained_world.pt"
        if not pretrained_path.exists():
            print("ERROR: Phase 1 must complete before Phase 0.")
            sys.exit(1)
        distil_path = phase0_distillation(
            argparse.Namespace(**{**vars(args), "steps": p0_steps}), exp_dir,
            pretrained_path,
        )
        results["distil_path"] = str(distil_path)

    if "2" in args.phase or "3" in args.phase:
        pretrained_path = exp_dir / "pretrained_world.pt"
        if not pretrained_path.exists():
            print("ERROR: Phase 1 must complete before Phase 2-3.")
            sys.exit(1)
        distil_path = exp_dir / "distilled_encoder.pt" if (exp_dir / "distilled_encoder.pt").exists() else None
        surgery_losses = phase23_surgery(
            argparse.Namespace(**{**vars(args), "steps": p23_steps}), exp_dir,
            pretrained_path, distil_path,
        )
        results["surgery_final_loss"] = surgery_losses[-1] if surgery_losses else None
        results["surgery_loss_trace"] = surgery_losses

    if "4" in args.phase:
        scratch_losses = phase4_from_scratch(
            argparse.Namespace(**{**vars(args), "steps": p4_steps}), exp_dir,
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

    for k, v in results.items():
        if isinstance(v, list):
            results[k] = [round(x, 4) for x in v[-10:]]
    (exp_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {exp_dir}/")


if __name__ == "__main__":
    main()
