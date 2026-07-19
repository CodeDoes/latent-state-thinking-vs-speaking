#!/usr/bin/env python3
"""Tokenizer surgery experiment for RWKV nano.

Three-phase experiment:
  Phase 1: Pre-train a token-level RWKV on a synthetic rule task.
  Phase 2: Replace embed + head with byte-level versions ("surgery").
  Phase 3: Freeze core RWKV blocks, train only the new byte interface.
  Phase 4 (control): Same architecture, train from scratch on bytes.

Single variable: does pre-trained RWKV knowledge transfer through
a tokenizer surgery (token → byte) interface replacement?

Usage:
    # Full run (all phases)
    python src/token_surgery.py --exp_id token_surgery_001

    # Phase 1 only (pre-train)
    python src/token_surgery.py --exp_id token_surgery_001 --phase 1 --steps 300

    # Phase 2+3 (surgery + finetune)
    python src/token_surgery.py --exp_id token_surgery_001 --phase 23 --steps 500

    # Phase 4 (control: from scratch on bytes)
    python src/token_surgery.py --exp_id token_surgery_scratch_001 --phase 4 --steps 500
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

# ── Phase 1 imports (character-level) ──
from src.rule_generator import (
    RULES as CHAR_RULES,
    encode as char_encode,
    decode as char_decode,
    PAD_ID as CHAR_PAD,
    VOCAB as CHAR_VOCAB,
    char_to_id,
    id_to_char,
)

# ── Phase 2+ imports (byte-level) ──
from src.byte_vocab import (
    encode as byte_encode,
    decode as byte_decode,
    VOCAB_SIZE as BYTE_VOCAB_SIZE,
    PAD_ID as BYTE_PAD,
    BYTE_TO_ID,
    ID_TO_BYTE,
)

from src.rwkv_nano import RWKVNano, count_params


# ── Helpers ───────────────────────────────────────────────────────────────

def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_char_to_byte_map(old_vocab_size: int) -> dict[int, int]:
    """Build a map from old token ID → byte token ID, for copying weights.

    The old tokenizer uses CHAR_VOCAB (chars like '0', '1', ..., 'a', 'b', ...
    plus special tokens <PAD>, <UNK>, <BOS>, <EOS>).

    The byte tokenizer uses BYTE_TO_ID which maps byte values to token IDs.

    For ASCII printable characters, the mapping is:
      old_token_id(char) → BYTE_TO_ID[ord(char)]

    For special tokens, we map to the closest byte equivalent:
      <PAD> → BYTE_PAD (0)
      <UNK> → BYTE_UNK (1)
      other specials → BYTE_UNK
    """
    mapping = {}
    special_map = {"<PAD>": BYTE_PAD, "<UNK>": 1}  # UNK_ID = 1
    for old_id, char in enumerate(CHAR_VOCAB):
        if char in special_map:
            mapping[old_id] = special_map[char]
        elif len(char) == 1:
            b = ord(char)
            if b in BYTE_TO_ID:
                mapping[old_id] = BYTE_TO_ID[b]
            else:
                mapping[old_id] = 1  # UNK
        else:
            mapping[old_id] = 1  # UNK for multi-char tokens
    return mapping


def perform_surgery(pretrained_model: RWKVNano, new_vocab_size: int) -> RWKVNano:
    """Perform tokenizer surgery on a pre-trained RWKVNano.

    Returns a new RWKVNano with:
    - Byte-level embed (new_vocab_size) initialized from old embed where possible
    - Byte-level head (new_vocab_size) initialized from old head where possible
    - All RWKV block weights copied from pretrained_model
    """
    old_vocab = pretrained_model.vocab_size
    dim = pretrained_model.dim
    num_layers = pretrained_model.num_layers

    # Build the character → byte mapping
    char_to_byte_id = build_char_to_byte_map(old_vocab)

    # Create a new model with byte vocab
    new_model = RWKVNano(
        vocab_size=new_vocab_size,
        dim=dim,
        num_layers=num_layers,
        pad_token_id=BYTE_PAD,
    )

    # Copy RWKV block weights (always frozen)
    with torch.no_grad():
        for old_block, new_block in zip(pretrained_model.blocks, new_model.blocks):
            new_block.load_state_dict(old_block.state_dict())

        # Copy layer norm and other non-block params
        new_model.ln_out.weight.copy_(pretrained_model.ln_out.weight)
        new_model.ln_out.bias.copy_(pretrained_model.ln_out.bias)

        # ── Surgery on embed ──
        # Zero-init the new embedding, then copy old weights where chars match bytes
        new_model.embed.weight.zero_()
        new_model.embed.weight[BYTE_PAD].zero_()  # PAD stays zero
        for old_id, byte_id in char_to_byte_id.items():
            if byte_id < new_vocab_size:
                new_model.embed.weight[byte_id].copy_(
                    pretrained_model.embed.weight[old_id]
                )

        # ── Surgery on head ──
        # Zero-init new head, copy old weights where possible
        new_model.head.weight.zero_()
        new_model.head.bias.zero_()
        for old_id, byte_id in char_to_byte_id.items():
            if byte_id < new_vocab_size:
                new_model.head.weight[byte_id].copy_(
                    pretrained_model.head.weight[old_id]
                )
                new_model.head.bias[byte_id].copy_(
                    pretrained_model.head.bias[old_id]
                )

    return new_model


# ── Data generation ────────────────────────────────────────────────────────

def char_example_to_tensor(rule, rng_seed: int, max_len: int = 128):
    """Generate one example using the char-level tokenizer."""
    import random
    rng = random.Random(rng_seed)
    rule.rng = rng
    ex = rule.generate()
    text = ex["text"]
    toks = char_encode(text)
    # Truncate/pad
    if len(toks) > max_len:
        toks = toks[:max_len]
    else:
        toks = toks + [CHAR_PAD] * (max_len - len(toks))
    input_ids = torch.tensor(toks, dtype=torch.long)
    targets = torch.roll(input_ids, shifts=-1)
    targets[-1] = CHAR_PAD
    # Mask: loss on label tokens only
    mask = torch.zeros(max_len, dtype=torch.float)
    # Label is at the last character of the answer
    label_pos = min(len(ex["text"]) - 1, max_len - 2)
    mask[label_pos] = 1.0
    return input_ids, targets, mask


def byte_example_to_tensor(rule, rng_seed: int, max_len: int = 128):
    """Generate one example using the byte-level tokenizer."""
    import random
    rng = random.Random(rng_seed)
    rule.rng = rng
    ex = rule.generate()
    text = ex["text"]
    toks = byte_encode(text, max_len=max_len)
    if len(toks) < max_len:
        toks = toks[:max_len]
    input_ids = torch.tensor(toks, dtype=torch.long)
    targets = torch.roll(input_ids, shifts=-1)
    targets[-1] = BYTE_PAD
    # Mask: loss on label tokens only
    mask = torch.zeros(max_len, dtype=torch.float)
    label_pos = min(len(ex["text"]) - 1, max_len - 2)
    mask[label_pos] = 1.0
    return input_ids, targets, mask


# ── Phase functions ────────────────────────────────────────────────────────

def phase1_pretrain(args, exp_dir):
    """Phase 1: Pre-train token-level RWKV on a char rule task."""
    print("=" * 60)
    print(f"Phase 1: Pre-training on char-level task (rule={args.rule})")
    print("=" * 60)

    rule = CHAR_RULES[args.rule]
    char_vocab_size = len(CHAR_VOCAB)

    model = RWKVNano(
        vocab_size=char_vocab_size,
        dim=args.dim,
        num_layers=args.layers,
        pad_token_id=CHAR_PAD,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model params: {count_params(model):,} (vocab={char_vocab_size})")

    t_start = time.time()
    for step in range(1, args.steps + 1):
        # Generate batch
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

    # Save checkpoint
    ckpt_path = exp_dir / "pretrained_char.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": char_vocab_size,
            "dim": args.dim,
            "num_layers": args.layers,
            "vocab_kind": "char",
            "rule": args.rule,
        },
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return ckpt_path


def phase23_surgery(args, exp_dir, pretrained_path: Path):
    """Phase 2+3: Surgery + fine-tune on byte-level task."""
    print("=" * 60)
    print(f"Phase 2-3: Surgery + byte-level fine-tune (rule={args.rule})")
    print("=" * 60)

    # Load pre-trained
    ckpt = torch.load(pretrained_path, map_location="cpu")
    old_vocab = ckpt["config"]["vocab_size"]
    dim = ckpt["config"]["dim"]
    layers = ckpt["config"]["num_layers"]

    old_model = RWKVNano(vocab_size=old_vocab, dim=dim, num_layers=layers)
    old_model.load_state_dict(ckpt["model_state"])
    print(f"Loaded pre-trained model (vocab={old_vocab}, dim={dim})")

    # Perform surgery
    model = perform_surgery(old_model, BYTE_VOCAB_SIZE)
    new_params = count_params(model)

    # Freeze RWKV blocks
    trainable_params = 0
    for name, p in model.named_parameters():
        if "embed" in name or "head" in name:
            p.requires_grad = True
            trainable_params += p.numel()
        else:
            p.requires_grad = False
    print(f"Frozen blocks: {new_params - trainable_params:,} params")
    print(f"Trainable (embed + head): {trainable_params:,} params")

    # Verify: logits for known char '0' should be reasonable after init
    with torch.no_grad():
        test_ids = torch.tensor([[BYTE_TO_ID[ord('0')]]])
        logits, _ = model(test_ids)
        top5 = logits[0, 0].topk(5)
        print(f"  After surgery: top-5 logits for byte '0': "
              f"{[ID_TO_BYTE.get(i.item(), '?') for i in top5.indices]} "
              f"(values={top5.values.tolist()})")

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
            seed = step * args.batch_size + i + 100000  # offset to avoid repeating phase-1 data
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

    # Save surgery model
    ckpt_path = exp_dir / "surgery_finetuned.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": BYTE_VOCAB_SIZE,
            "dim": dim,
            "num_layers": layers,
            "vocab_kind": "byte_surgery",
            "rule": args.rule,
            "trainable_params": trainable_params,
            "frozen_params": new_params - trainable_params,
        },
        "loss_trace": losses,
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return losses


def phase4_from_scratch(args, exp_dir):
    """Phase 4 (control): Train from scratch on bytes, same architecture."""
    print("=" * 60)
    print(f"Phase 4 (control): From-scratch byte-level (rule={args.rule})")
    print("=" * 60)

    rule = CHAR_RULES[args.rule]
    model = RWKVNano(
        vocab_size=BYTE_VOCAB_SIZE,
        dim=args.dim,
        num_layers=args.layers,
        pad_token_id=BYTE_PAD,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model params: {count_params(model):,} (vocab={BYTE_VOCAB_SIZE})")

    t_start = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        input_ids_list, target_list, mask_list = [], [], []
        for i in range(args.batch_size):
            seed = step * args.batch_size + i + 200000  # distinct seed range
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

    # Save
    ckpt_path = exp_dir / "from_scratch_byte.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": BYTE_VOCAB_SIZE,
            "dim": args.dim,
            "num_layers": args.layers,
            "vocab_kind": "byte_scratch",
            "rule": args.rule,
        },
        "loss_trace": losses,
    }, ckpt_path)
    print(f"  saved: {ckpt_path}")
    return losses


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Tokenizer surgery experiment")
    ap.add_argument("--exp_id", default="token_surgery_001")
    ap.add_argument("--phase", default="1234", help="Phases to run: 1, 23, 4, 1234")
    ap.add_argument("--rule", default="sum_threshold",
                    choices=list(CHAR_RULES.keys()))
    ap.add_argument("--steps", type=int, default=300,
                    help="Steps per phase (phase 1 and phase 23/4)")
    ap.add_argument("--pretrain_steps", type=int, default=None,
                    help="Override phase 1 steps")
    ap.add_argument("--finetune_steps", type=int, default=None,
                    help="Override phase 23/4 steps")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Resolve step overrides
    p1_steps = args.pretrain_steps if args.pretrain_steps is not None else args.steps
    p23_steps = args.finetune_steps if args.finetune_steps is not None else args.steps
    p4_steps = args.finetune_steps if args.finetune_steps is not None else args.steps

    torch.manual_seed(args.seed)
    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["git_hash"] = get_git_hash()
    config["p1_steps"] = p1_steps
    config["p23_steps"] = p23_steps
    config["p4_steps"] = p4_steps
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Config: {exp_dir / 'config.json'}")
    print(f"  git hash: {config['git_hash']}")
    print(f"  rule: {args.rule}")
    print(f"  phases: {args.phase} (p1={p1_steps} st, p23={p23_steps} st, p4={p4_steps} st)")
    print()

    results = {}

    # Phase 1: pre-train on chars
    if "1" in args.phase:
        pretrained_path = exp_dir / "pretrained_char.pt"
        if not pretrained_path.exists():
            pretrained_path = phase1_pretrain(
                argparse.Namespace(
                    **{**vars(args), "steps": p1_steps}
                ),
                exp_dir,
            )
        else:
            print(f"Phase 1 already done: {pretrained_path}")
        results["pretrain_path"] = str(pretrained_path)

    # Phase 2+3: surgery + fine-tune
    if "2" in args.phase or "3" in args.phase:
        pretrained_path = exp_dir / "pretrained_char.pt"
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

    # Phase 4: from scratch on bytes (control)
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
    if results.get("surgery_final_loss") is not None:
        print(f"  Surgery (pretrained → byte): final loss = {results['surgery_final_loss']:.4f}")
    if results.get("scratch_final_loss") is not None:
        print(f"  Scratch (byte from zero):    final loss = {results['scratch_final_loss']:.4f}")
    if results.get("surgery_final_loss") is not None and results.get("scratch_final_loss") is not None:
        diff = results["scratch_final_loss"] - results["surgery_final_loss"]
        print(f"  Δ (scratch - surgery):       {diff:+.4f}")
        if diff > 0:
            print(f"  → Surgery transfers knowledge (lower loss)")
        elif diff < 0:
            print(f"  → Surgery HURTS (from-scratch is better)")
        else:
            print(f"  → No measurable transfer")

    # Save metrics
    for k, v in results.items():
        if isinstance(v, list):
            # Don't save full traces in metrics.json (too big)
            results[k] = [round(x, 4) for x in v[-10:]]  # last 10
    (exp_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {exp_dir}/")


if __name__ == "__main__":
    main()
