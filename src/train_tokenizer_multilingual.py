#!/usr/bin/env python3
"""Train the byte-level tokenizer interface on multilingual TinyStories.

Uses alternating training (distil → task → distil → task → ...) with
real multilingual story data instead of synthetic rules.

Usage:
    # Pre-train on world tokenizer + multilingual stories
    PYTHONPATH=. python src/train_tokenizer_multilingual.py --exp_id ml_train_001 --mode pretrain --steps 1000

    # Alternating loop
    PYTHONPATH=. python src/train_tokenizer_multilingual.py --exp_id ml_train_001 --mode alternate --rounds 6

    # Full run
    PYTHONPATH=. python src/train_tokenizer_multilingual.py --exp_id ml_train_001 --mode full --steps 500 --rounds 6
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

from src.hf_rwkv_tokenizer import RWKV_TOKENIZER
from src.multilingual_tinystories import TinyStoriesDataset
from src.rwkv_nano import RWKV7Nano, count_params

WORLD_VOCAB_PATH = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
world_tokenizer = RWKV_TOKENIZER(str(WORLD_VOCAB_PATH))
WORLD_VOCAB_SIZE = len(world_tokenizer.token2idx)  # 65529
WORLD_PAD_ID = 0

# Byte vocab
BYTE_VOCAB_SIZE = 258
BYTE_PAD = 0
BYTE_TO_ID = {b: 2 + b for b in range(256)}


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def encode_with_spans(text: str, max_tokens: int = 64):
    """Encode with world tokenizer, return (token_ids, byte_spans)."""
    raw = text.encode("utf-8")
    tokens = []
    spans = []
    idx = 0
    while idx < len(raw):
        old_idx = idx
        idx, node, values = world_tokenizer.root.find_longest(raw, idx)
        if idx == old_idx:
            break
        _, token_id = next(iter(values))
        tokens.append(token_id)
        spans.append((old_idx, idx))
        if len(tokens) >= max_tokens:
            break
    return tokens, spans


def pool_byte_states(byte_states, spans, n_tokens):
    """Average byte states per token span."""
    B, _, D = byte_states.shape
    pooled = byte_states.new_zeros(B, n_tokens, D)
    for bi in range(B):
        for ti, (s, e) in enumerate(spans[bi]):
            if ti >= n_tokens or e > byte_states.shape[1]:
                break
            if e > s:
                pooled[bi, ti] = byte_states[bi, s:e].mean(dim=0)
    return pooled


class ByteLevelModel(nn.Module):
    """Byte-level encoder + frozen RWKV-7 core + byte-level decoder."""

    def __init__(self, core_model, n_encoder=1, n_decoder=1):
        super().__init__()
        self.dim = core_model.dim
        self.n_core = core_model.num_layers - n_encoder - n_decoder
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        assert self.n_core >= 0

        self.byte_embed = nn.Embedding(BYTE_VOCAB_SIZE, self.dim, padding_idx=BYTE_PAD)
        self.byte_encoder = nn.ModuleList([
            type(core_model.blocks[0])(self.dim, core_model.head_size)
            for _ in range(n_encoder)
        ])

        # Frozen core: middle layers from pre-trained model
        self.core = nn.ModuleList()
        start = n_encoder
        for i in range(start, start + self.n_core):
            layer = type(core_model.blocks[0])(self.dim, core_model.head_size)
            layer.load_state_dict(core_model.blocks[i].state_dict())
            for p in layer.parameters():
                p.requires_grad = False
            self.core.append(layer)

        self.byte_decoder = nn.ModuleList([
            type(core_model.blocks[0])(self.dim, core_model.head_size)
            for _ in range(n_decoder)
        ])

        self.ln_out = nn.LayerNorm(self.dim)
        self.ln_out.weight.data.copy_(core_model.ln_out.weight.data)
        self.ln_out.bias.data.copy_(core_model.ln_out.bias.data)

        self.byte_head = nn.Linear(self.dim, BYTE_VOCAB_SIZE, bias=True)

    def forward_encoder(self, input_ids):
        x = self.byte_embed(input_ids)
        for block in self.byte_encoder:
            x, _, _ = block(x)
        return x

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.byte_embed(input_ids)
        for block in self.byte_encoder:
            x, _, _ = block(x)
        for block in self.core:
            x, _, _ = block(x)
        for block in self.byte_decoder:
            x, _, _ = block(x)
        x = self.ln_out(x)
        return self.byte_head(x)


def create_model(core_model, n_encoder=1, n_decoder=1):
    """Create ByteLevelModel, init from world weights."""
    model = ByteLevelModel(core_model, n_encoder, n_decoder)
    with torch.no_grad():
        # Init byte embed from world tokenizer's first 256 tokens
        model.byte_embed.weight.zero_()
        for bv in range(256):
            model.byte_embed.weight[BYTE_TO_ID[bv]].copy_(
                core_model.embed.weight[bv]
            )
        # Init byte head
        model.byte_head.weight.zero_()
        model.byte_head.bias.zero_()
        for bv in range(256):
            model.byte_head.weight[BYTE_TO_ID[bv]].copy_(
                core_model.head.weight[bv]
            )
            model.byte_head.bias[BYTE_TO_ID[bv]].copy_(
                core_model.head.bias[bv]
            )
        # Init encoder from first n layers
        if n_encoder > 0:
            for new_b, old_b in zip(model.byte_encoder, core_model.blocks[:n_encoder]):
                new_b.load_state_dict(old_b.state_dict())
        # Init decoder from second-to-last layers
        if n_decoder > 0:
            start = max(0, len(core_model.blocks) - n_decoder - 1)
            for new_b, old_b in zip(
                model.byte_decoder, core_model.blocks[start:start + n_decoder]
            ):
                new_b.load_state_dict(old_b.state_dict())
    return model


def run_distillation(model, core_model, dataset, steps, batch_size, lr,
                     enc_layer_idx, device):
    """Train byte encoder to match world states via MSE."""
    optimizer = torch.optim.AdamW(
        [p for p in model.byte_embed.parameters()] +
        [p for p in model.byte_encoder.parameters()], lr=lr,
    )
    for p in model.core.parameters():
        p.requires_grad = False
    for p in model.byte_decoder.parameters():
        p.requires_grad = False
    for p in model.ln_out.parameters():
        p.requires_grad = False
    for p in model.byte_head.parameters():
        p.requires_grad = False

    model.train()
    core_model.eval()
    losses = []
    t_start = time.time()
    for step in range(1, steps + 1):
        # Get a batch of texts (random stories)
        texts = [dataset.rng.choice(dataset.stories)["output"] for _ in range(batch_size)]
        # Encode with spans
        all_tokens = []
        all_spans = []
        max_nt = 0
        for text in texts:
            toks, spans = encode_with_spans(text, 32)
            if len(toks) > 0:
                all_tokens.append(toks)
                all_spans.append(spans)
                max_nt = max(max_nt, len(toks))

        if max_nt == 0:
            continue

        # World model forward → states
        world_ids = torch.zeros(batch_size, max_nt, dtype=torch.long)
        for bi, toks in enumerate(all_tokens):
            for ti, tid in enumerate(toks[:max_nt]):
                world_ids[bi, ti] = tid

        with torch.no_grad():
            # Extract states after encoder layer
            x = core_model.embed(world_ids.to(device))
            for i in range(enc_layer_idx + 1):
                x, _, _ = core_model.blocks[i](x)
            world_states = x

        # Byte encoder forward
        # Build byte inputs from the same texts
        byte_ids = torch.zeros(batch_size, 128, dtype=torch.long, device=device)
        for bi, text in enumerate(texts):
            raw = text.encode("utf-8")
            byte_toks = [BYTE_TO_ID[b] for b in raw[:128]]
            for ti, tid in enumerate(byte_toks):
                byte_ids[bi, ti] = tid

        byte_states = model.forward_encoder(byte_ids)

        # Pool and compute MSE
        # Adjust spans to match byte_states tensor
        pooled = pool_byte_states(byte_states, all_spans, max_nt)
        mask = (world_states.abs().sum(dim=-1) > 1e-8).float()
        diff = (pooled - world_states).pow(2).sum(dim=-1)
        loss = (diff * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.byte_embed.parameters()] +
            [p for p in model.byte_encoder.parameters()], 1.0,
        )
        optimizer.step()
        losses.append(loss.item())

        if step == 1 or step % max(1, steps // 5) == 0:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            print(f"    distil step {step:4d}/{steps}  mse={loss.item():.6f}  {sps:.1f} st/s")

    return losses


def run_task(model, dataset, steps, batch_size, lr, device):
    """Train encoder + decoder on byte-level next-token prediction."""
    for p in model.parameters():
        p.requires_grad = True
    for p in model.core.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )

    model.train()
    losses = []
    t_start = time.time()
    for step in range(1, steps + 1):
        ids, targets, mask = dataset.get_byte_batch(batch_size, 128)
        ids, targets, mask = ids.to(device), targets.to(device), mask.to(device)

        logits = model(ids)
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

        if step == 1 or step % max(1, steps // 5) == 0:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            recent = sum(losses[-min(len(losses), 50):]) / min(len(losses), 50)
            print(f"    task step {step:4d}/{steps}  loss={recent:.4f}  {sps:.1f} st/s")

    return losses


def main():
    ap = argparse.ArgumentParser(description="Multilingual tokenizer training")
    ap.add_argument("--exp_id", default="ml_train_001")
    ap.add_argument("--mode", default="full",
                    choices=["pretrain", "alternate", "full"])
    ap.add_argument("--steps", type=int, default=500,
                    help="Steps per phase")
    ap.add_argument("--rounds", type=int, default=6,
                    help="Alternating rounds")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--head_size", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--languages", type=str, default="en,fr,de,ja,zh,ar,hi,ru,es,pt",
                    help="Comma-separated language codes")
    ap.add_argument("--max_stories", type=int, default=10000)
    ap.add_argument("--min_score", type=float, default=7.0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    exp_dir = Path("experiments") / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["git_hash"] = get_git_hash()
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Config: {exp_dir / 'config.json'}")
    print(f"  languages: {args.languages}")
    print(f"  mode: {args.mode}, rounds: {args.rounds}")
    print()

    # Load dataset
    languages = [l.strip() for l in args.languages.split(",")]
    dataset = TinyStoriesDataset(
        languages=languages,
        min_score=args.min_score,
        max_stories=args.max_stories,
        seed=args.seed,
    )
    print(f"Dataset: {len(dataset)} stories")
    print()

    # Phase 1: Pre-train on world tokenizer
    if args.mode in ("pretrain", "full"):
        print("=" * 60)
        print(f"Phase 1: Pre-training on world tokenizer ({len(languages)} langs)")
        print("=" * 60)

        core_model = RWKV7Nano(
            vocab_size=WORLD_VOCAB_SIZE, dim=args.dim,
            head_size=args.head_size, num_layers=args.layers,
        ).to(device)

        optimizer = torch.optim.AdamW(core_model.parameters(), lr=args.lr)
        print(f"Model params: {count_params(core_model):,}")

        t_start = time.time()
        for step in range(1, args.steps + 1):
            ids, targets, mask = dataset.get_world_batch(args.batch_size, 64)
            ids, targets, mask = ids.to(device), targets.to(device), mask.to(device)

            logits, _ = core_model(ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none",
            )
            loss = loss.view_as(mask)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % args.log_every == 0:
                elapsed = time.time() - t_start
                sps = step / max(elapsed, 1e-6)
                print(f"  step {step:4d}/{args.steps}  loss={loss.item():.4f}  {sps:.1f} st/s")

        ckpt_path = exp_dir / "pretrained_world.pt"
        torch.save({
            "model_state": core_model.state_dict(),
            "config": {
                "vocab_size": WORLD_VOCAB_SIZE, "dim": args.dim,
                "head_size": args.head_size, "num_layers": args.layers,
                "languages": languages,
            },
        }, ckpt_path)
        print(f"  saved: {ckpt_path}")

    # Phase 2+: Alternating loop
    if args.mode in ("alternate", "full"):
        pretrain_path = exp_dir / "pretrained_world.pt"
        if not pretrain_path.exists():
            print("ERROR: Need pretrained model first")
            sys.exit(1)

        ckpt = torch.load(pretrain_path, map_location=device)
        cfg = ckpt["config"]
        core_model = RWKV7Nano(
            vocab_size=cfg["vocab_size"], dim=cfg["dim"],
            head_size=cfg["head_size"], num_layers=cfg["num_layers"],
        ).to(device)
        core_model.load_state_dict(ckpt["model_state"])
        core_model.eval()
        for p in core_model.parameters():
            p.requires_grad = False

        n_enc = 1
        n_dec = 1
        model = create_model(core_model, n_enc, n_dec).to(device)
        enc_layer_idx = n_enc - 1

        print()
        print("=" * 60)
        print(f"Alternating loop: {args.rounds} rounds (distil→task→...)")
        print(f"  Model: {sum(p.numel() for p in model.parameters()):,} total, "
              f"core frozen: {sum(p.numel() for p in model.core.parameters()):,}")
        print("=" * 60)

        for r in range(1, args.rounds + 1):
            d_steps = max(25, args.steps // (r + 1))
            t_steps = max(50, args.steps // (r + 1))
            print(f"\n--- Round {r}/{args.rounds} (distil={d_steps}, task={t_steps}) ---")

            # Distil
            run_distillation(
                model, core_model, dataset,
                steps=d_steps, batch_size=min(4, args.batch_size),
                lr=args.lr, enc_layer_idx=enc_layer_idx, device=device,
            )

            # Task
            run_task(
                model, dataset,
                steps=t_steps, batch_size=args.batch_size,
                lr=args.lr, device=device,
            )

            # Log current loss
            model.eval()
            with torch.no_grad():
                val_ids, val_tgt, val_msk = dataset.get_byte_batch(4, 128)
                val_ids, val_tgt, val_msk = val_ids.to(device), val_tgt.to(device), val_msk.to(device)
                val_logits = model(val_ids)
                val_loss = F.cross_entropy(
                    val_logits.view(-1, val_logits.size(-1)),
                    val_tgt.view(-1), reduction="none",
                )
                val_loss = (val_loss.view_as(val_msk) * val_msk).sum() / (val_msk.sum() + 1e-8)
                print(f"  → Eval loss after round {r}: {val_loss.item():.4f}")

        # Save
        torch.save({
            "model_state": model.state_dict(),
            "config": {
                "vocab_kind": "byte_multilingual",
                "dim": cfg["dim"],
                "head_size": cfg["head_size"],
                "n_encoder": n_enc, "n_core": model.n_core, "n_decoder": n_dec,
                "rounds": args.rounds, "languages": languages,
            },
        }, exp_dir / "alternating_final.pt")
        print(f"\nSaved: {exp_dir / 'alternating_final.pt'}")


if __name__ == "__main__":
    main()
