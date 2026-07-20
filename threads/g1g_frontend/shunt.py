#!/usr/bin/env python3
"""Shunt: train a byte encoder that feeds into frozen RWKV-7 layer 1.

Instead of splitting layers or replacing weights, the shunt sits in front
of the frozen model and learns to produce state vectors that layer 1+
can consume. Layer 0 is completely bypassed.

Architecture:
    bytes → byte_embed → shunt_encoder → [frozen layer 1, layer 2, ...] → head

The shunt_encoder must produce:
- x: (B, T, dim) hidden states
- state: dict with 'xx' (B, dim) and 'state' (B, H, N, N) matrix

Training:
    1. Distil: match shunt encoder states to layer 0's output states
    2. Task: train shunt on byte-level next-token prediction
    3. Repeat: alternating loop compounds the alignment

Usage:
    # Pre-train world model
    PYTHONPATH=. python src/shunt.py --exp_id shunt_001 --mode pretrain --steps 500

    # Alternating shunt training
    PYTHONPATH=. python src/shunt.py --exp_id shunt_001 --mode alternate --rounds 6
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
from domains.rwkv.rwkv_nano import RWKV7Nano, RWKV7Block, count_params

WORLD_VOCAB_PATH = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
world_tokenizer = RWKV_TOKENIZER(str(WORLD_VOCAB_PATH))
WORLD_VOCAB_SIZE = len(world_tokenizer.token2idx)  # 65529
WORLD_PAD_ID = 0

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


# ── Shunt encoder ───────────────────────────────────────────────────────

class ShuntEncoder(nn.Module):
    """Tiny encoder that maps bytes → state vectors for frozen layer 1.

    Learns to mimic what layer 0 would have output, so the rest of the
    model (layer 1, layer 2, ..., head) can run unmodified.
    """

    def __init__(self, dim: int, head_size: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.head_size = head_size
        self.n_heads = n_heads

        self.byte_embed = nn.Embedding(BYTE_VOCAB_SIZE, dim, padding_idx=BYTE_PAD)

        # Lightweight processor: 1 RWKV-7 block, initialized for byte-level
        # This block learns the same transformation layer 0 does,
        # but trained via state-matching distillation.
        self.processor = RWKV7Block(dim, head_size)

    def forward(self, input_ids: torch.Tensor):
        """Encode bytes → (hidden_states, state_dict).

        Returns:
            x: (B, T, dim) hidden states suitable for layer 1
            state: dict with 'xx' (B, dim) and 'state' (B, H, N, N)
        """
        x = self.byte_embed(input_ids)
        # Run through processor (like layer 0 does for tokens)
        x, state, _ = self.processor(x)
        return x, state


class ShuntedModel(nn.Module):
    """Full model: shunt → frozen layers 1..L-1 → head."""

    def __init__(self, core: RWKV7Nano, shunt: ShuntEncoder):
        super().__init__()
        self.dim = core.dim
        self.head_size = core.head_size
        self.n_head = core.n_head

        self.shunt = shunt

        # Freeze all core layers (1..L-1) + ln_out + head
        self.core_layers = nn.ModuleList(core.blocks[1:])  # skip layer 0
        for p in self.core_layers.parameters():
            p.requires_grad = False

        self.ln_out = core.ln_out
        for p in self.ln_out.parameters():
            p.requires_grad = False

        self.head = core.head
        for p in self.head.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        # Shunt: bytes → states for layer 1
        x, state = self.shunt(input_ids)

        # Run through frozen layers 1..L-1
        for block in self.core_layers:
            x, state, _ = block(x, state)

        x = self.ln_out(x)
        return self.head(x)

    def forward_with_states(self, input_ids: torch.Tensor):
        """Returns both logits and intermediate states for distillation."""
        B, T = input_ids.shape
        x, state = self.shunt(input_ids)
        shunt_out = x  # (B, T, dim) — what layer 1 receives

        for block in self.core_layers:
            x, state, _ = block(x, state)

        x = self.ln_out(x)
        logits = self.head(x)
        return logits, shunt_out


# ── Data ─────────────────────────────────────────────────────────────────

class SyntheticDataset:
    """Simple rule-based dataset for quick experiments."""

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


# ── Training functions ──────────────────────────────────────────────────

@torch.no_grad()
def get_layer0_states(core_model: RWKV7Nano, input_ids: torch.Tensor):
    """Extract state vectors after layer 0 of the frozen core model."""
    x = core_model.embed(input_ids)
    x, state, _ = core_model.blocks[0](x)
    return x, state


def run_distillation(shunt, core_model, dataset, steps, batch_size, lr, device):
    """Train shunt encoder to match layer 0's output states."""
    optimizer = torch.optim.AdamW(shunt.parameters(), lr=lr)

    shunt.train()
    core_model.eval()
    t_start = time.time()
    for step in range(1, steps + 1):
        # Generate a batch of text
        texts = []
        for _ in range(batch_size):
            dataset.rule.rng = dataset.rng
            texts.append(dataset.rule.generate()["text"])

        # Encode with world tokenizer + spans
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

        # World model forward through layer 0 → get target states
        world_ids = torch.zeros(batch_size, max_nt, dtype=torch.long, device=device)
        for bi, toks in enumerate(all_tokens):
            for ti, tid in enumerate(toks[:max_nt]):
                world_ids[bi, ti] = tid

        target_x, target_state = get_layer0_states(core_model, world_ids)

        # Also get the initial state xx for layer 1 matching
        target_xx = target_state['xx']  # (B, dim)

        # Byte encode the same texts
        byte_ids = torch.zeros(batch_size, 128, dtype=torch.long, device=device)
        for bi, text in enumerate(texts):
            raw = text.encode("utf-8")
            byte_toks = [BYTE_TO_ID[b] for b in raw[:128]]
            for ti, tid in enumerate(byte_toks):
                byte_ids[bi, ti] = tid

        # Shunt forward → get byte states
        shunt_x, shunt_state = shunt(byte_ids)

        # Pool byte states per token span
        pooled_x = pool_byte_states(shunt_x, all_spans, max_nt)

        # MSE on hidden states
        mask = (target_x.abs().sum(dim=-1) > 1e-8).float()
        diff_x = (pooled_x - target_x).pow(2).sum(dim=-1)
        loss_x = (diff_x * mask).sum() / (mask.sum() + 1e-8)

        # MSE on xx state (last timestep's hidden, what feeds into layer 1's time-mixing)
        shunt_xx = shunt_state['xx']  # (B, dim)
        loss_xx = (shunt_xx - target_xx).pow(2).mean()

        loss = loss_x + 0.1 * loss_xx

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shunt.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % max(1, steps // 5) == 0:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            print(f"    distil step {step:4d}/{steps}  loss_x={loss_x.item():.6f}  "
                  f"loss_xx={loss_xx.item():.6f}  {sps:.1f} st/s")

    return loss_x.item(), loss_xx.item()


def run_task(shunted, dataset, steps, batch_size, lr, device):
    """Train shunt on byte-level next-token prediction."""
    optimizer = torch.optim.AdamW(shunted.shunt.parameters(), lr=lr)

    shunted.train()
    t_start = time.time()
    losses = []
    for step in range(1, steps + 1):
        ids, targets, mask = dataset.get_byte_batch(batch_size, 128)
        ids, targets, mask = ids.to(device), targets.to(device), mask.to(device)

        logits = shunted(ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none",
        )
        loss = loss.view_as(mask)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shunted.shunt.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step == 1 or step % max(1, steps // 5) == 0:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1e-6)
            recent = sum(losses[-min(len(losses), 50):]) / min(len(losses), 50)
            print(f"    task step {step:4d}/{steps}  loss={recent:.4f}  {sps:.1f} st/s")

    return losses


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Shunt: bypass layer 0, train byte encoder for layer 1")
    ap.add_argument("--exp_id", default="shunt_001")
    ap.add_argument("--mode", default="full", choices=["pretrain", "alternate", "full"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--head_size", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
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

    dataset = SyntheticDataset(args.seed)
    print(f"Config: {exp_dir / 'config.json'}")
    print(f"  dim={args.dim}, head={args.head_size}, layers={args.layers}")
    print(f"  mode={args.mode}, rounds={args.rounds}")
    print()

    # Phase 1: Pre-train the world-tokenizer model
    if args.mode in ("pretrain", "full"):
        print("=" * 60)
        print("Phase 1: Pre-train world-tokenizer model")
        print("=" * 60)

        core = RWKV7Nano(
            vocab_size=WORLD_VOCAB_SIZE, dim=args.dim,
            head_size=args.head_size, num_layers=args.layers,
        ).to(device)
        optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr)
        print(f"  Params: {count_params(core):,}")

        t_start = time.time()
        for step in range(1, args.steps + 1):
            ids, targets, mask = dataset.get_world_batch(args.batch_size, 64)
            ids, targets, mask = ids.to(device), targets.to(device), mask.to(device)
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

        torch.save({
            "model_state": core.state_dict(),
            "config": {"dim": args.dim, "head_size": args.head_size,
                       "num_layers": args.layers, "vocab_size": WORLD_VOCAB_SIZE},
        }, exp_dir / "pretrained_world.pt")
        print(f"  saved: {exp_dir / 'pretrained_world.pt'}")

    # Phase 2+: Alternating shunt training
    if args.mode in ("alternate", "full"):
        pretrain_path = exp_dir / "pretrained_world.pt"
        if not pretrain_path.exists():
            print("ERROR: Need pretrained model first")
            sys.exit(1)

        ckpt = torch.load(pretrain_path, map_location=device)
        cfg = ckpt["config"]
        core = RWKV7Nano(
            vocab_size=cfg["vocab_size"], dim=cfg["dim"],
            head_size=cfg["head_size"], num_layers=cfg["num_layers"],
        ).to(device)
        core.load_state_dict(ckpt["model_state"])
        core.eval()
        for p in core.parameters():
            p.requires_grad = False

        # Build shunt
        shunt = ShuntEncoder(cfg["dim"], cfg["head_size"], cfg["dim"] // cfg["head_size"])
        init_shunt(shunt, core)
        shunt.to(device)

        shunted = ShuntedModel(core, shunt).to(device)
        trainable = sum(p.numel() for p in shunted.shunt.parameters())
        frozen = sum(p.numel() for p in shunted.parameters()) - trainable
        print()
        print("=" * 60)
        print(f"Shunt training: {args.rounds} alternating rounds")
        print(f"  Trainable (shunt only): {trainable:,}")
        print(f"  Frozen (layers 1..L-1 + ln_out + head): {frozen:,}")
        print(f"  Total: {trainable + frozen:,}")
        print("=" * 60)

        for r in range(1, args.rounds + 1):
            d_steps = max(25, args.steps // (r + 1))
            t_steps = max(50, args.steps // (r + 1))
            print(f"\n--- Round {r}/{args.rounds} (distil={d_steps}, task={t_steps}) ---")

            # Distil: match layer 0 states
            run_distillation(shunt, core, dataset, d_steps, min(4, args.batch_size), args.lr, device)

            # Task: byte-level next-token prediction
            run_task(shunted, dataset, t_steps, args.batch_size, args.lr, device)

            # Eval
            shunted.eval()
            with torch.no_grad():
                val_ids, val_tgt, val_msk = dataset.get_byte_batch(4, 128)
                val_ids, val_tgt, val_msk = val_ids.to(device), val_tgt.to(device), val_msk.to(device)
                val_logits = shunted(val_ids)
                val_loss = F.cross_entropy(
                    val_logits.view(-1, val_logits.size(-1)),
                    val_tgt.view(-1), reduction="none",
                )
                val_loss = (val_loss.view_as(val_msk) * val_msk).sum() / (val_msk.sum() + 1e-8)
                print(f"  → Eval loss after round {r}: {val_loss.item():.4f}")

        torch.save({
            "shunt_state": shunt.state_dict(),
            "config": {"dim": cfg["dim"], "head_size": cfg["head_size"],
                       "rounds": args.rounds},
        }, exp_dir / "shunt_final.pt")
        print(f"\nSaved: {exp_dir / 'shunt_final.pt'}")


def init_shunt(shunt: ShuntEncoder, core: RWKV7Nano):
    """Initialize shunt from layer 0 weights where possible."""
    with torch.no_grad():
        # Init byte embed from first 256 token positions
        shunt.byte_embed.weight.zero_()
        for bv in range(256):
            shunt.byte_embed.weight[BYTE_TO_ID[bv]].copy_(
                core.embed.weight[bv]
            )
        # Init processor from layer 0's weights
        shunt.processor.load_state_dict(core.blocks[0].state_dict())
    print(f"  Shunt initialized from layer 0: "
          f"{sum(p.numel() for p in shunt.parameters()):,} params")


if __name__ == "__main__":
    main()
