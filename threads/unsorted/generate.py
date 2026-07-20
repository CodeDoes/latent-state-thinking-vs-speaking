#!/usr/bin/env python3
"""Fast generation using g1g with trained BlackGoose channel-mix.

Feeds one token at a time and preserves RNN states — ~0.9s per token.
Usage:
    python src/generate.py --prompt "Hello" --steps 100
"""

import sys, os, argparse
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from domains.g1g.g1g_blackgoose_nf4 import G1GBlackGooseNF4


def tokenize(text: str) -> list[int]:
    return [b + 2 for b in text.encode("utf-8")]


def detokenize(ids: list[int]) -> str:
    raw = bytes(b - 2 for b in ids if b >= 2)
    return raw.decode("utf-8", errors="replace")


def generate(model, prompt: str, n_steps: int,
             temperature: float = 0.8, top_p: float = 0.9) -> str:
    model.eval()
    device = next(model.parameters()).device

    prompt_ids = tokenize(prompt)
    if not prompt_ids:
        prompt_ids = [194]

    byte_buf = bytearray()

    print(f"Prompt ({len(prompt_ids)} bytes): {prompt!r}")
    print("─" * 50)
    sys.stdout.write(prompt)
    sys.stdout.flush()

    states = None
    all_ids = prompt_ids[:]

    # Process prompt (build states)
    for pid in prompt_ids:
        x = torch.tensor([[pid]], device=device)
        logits, states = model(x, states=states)

    for step in range(n_steps):
        x = torch.tensor([[all_ids[-1]]], device=device)
        logits, states = model(x, states=states)

        next_logits = logits[0, 0, :]

        if temperature > 0:
            next_logits = next_logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

        if temperature == 0:
            next_id = int(torch.argmax(next_logits))
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

        all_ids.append(next_id)

        byte_buf.append(next_id - 2)
        try:
            text = byte_buf.decode("utf-8")
            sys.stdout.write(text)
            sys.stdout.flush()
            byte_buf = bytearray()
        except UnicodeDecodeError:
            pass

    if byte_buf:
        sys.stdout.write(byte_buf.decode("utf-8", errors="replace"))
        sys.stdout.flush()

    print("\n" + "─" * 50)
    return detokenize(all_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="In the beginning")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--layers", type=str, default="0")
    parser.add_argument("--checkpoint", type=str, default="layer_0_trained.pt")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    layers = [int(x) for x in args.layers.split(",")]

    model = G1GBlackGooseNF4(layers_to_replace=layers, device="cuda")

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
        for lid in layers:
            sd = model.trainable_channels[str(lid)].state_dict()
            to_load = {k: v for k, v in ckpt.items() if k in sd}
            model.trainable_channels[str(lid)].load_state_dict(to_load, strict=False)
            n_loaded = len(to_load)
            print(f"Loaded {n_loaded}/{len(sd)} keys for layer {lid}")

    output = generate(model, args.prompt, args.steps,
                      temperature=args.temp, top_p=args.top_p)


if __name__ == "__main__":
    main()
