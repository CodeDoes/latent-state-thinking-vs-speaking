#!/usr/bin/env python3
"""
Generate training data for BlackGoose channel-mix.
Runs the frozen g1g on text, saves (ln2, ffn_output) pairs for specified layers.

Usage:
    python src/gen_blackgoose_data.py --layers 0,5,10 --text <file> --n 5000


[meta]
status: active
verdict: supports
[/meta]
"""

import sys, os, time, json, argparse
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_bytes(text: str, seqlen: int, max_examples: int):
    raw = text.encode("utf-8")
    raw = bytes(b for b in raw if b >= 2)
    xs = []
    stride = max(1, (len(raw) - seqlen) // max_examples)
    for i in range(0, len(raw) - seqlen, stride):
        xs.append(torch.tensor(list(raw[i:i+seqlen]), dtype=torch.long))
        if len(xs) >= max_examples:
            break
    return torch.stack(xs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="0",
                        help="Comma-sep layer indices to collect data for")
    parser.add_argument("--text", type=str, default=None,
                        help="Text file (default: embedded sample)")
    parser.add_argument("--n", type=int, default=5000,
                        help="Number of samples per layer")
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--out", type=str, default="experiments/blackgoose_data",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────
    # We need the frozen backbone to generate FFN outputs.
    # Accept the slow load once, then discard.
    from domains.g1g.g1g_blackgoose_nf4 import G1GBlackGooseNF4
    model = G1GBlackGooseNF4(layers_to_replace=[], device=device)
    model.eval()
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── Text data ─────────────────────────────────────────────────────
    if args.text:
        with open(args.text) as f:
            text = f.read()
    else:
        text = (
            "Alice was beginning to get very tired of sitting by her sister on the bank, "
            "and of having nothing to do. once or twice she had peeped into the book her "
            "sister was reading, but it had no pictures or conversations in it, "
            "and where is the use of a book, thought Alice without pictures or conversation? "
            "So she was considering in her own mind (as well as she could, for the hot day "
            "made her feel very sleepy and stupid), whether the pleasure of making a "
            "daisy-chain would be worth the trouble of getting up and picking the daisies, "
            "when suddenly a White Rabbit with pink eyes ran close by her."
        ) * 10  # repeat to get more data

    ids = make_bytes(text, args.seqlen, args.n * 2)  # generate extra
    print(f"Input sequences: {ids.shape[0]}")

    # ── Generate targets ──────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16 = model._sd._bf16
    n_collected = {lid: 0 for lid in layers}
    n_needed = {lid: args.n for lid in layers}

    examples = {lid: {"inputs": [], "targets": []} for lid in layers}
    t0 = time.time()

    with torch.no_grad():
        for batch_idx in range(0, ids.shape[0], 1):
            batch = ids[batch_idx:batch_idx+1].to(device)
            B, T = batch.shape

            # Forward through the model, capturing each layer's LN2 and FFN output
            h = F.embedding(batch, model._embed_w)

            # Create states
            states = []
            for i in range(model.n_layers):
                states.append({
                    "xx": torch.zeros(model.dim, device=device, dtype=model.dtype),
                    "xx_c": torch.zeros(model.dim, device=device, dtype=model.dtype),
                    "mat": torch.zeros(model.n_heads, model.head_size, model.head_size,
                                       device=device, dtype=torch.float32),
                    "v_first": None,
                })

            # Process token by token
            for t in range(T):
                bt = h[:, t]  # (B, D)
                for i in range(model.n_layers):
                    s = states[i]

                    ln1 = F.layer_norm(bt, (model.dim,),
                                       weight=bf16[f"blocks.{i}.ln1.weight"],
                                       bias=bf16[f"blocks.{i}.ln1.bias"])

                    w = model._sd.dequantize_layer(i)
                    tm_h = model._time_mix(i, ln1, s, w)

                    ln2 = F.layer_norm(bt + tm_h, (model.dim,),
                                       weight=bf16[f"blocks.{i}.ln2.weight"],
                                       bias=bf16[f"blocks.{i}.ln2.bias"])

                    if i in layers and n_collected[i] < n_needed[i]:
                        # Capture LN2 input and FFN target
                        ffn = f"blocks.{i}.ffn."
                        xx_c = s.get("xx_c", torch.zeros_like(bt))
                        xk = ln2 + (xx_c - ln2) * bf16[ffn + "x_k"].squeeze()
                        k_c = F.relu(xk @ w[ffn + "key.weight"].T) ** 2
                        ffn_out = k_c @ w[ffn + "value.weight"].T

                        for b in range(B):
                            if n_collected[i] < n_needed[i]:
                                examples[i]["inputs"].append(ln2[b].cpu())
                                examples[i]["targets"].append(ffn_out[b].cpu())
                                n_collected[i] += 1

                    # FFN (original)
                    if str(i) not in model.trainable_channels:
                        ffn = f"blocks.{i}.ffn."
                        xx_c = s.get("xx_c", torch.zeros_like(bt))
                        xk = ln2 + (xx_c - ln2) * bf16[ffn + "x_k"].squeeze()
                        k_c = F.relu(xk @ w[ffn + "key.weight"].T) ** 2
                        cm_out = k_c @ w[ffn + "value.weight"].T
                    else:
                        cm_out, _ = model.trainable_channels[str(i)](ln2)

                    bt = bt + tm_h + cm_out
                    del w

                    s["xx"] = ln1.detach().clone()
                    s["xx_c"] = ln2.detach().clone()

                h[:, t] = bt

            # Check progress
            counts = ", ".join(f"L{l}={n_collected[l]}" for l in layers)
            if batch_idx % 8 == 0:
                elapsed = time.time() - t0
                print(f"  batch {batch_idx:4d}/{ids.shape[0]}  [{counts}]  {elapsed:.0f}s")

            # Save intermediate if any layer is full
            all_done = all(n_collected[l] >= n_needed[l] for l in layers)
            if all_done or (batch_idx > 0 and batch_idx % 32 == 0):
                for l in layers:
                    if len(examples[l]["inputs"]) > 0:
                        inp = torch.stack(examples[l]["inputs"])
                        tgt = torch.stack(examples[l]["targets"])
                        path = out_dir / f"layer_{l}.pt"
                        torch.save({"inputs": inp, "targets": tgt}, path)
                        print(f"  Saved {path}: {inp.shape[0]} samples")
                if all_done:
                    break

    # Final save
    for l in layers:
        if len(examples[l]["inputs"]) > 0:
            inp = torch.stack(examples[l]["inputs"])
            tgt = torch.stack(examples[l]["targets"])
            path = out_dir / f"layer_{l}.pt"
            torch.save({"inputs": inp, "targets": tgt}, path)
            print(f"Final {path}: {inp.shape[0]} samples")

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
