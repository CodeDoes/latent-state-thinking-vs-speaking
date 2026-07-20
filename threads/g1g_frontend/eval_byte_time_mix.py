"""Evaluate the byte -> (state, mask) time-mix encoder.

Loads experiments/byte_time_mix/encoder.pt and reports:
  - state cosine similarity vs real layer-0 tm_out (full dataset)
  - split accuracy (find_split vs num_bytes)
  - qualitative examples: token text -> predicted tm_out cos
"""
import torch, torch.nn.functional as F
from pathlib import Path
import sys; sys.path.insert(0, '.')
from minGRU_pytorch import minGRU
from threads.g1g_frontend.byte_time_mix_model import ByteTimeMixEncoder, find_split

DEVICE = torch.device("cuda")
BATCH = 256
SAVE_PATH = "experiments/byte_time_mix/encoder.pt"
DATA_DIR = Path("experiments/byte_time_mix/training_data")


def load_data():
    all_bytes, all_targets, all_nbytes = [], [], []
    for f in sorted(DATA_DIR.glob("story_*.pt")):
        try:
            chunk = torch.load(f, map_location='cpu', weights_only=True)
        except Exception as e:
            print(f"  skip {f.name}: {e}"); continue
        for s in chunk:
            all_bytes.append(s['bytes'])
            all_targets.append(s['tm_out'].float())
            all_nbytes.append(s['num_bytes'])
    return (torch.tensor(all_bytes, dtype=torch.long),
            torch.stack(all_targets),
            torch.tensor(all_nbytes, dtype=torch.long))


def main():
    bytes_t, targets_t, nbytes_t = load_data()
    N = len(bytes_t)
    print(f"Eval data: {N} samples", flush=True)

    enc = ByteTimeMixEncoder().to(DEVICE)
    enc.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    enc.eval()

    all_cos, all_split_ok = [], []
    with torch.no_grad():
        for i in range(0, N, BATCH):
            inp = bytes_t[i:i+BATCH].to(DEVICE)
            tgt = targets_t[i:i+BATCH].to(DEVICE)
            nb = nbytes_t[i:i+BATCH].to(DEVICE)
            state, mask = enc(inp)
            cos = F.cosine_similarity(state, tgt, dim=-1)   # (b,)
            split = find_split(mask)
            all_cos.append(cos)
            all_split_ok.append((split == nb).float())

    cos = torch.cat(all_cos)
    split_ok = torch.cat(all_split_ok)
    print(f"\n=== RESULTS ===")
    print(f"  state cosine vs tm_out:  mean={cos.mean():.4f}  median={cos.median():.4f}  "
          f"p10={cos.quantile(0.1):.4f}  p90={cos.quantile(0.9):.4f}")
    print(f"  split accuracy:          {split_ok.mean():.4f}")
    print(f"  (tm_out std={targets_t.std():.3f}, so MSE floor ~ {targets_t.var():.2f})")

    # Qualitative: show a few tokens and their predicted cos
    print("\n=== examples (first 8 samples) ===")
    with torch.no_grad():
        inp = bytes_t[:8].to(DEVICE)
        tgt = targets_t[:8].to(DEVICE)
        nb = nbytes_t[:8].to(DEVICE)
        state, mask = enc(inp)
        cos = F.cosine_similarity(state, tgt, dim=-1)
        split = find_split(mask)
        for k in range(8):
            ids = bytes_t[k].tolist()
            real = [b - 2 for b in ids[:nb[k].item()]]
            txt = b"".join(bytes([max(0, x)]) for x in real).decode('utf-8', 'replace')
            print(f"  nb={nb[k].item():2d} split={split[k].item():2d} "
                  f"cos={cos[k].item():.3f}  token={txt!r}")


if __name__ == "__main__":
    main()
