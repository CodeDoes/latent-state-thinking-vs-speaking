"""Train byte encoder to match layer 0's time-mix output (tm_out).

Data: experiments/byte_time_mix/training_data/story_*.pt
Each sample: bytes (24 byte IDs) -> target = tm_out (2560-dim) from real g1g layer 0.

Architecture (see src/byte_time_mix_model.py):
  raw uint8[24] -> scalar-proj -> minGRU -> (state, mask)
  - state (B,2560): tm_out, a MASK-GATED mean-pool of minGRU hiddens.
  - mask  (B,24):  per-position validity logit, estimated FIRST, then used
                   to gate the state. find_split() recovers the token boundary.

The confidence mask is computed before the state estimate, so padding bytes
cannot corrupt the tm_out prediction. See byte_time_mix_model for details.
"""
import torch, torch.nn.functional as F, time
from pathlib import Path
import sys; sys.path.insert(0, '.')
from threads.g1g_frontend.byte_time_mix_model import ByteTimeMixEncoder, find_split

torch.set_float32_matmul_precision('high')

DEVICE = "cuda"
DIM = 256
LR = 3e-4
STEPS = 20000
BATCH = 256
MAX_BYTES = 24
SAVE_PATH = "experiments/byte_time_mix/encoder.pt"
DATA_DIR = Path("experiments/byte_time_mix/training_data")

# NOTE: gen_time_mix_data.py MAX_STORIES is capped at 50 for quick runs; raise
# for full training (each story -> ~64MB .pt, watch disk).
device = torch.device(DEVICE)


def main():
    # ── Load data ──
    files = sorted(DATA_DIR.glob("story_*.pt"))
    print(f"Found {len(files)} data files", flush=True)

    all_bytes, all_targets, all_nbytes = [], [], []
    for f in files:
        try:
            chunk = torch.load(f, map_location='cpu', weights_only=True)
            for s in chunk:
                all_bytes.append(s['bytes'])
                all_targets.append(s['tm_out'].float())
                all_nbytes.append(s['num_bytes'])
        except Exception as e:
            print(f"  Skipping {f.name}: {e}", flush=True)

    if len(all_bytes) == 0:
        print("No training data!", flush=True)
        sys.exit(1)

    bytes_t = torch.tensor(all_bytes, dtype=torch.long)
    targets_t = torch.stack(all_targets)
    nbytes_t = torch.tensor(all_nbytes, dtype=torch.long)
    MAX_BYTES = bytes_t.shape[1]  # 24
    print(f"Data: {len(all_bytes)} samples", flush=True)
    print(f"  bytes: {bytes_t.shape}, tm_out: {targets_t.shape}", flush=True)
    print(f"  tm_out mean={targets_t.mean():.3f} std={targets_t.std():.3f}", flush=True)

    # ── Model ──
    enc = ByteTimeMixEncoder().to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in enc.parameters()):,}", flush=True)

    # ── Train ──
    MASK_THRESHOLD = 0.5
    MASK_LAMBDA = 0.1
    t0 = time.time()
    for step in range(STEPS):
        idx = torch.randperm(len(all_bytes), device='cpu')[:BATCH]
        inp = bytes_t[idx].to(device)
        target = targets_t[idx].to(device)

        state, mask = enc(inp)                       # mask: (B, 24) per-position logit
        state_loss = F.mse_loss(state, target)

        # mask target: validity gate — 1 for real bytes (pos < num_bytes), 0 after.
        nb = nbytes_t[idx].to(device)                                  # (B,)
        pos_idx = torch.arange(MAX_BYTES, device=device).unsqueeze(0)  # (1, T)
        mask_tgt = (pos_idx < nb.unsqueeze(1)).float()                 # (B, 24) 1=real
        mask_loss = F.binary_cross_entropy_with_logits(mask, mask_tgt)

        loss = state_loss + MASK_LAMBDA * mask_loss

        opt.zero_grad(); loss.backward(); opt.step()

        if (step+1) % 1000 == 0:
            sps = (step+1) / (time.time() - t0)
            cos = F.cosine_similarity(state, target).mean().item()
            split = find_split(mask, MASK_THRESHOLD)
            acc = (split == nb).float().mean().item()
            print(f"step {step+1:5d}  loss={loss.item():.4f}  state={state_loss.item():.4f}  "
                  f"mask={mask_loss.item():.4f}  cos={cos:.4f}  split_acc={acc:.3f}  {sps:.1f} st/s", flush=True)
        if (step+1) % 5000 == 0:
            Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(enc.state_dict(), Path(SAVE_PATH))

    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(enc.state_dict(), Path(SAVE_PATH))
    print(f"\nSaved to {SAVE_PATH}", flush=True)
    print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
