"""Train B3D-RWKV nano – triplet-block diffusion at small scale.

Ready-to-go for theories/b3d-rwkv-nano.md

Compares:
- AR baseline (causal next-token)
- Triplet diffusion (b1 masked, b2 lossable, b3 clean)

Usage:
  python -m src.train_b3d_rwkv --mode triplet --exp_id b3d_triplet_001 --steps 2000 --dim 128 --block_size 8
  python -m src.train_b3d_rwkv --mode ar --exp_id b3d_ar_001 --steps 2000 --dim 128
  python -m src.train_b3d_rwkv --mode no_b3 --exp_id b3d_no_b3_001 --steps 2000

Also supports diffusion inference eval with tau sweep.
"""

import argparse, json, time, random
from pathlib import Path
import torch
import torch.nn.functional as F

from threads.b3d.b3d_rwkv_model import B3DRWKVModel, build_triplet_batch, count_params

# Vocab: use char vocab from train_rwkv.py (70 tokens) or byte vocab 258
# We'll support both via --vocab_mode

CHARS = [
    '\n', ' ', '!', ',', '-', '.', ':', '=',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
]
SPECIAL = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']
VOCAB = SPECIAL + CHARS
char_to_id = {c: i for i, c in enumerate(VOCAB)}
id_to_char = {i: c for c, i in char_to_id.items()}
PAD_ID = char_to_id['<PAD>']
UNK_ID = char_to_id['<UNK>']
MASK_ID = char_to_id['<MASK>']

def encode(text: str):
    return [char_to_id.get(c, UNK_ID) for c in text]

def decode(ids):
    return ''.join(id_to_char.get(i, '<UNK>') for i in ids)

# Byte vocab option
from domains.byte.byte_vocab import BYTE_TO_ID as BYTE_MAP, PAD_ID as BYTE_PAD, UNK_ID as BYTE_UNK
BYTE_VOCAB_SIZE = 258
BYTE_MASK_ID = 257  # use last as mask (257)

def load_text_byte(path: Path):
    txt = path.read_bytes().decode("utf-8", errors="replace")
    return [BYTE_MAP.get(ord(c), BYTE_UNK) for c in txt]

def make_batch_char(generator, batch_size, max_len):
    # generator is LogicNiiahGenerator for char task, or simple random for byte_ts
    # For simplicity if generator is None, we use random text from byte_ts
    # Here we handle both modes via closure in train()
    raise NotImplementedError

def train(exp_id="b3d_triplet_001", mode="triplet", steps=2000, batch_size=8, max_len=128,
          dim=128, layers=3, block_size=8, mask_ratio=0.3, tau=0.9,
          lr=3e-4, vocab_mode="char", text_path="threads/g1g_frontend/experiments/byte_ts_001/text.txt",
          log_every=100, eval_every=500, device="cpu"):

    dev = torch.device(device)
    if torch.cuda.is_available() and device != "cpu":
        dev = torch.device("cuda")

    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    if vocab_mode == "char":
        vocab_size = len(VOCAB)
        pad_id = PAD_ID
        mask_id = MASK_ID
        # for char mode we need logic_niiah generator
        from threads.memory_growth.logic_niiah_generator import LogicNiiahGenerator
        generator = LogicNiiahGenerator(seed=42)
        gen_kwargs = dict(num_vars=3, min_transforms=1, max_transforms=3, noise_min=1, noise_max=3)
        # we will generate text batches
        def get_batch():
            texts = []
            for _ in range(batch_size):
                sample = generator.generate(**gen_kwargs)
                enc = encode(sample['text'])[:max_len]
                # pad
                while len(enc) < max_len:
                    enc.append(pad_id)
                texts.append(enc[:max_len])
            return torch.tensor(texts, dtype=torch.long)
    else:
        vocab_size = BYTE_VOCAB_SIZE
        pad_id = BYTE_PAD
        mask_id = BYTE_MASK_ID
        stream = load_text_byte(Path(text_path))
        print(f"Loaded {len(stream):,} bytes for byte mode")
        # make batches from stream
        def make_batches():
            n = (len(stream)-1)//max_len*max_len
            s = stream[:n+1]
            while True:
                for start in range(0, len(s)-max_len-1, batch_size*max_len):
                    rows=[]
                    for i in range(batch_size):
                        chunk=s[start+i*max_len:start+(i+1)*max_len]
                        if len(chunk)<max_len: continue
                        rows.append(chunk)
                    if not rows: continue
                    yield torch.tensor(rows, dtype=torch.long)
        batch_iter = make_batches()
        def get_batch():
            return next(batch_iter)

    model = B3DRWKVModel(vocab_size=vocab_size, dim=dim, num_layers=layers, pad_token_id=pad_id, mask_token_id=mask_id).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    config = dict(exp_id=exp_id, mode=mode, steps=steps, batch_size=batch_size, max_len=max_len,
                  dim=dim, layers=layers, block_size=block_size, mask_ratio=mask_ratio, tau=tau,
                  vocab_mode=vocab_mode, lr=lr, hypothesis="triplet-block diffusion vs AR at nano")
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Model {mode} params {count_params(model):,} vocab {vocab_size} mask {mask_id} device {dev}")

    best_loss = float("inf")
    metrics_log = []

    t0 = time.time()
    for step in range(1, steps+1):
        clean = get_batch().to(dev)  # [B, L]
        B_, L = clean.shape
        # ensure L divisible by block_size
        trunc = (L // block_size) * block_size
        clean = clean[:, :trunc]
        L = trunc

        if mode == "ar":
            # causal AR: input = clean[:,:-1], target = clean[:,1:]
            inp = clean[:, :-1]
            tgt = clean[:, 1:]
            logits = model(inp)
            # logits [B, L-1, vocab]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=pad_id)
            loss_mask_sum = (tgt != pad_id).sum().item()
        elif mode == "triplet":
            physical, loss_mask, targets, meta = build_triplet_batch(clean, block_size, mask_ratio, mask_id)
            physical = physical.to(dev)
            targets = targets.to(dev)
            loss_mask = loss_mask.to(dev)
            logits = model(physical)  # [B, 3L, vocab]
            # loss only where loss_mask true
            # flatten masked positions
            logits_masked = logits[loss_mask]  # [N, vocab]
            targets_masked = targets[loss_mask]  # [N]
            if logits_masked.numel() == 0:
                loss = torch.tensor(0.0, device=dev)
            else:
                loss = F.cross_entropy(logits_masked, targets_masked)
            loss_mask_sum = int(loss_mask.sum().item())
        elif mode == "no_b3":
            # ablate b3: only b1+b2, no clean refresh
            # build without b3: physical = b1+b2 per block = 2L
            # reuse build_triplet_batch but drop b3 part
            physical_full, loss_mask_full, targets_full, meta = build_triplet_batch(clean, block_size, mask_ratio, mask_id)
            # physical_full is [B, 3L] with pattern b1,b2,b3 per block
            # we need to extract only b1,b2
            B_phys, T_phys = physical_full.shape
            # reshape per block: each block 3*block_size, we want first 2*block_size
            num_blocks = L // block_size
            # rewrite: iterate blocks
            new_physical = []
            new_loss_mask = []
            new_targets = []
            for blk in range(num_blocks):
                start = blk * 3 * block_size
                b1_b2 = physical_full[:, start:start+2*block_size]
                lm_b1_b2 = loss_mask_full[:, start:start+2*block_size]
                tgt_b1_b2 = targets_full[:, start:start+2*block_size]
                new_physical.append(b1_b2)
                new_loss_mask.append(lm_b1_b2)
                new_targets.append(tgt_b1_b2)
            physical = torch.cat(new_physical, dim=1).to(dev)
            loss_mask = torch.cat(new_loss_mask, dim=1).to(dev)
            targets = torch.cat(new_targets, dim=1).to(dev)
            logits = model(physical)
            logits_masked = logits[loss_mask]
            targets_masked = targets[loss_mask]
            loss = F.cross_entropy(logits_masked, targets_masked) if logits_masked.numel()>0 else torch.tensor(0.0, device=dev)
            loss_mask_sum = int(loss_mask.sum().item())
        else:
            raise ValueError(mode)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:4d}/{steps} mode {mode} loss {loss.item():.4f} masked {loss_mask_sum} elapsed {elapsed:.0f}s")

            # eval diffusion decode throughput for triplet mode
            extra = {}
            if mode == "triplet" and step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # take one batch context and decode one block
                    ctx = clean[:1, :block_size*2]  # 2 blocks as context
                    decoded, iters, info = model.diffusion_decode_block(ctx, block_size=block_size, mask_id=mask_id, tau=tau, max_iters=10, device=dev)
                    avg_committed = sum(h["committed"] for h in info["commit_history"]) / max(len(info["commit_history"]),1)
                    print(f"  diffusion decode: iters {iters} avg committed {avg_committed:.1f} history {info['commit_history']}")
                    extra = {"decode_iters": iters, "commit_history": info["commit_history"]}
                model.train()

            metrics_log.append({"step": step, "loss": loss.item(), "mode": mode, **extra})
            (exp_dir / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics_log))
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), exp_dir / "best.pt")

    # final eval
    result = {"final_loss": loss.item(), "best_loss": best_loss, "exp_id": exp_id, "mode": mode, "params": count_params(model)}
    (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"Done {exp_id} best {best_loss:.4f}")

    # tau sweep for throughput if triplet
    if mode == "triplet":
        print("\n=== Tau sweep for throughput ===")
        model.eval()
        sweep_results = {}
        for tau_test in [0.8, 0.9, 0.95]:
            with torch.no_grad():
                clean_test = get_batch().to(dev)[:1, :block_size*4]
                ctx = clean_test[:, :block_size*2]
                # decode next block
                decoded, iters, info = model.diffusion_decode_block(ctx, block_size=block_size, mask_id=mask_id, tau=tau_test, max_iters=20, device=dev)
                sweep_results[str(tau_test)] = {"iters": iters, "commit_history": info["commit_history"]}
                print(f"tau {tau_test}: iters {iters}")
        (exp_dir / "tau_sweep.json").write_text(json.dumps(sweep_results, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="b3d_triplet_001")
    ap.add_argument("--mode", default="triplet", choices=["triplet", "ar", "no_b3"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--block_size", type=int, default=8)
    ap.add_argument("--mask_ratio", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vocab_mode", default="char", choices=["char", "byte"])
    ap.add_argument("--text_path", type=str, default="threads/g1g_frontend/experiments/byte_ts_001/text.txt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(**vars(args))
