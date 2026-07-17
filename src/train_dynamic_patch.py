"""Dynamic vs fixed patch ablation.

Ready-to-go for theories/dynamic-patch-vs-fixed.md

Usage:
  python -m src.train_dynamic_patch --patch_mode fixed --exp_id dyn_patch_fixed_001 --steps 2000
  python -m src.train_dynamic_patch --patch_mode dynamic --threshold 0.7 --exp_id dyn_patch_dynamic_07_001
  python -m src.train_dynamic_patch --patch_mode dynamic --threshold 0.5 --exp_id dyn_patch_dynamic_05_001
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F

from src.adaptive_loop_model import AdaptiveLoopModel
from src.byte_vocab import PAD_ID, BYTE_TO_ID, UNK_ID, ID_TO_BYTE

def load_text(p: Path):
    txt=p.read_bytes().decode("utf-8", errors="replace")
    return [BYTE_TO_ID.get(ord(c), UNK_ID) for c in txt]

def make_batches(stream, max_len, batch_size):
    n=(len(stream)-1)//max_len*max_len
    stream=stream[:n+1]
    while True:
        for start in range(0, len(stream)-max_len-1, batch_size*max_len):
            rows=[]
            for i in range(batch_size):
                chunk=stream[start+i*max_len:start+(i+1)*max_len+1]
                if len(chunk)<max_len+1: continue
                rows.append(chunk)
            if not rows: continue
            for r in rows:
                while len(r)<max_len+1: r.append(PAD_ID)
            batch=torch.tensor(rows, dtype=torch.long)
            yield batch[:,:-1], batch[:,1:]

def train(exp_id="dyn_patch_fixed_001", patch_mode="fixed", threshold=0.7, min_patch=2, max_patch=8,
          steps=2000, batch_size=8, max_len=128, lr=3e-4, dim=64, patch_size=4,
          enc_layers=2, core_layers=2, dec_layers=2,
          enc_max_loops=3, core_depth_loops=2, dec_max_loops=3,
          entropy_weight=0.01, log_every=100,
          text_path="experiments/byte_ts_001/text.txt",
          warmup_steps=0):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream=load_text(Path(text_path))
    print(f"{exp_id} mode={patch_mode} thr={threshold} device={device} bytes={len(stream):,}")

    dynamic = (patch_mode=="dynamic")

    model=AdaptiveLoopModel(
        dim=dim, patch_size=patch_size,
        enc_layers=enc_layers, core_layers=core_layers, dec_layers=dec_layers,
        enc_max_loops=enc_max_loops, core_depth_loops=core_depth_loops, dec_max_loops=dec_max_loops,
        dynamic_patch=dynamic if warmup_steps==0 else False,  # start fixed if warmup
        patch_threshold=threshold, min_patch=min_patch, max_patch=max_patch
    ).to(device)

    optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
    batch_iter=make_batches(stream, max_len, batch_size)

    exp_dir=Path("experiments")/exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg=dict(exp_id=exp_id, patch_mode=patch_mode, threshold=threshold, min_patch=min_patch, max_patch=max_patch,
             steps=steps, dim=dim, patch_size=patch_size, dynamic_patch=dynamic, warmup_steps=warmup_steps,
             hypothesis="dynamic surprise patch vs fixed")
    (exp_dir/"config.json").write_text(json.dumps(cfg,indent=2))

    metrics=[]
    model.train()
    t0=time.time()
    best=float("inf")

    for step in range(steps):
        # warmup switch
        if warmup_steps>0 and step==warmup_steps:
            print(f"Switching to dynamic patch at step {step}")
            model.dynamic_patch=True
            model.patch_threshold=threshold

        input_ids, targets = next(batch_iter)
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits, info = model(input_ids)
        recon=F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID)

        ent_loss=torch.tensor(0.0, device=device)
        for lam_list in [info["core"]["exit_lambdas"], info["decoder"]["exit_lambdas"]]:
            if lam_list:
                lams=torch.stack(lam_list)
                cum=torch.ones_like(lams[0])
                ent=torch.zeros_like(lams[0])
                for r in range(lams.shape[0]):
                    pi=lams[r]*cum
                    ent=ent-pi*(pi+1e-7).log()
                    cum=cum*(1-lams[r])
                ent_loss=ent_loss-ent.mean()
        loss=recon+entropy_weight*ent_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%log_every==0 or step==steps-1:
            elapsed=time.time()-t0
            # patch stats
            pl=info.get("patch_lengths")
            if pl is not None and hasattr(pl,"shape"):
                # dynamic: histogram
                # pl is [B, P] with zero padding
                non_zero=pl[pl>0]
                if non_zero.numel()>0:
                    mean_pl=float(non_zero.float().mean().item())
                    std_pl=float(non_zero.float().std().item()) if non_zero.numel()>1 else 0.0
                    rho=float(max_len / max(info.get("n_latents",1),1))
                else:
                    mean_pl=0; std_pl=0; rho=0
            else:
                mean_pl=patch_size
                std_pl=0
                rho=max_len/max(info.get("n_latents",1),1)

            print(f"step {step:4d} loss {recon.item():.4f} rho {rho:.1f} mean_pl {mean_pl:.1f} std {std_pl:.1f} lat {info.get('n_latents',0)}")

            metrics.append({"step":step,"loss":recon.item(),"rho":rho,"mean_patch_len":mean_pl,"std_patch_len":std_pl,
                            "latents":info.get("n_latents",0),
                            "enc_loops":info.get("enc_loops",1),
                            "core_loops":info["core"]["depth_loop_count"],
                            "dec_loops":info["decoder"]["depth_loop_count"]})
            if recon.item()<best: best=recon.item()
            (exp_dir/"metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))

            if step%(log_every*5)==0:
                with torch.no_grad():
                    model.eval()
                    inp,_=next(batch_iter)
                    inp=inp[:1].to(device)
                    log,_=model(inp)
                    pred=torch.argmax(log,dim=-1)[0].tolist()
                    dec="".join(chr(ID_TO_BYTE.get(i,63)) for i in pred[:120])
                    (exp_dir/"sample.txt").write_text(dec)
                    model.train()

    (exp_dir/"metrics.json").write_text(json.dumps({"final_loss":recon.item(),"best_loss":best,"exp_id":exp_id},indent=2))
    print(f"Done {exp_id} best {best:.4f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="dyn_patch_fixed_001")
    ap.add_argument("--patch_mode", default="fixed", choices=["fixed","dynamic"])
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--min_patch", type=int, default=2)
    ap.add_argument("--max_patch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--core_layers", type=int, default=2)
    ap.add_argument("--dec_layers", type=int, default=2)
    ap.add_argument("--enc_max_loops", type=int, default=3)
    ap.add_argument("--core_depth_loops", type=int, default=2)
    ap.add_argument("--dec_max_loops", type=int, default=3)
    ap.add_argument("--entropy_weight", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--text_path", type=str, default="experiments/byte_ts_001/text.txt")
    ap.add_argument("--warmup_steps", type=int, default=0)
    args=ap.parse_args()
    train(**vars(args))
