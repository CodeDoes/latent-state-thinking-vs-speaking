"""Adaptive exit entropy sweep.

Ready-to-go for theories/adaptive-exit-entropy.md

Usage:
  python -m src.train_adaptive_entropy --entropy_weight 0.0 --exp_id adapt_ent_0.0_001
  python -m src.train_adaptive_entropy --entropy_weight 0.01 --exp_id adapt_ent_0.01_001
  python -m src.train_adaptive_entropy --entropy_weight 0.1 --exp_id adapt_ent_0.1_001
  # run full sweep
  python -m src.train_adaptive_entropy --sweep --exp_id_prefix adapt_ent_sweep
"""

import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from threads.adaptive_compute.adaptive_loop_model import AdaptiveLoopModel
from domains.byte.byte_vocab import PAD_ID, BYTE_TO_ID, UNK_ID, ID_TO_BYTE

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

def train_one(exp_id, entropy_weight=0.01, steps=2000, batch_size=8, max_len=128,
              lr=3e-4, dim=64, patch_size=4, enc_layers=2, core_layers=2, dec_layers=2,
              enc_max_loops=3, core_depth_loops=2, dec_max_loops=3,
              log_every=100, text_path="threads/g1g_frontend/experiments/byte_ts_001/text.txt"):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream=load_text(Path(text_path))
    model=AdaptiveLoopModel(dim=dim, patch_size=patch_size,
                            enc_layers=enc_layers, core_layers=core_layers, dec_layers=dec_layers,
                            enc_max_loops=enc_max_loops, core_depth_loops=core_depth_loops, dec_max_loops=dec_max_loops).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
    batch_iter=make_batches(stream, max_len, batch_size)

    exp_dir=Path("experiments")/exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg=dict(exp_id=exp_id, entropy_weight=entropy_weight, steps=steps, dim=dim, hypothesis="entropy weight controls loop collapse")
    (exp_dir/"config.json").write_text(json.dumps(cfg,indent=2))

    metrics=[]
    t0=time.time()
    model.train()
    best=float("inf")

    # for gate variance analysis
    from threads.g1g_frontend.surprise_patcher import surprise_per_step

    for step in range(steps):
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
        loss=recon + entropy_weight*ent_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%log_every==0 or step==steps-1:
            elapsed=time.time()-t0
            enc_l=info.get("enc_loops",1)
            # collect lambda stats
            dec_lams=info["decoder"]["exit_lambdas"]
            core_lams=info["core"]["exit_lambdas"]
            def lam_stats(lams):
                if not lams: return {"mean":0,"var":0}
                stacked=torch.stack(lams)  # [R,B,T]
                return {"mean":float(stacked.mean().item()), "var":float(stacked.var().item()), "per_depth":[float(l.mean().item()) for l in lams]}
            dec_stats=lam_stats(dec_lams)
            core_stats=lam_stats(core_lams)

            # surprise correlation (approx): encoder output surprise vs dec loops? Use enc_loops as proxy
            # For deeper analysis, compute correlation between surprise_per_step(enc_out) and lambda
            # Here just log

            print(f"step {step:4d} loss {recon.item():.4f} ent {ent_loss.item():.4f} enc {enc_l} core {info['core']['depth_loop_count']} dec {info['decoder']['depth_loop_count']} lam_dec_mean {dec_stats['mean']:.3f} var {dec_stats['var']:.4f} ew {entropy_weight}")

            metrics.append({"step":step,"loss":recon.item(),"ent_loss":ent_loss.item(),
                            "enc_loops":enc_l if isinstance(enc_l,(int,float)) else 1,
                            "core_loops":info["core"]["depth_loop_count"],
                            "dec_loops":info["decoder"]["depth_loop_count"],
                            "dec_lambda_mean":dec_stats["mean"],
                            "dec_lambda_var":dec_stats["var"],
                            "dec_lambda_per_depth":dec_stats["per_depth"],
                            "core_lambda_mean":core_stats["mean"],
                            "core_lambda_var":core_stats["var"],
                            "entropy_weight":entropy_weight})
            if recon.item()<best: best=recon.item()
            (exp_dir/"metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))

    (exp_dir/"metrics.json").write_text(json.dumps({"final_loss":recon.item(),"best_loss":best,"exp_id":exp_id,"entropy_weight":entropy_weight},indent=2))
    print(f"Done {exp_id} best {best:.4f}")
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="adapt_ent_0.01_001")
    ap.add_argument("--exp_id_prefix", default="adapt_ent")
    ap.add_argument("--entropy_weight", type=float, default=0.01)
    ap.add_argument("--sweep", action="store_true", help="run sweep 0.0,0.001,0.01,0.05,0.1")
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
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--text_path", type=str, default="threads/g1g_frontend/experiments/byte_ts_001/text.txt")
    args=ap.parse_args()

    if args.sweep:
        weights=[0.0,0.001,0.01,0.05,0.1]
        results={}
        for w in weights:
            exp_id=f"{args.exp_id_prefix}_{str(w).replace('.','_')}_001"
            print(f"\n=== Running entropy weight {w} -> {exp_id} ===")
            best=train_one(exp_id=exp_id, entropy_weight=w, steps=args.steps, batch_size=args.batch_size,
                           max_len=args.max_len, lr=args.lr, dim=args.dim, patch_size=args.patch_size,
                           enc_layers=args.enc_layers, core_layers=args.core_layers, dec_layers=args.dec_layers,
                           enc_max_loops=args.enc_max_loops, core_depth_loops=args.core_depth_loops,
                           dec_max_loops=args.dec_max_loops, log_every=args.log_every, text_path=args.text_path)
            results[str(w)]=best
        sweep_path=Path("experiments")/f"{args.exp_id_prefix}_sweep.json"
        sweep_path.write_text(json.dumps(results,indent=2))
        print(f"\nSweep done: {results} -> {sweep_path}")
    else:
        train_one(exp_id=args.exp_id, entropy_weight=args.entropy_weight, steps=args.steps, batch_size=args.batch_size,
                  max_len=args.max_len, lr=args.lr, dim=args.dim, patch_size=args.patch_size,
                  enc_layers=args.enc_layers, core_layers=args.core_layers, dec_layers=args.dec_layers,
                  enc_max_loops=args.enc_max_loops, core_depth_loops=args.core_depth_loops,
                  dec_max_loops=args.dec_max_loops, log_every=args.log_every, text_path=args.text_path)

if __name__=="__main__":
    main()
