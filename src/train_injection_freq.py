"""Train injection-frequency ablation: front vs per-layer fusion.

Ready-to-go experiment for theory theories/injection-frequency.md

Usage:
  python -m src.train_injection_freq --fusion_mode front --exp_id inj_freq_front_001 --steps 2000
  python -m src.train_injection_freq --fusion_mode per_layer --exp_id inj_freq_perlayer_001 --steps 2000
  python -m src.train_injection_freq --fusion_mode per_layer_nogate --exp_id inj_freq_perlayer_nogate_001
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.injection_freq_model import InjectionFreqAdaptiveModel, count_params
from src.byte_vocab import VOCAB_SIZE, PAD_ID, BYTE_TO_ID, UNK_ID


def load_text(path: Path):
    text = path.read_bytes().decode("utf-8", errors="replace")
    return [BYTE_TO_ID.get(ord(c), UNK_ID) for c in text]

def make_batches(stream, max_len, batch_size):
    n = (len(stream)-1)//max_len*max_len
    stream = stream[:n+1]
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
            batch=torch.tensor(rows,dtype=torch.long)
            yield batch[:,:-1], batch[:,1:]

def train(exp_id="inj_freq_front_001", fusion_mode="front", steps=2000, batch_size=8, max_len=128,
          lr=3e-4, dim=64, patch_size=4, enc_layers=2, core_layers=2, dec_layers=2,
          enc_max_loops=3, core_depth_loops=2, dec_max_loops=3,
          entropy_weight=0.01, log_every=100,
          text_path="experiments/byte_ts_001/text.txt",
          dynamic_patch=False, patch_threshold=0.7):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream=load_text(Path(text_path))
    print(f"Device {device}, loaded {len(stream):,} bytes, exp {exp_id}, mode {fusion_mode}")

    model=InjectionFreqAdaptiveModel(
        dim=dim, patch_size=patch_size,
        enc_layers=enc_layers, core_layers=core_layers, dec_layers=dec_layers,
        enc_max_loops=enc_max_loops, core_depth_loops=core_depth_loops, dec_max_loops=dec_max_loops,
        fusion_mode=fusion_mode, dynamic_patch=dynamic_patch, patch_threshold=patch_threshold
    ).to(device)

    n_params=count_params(model)
    print(f"Params {n_params:,} (mode={fusion_mode}) encoder {count_params(model.encoder):,} core {count_params(model.core):,} decoder {count_params(model.decoder):,}")

    optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
    batch_iter=make_batches(stream, max_len, batch_size)

    exp_dir=Path("experiments")/exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    config=dict(exp_id=exp_id, fusion_mode=fusion_mode, steps=steps, batch_size=batch_size, max_len=max_len,
                lr=lr, dim=dim, patch_size=patch_size, enc_layers=enc_layers, core_layers=core_layers,
                dec_layers=dec_layers, enc_max_loops=enc_max_loops, core_depth_loops=core_depth_loops,
                dec_max_loops=dec_max_loops, entropy_weight=entropy_weight, dynamic_patch=dynamic_patch,
                patch_threshold=patch_threshold, params=n_params, hypothesis="per-layer vs front fusion")
    (exp_dir/"config.json").write_text(json.dumps(config,indent=2))

    metrics_log=[]
    model.train()
    t0=time.time()
    best_loss=float("inf")

    for step in range(steps):
        input_ids, targets = next(batch_iter)
        input_ids, targets = input_ids.to(device), targets.to(device)

        logits, info = model(input_ids)
        recon_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID)

        # entropy
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
                ent_loss=ent_loss - ent.mean()
        loss=recon_loss + entropy_weight*ent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every==0 or step==steps-1:
            elapsed=time.time()-t0
            # stats
            try:
                enc_l=info.get("enc_loops",1)
                if isinstance(enc_l, torch.Tensor):
                    enc_l=int(enc_l.mean().item())
            except: enc_l=1
            core_d=info["core"]["depth_loop_count"]
            dec_d=info["decoder"]["depth_loop_count"]
            n_lat=info.get("n_latents",0)
            rho=max_len/max(n_lat,1)
            print(f"step {step:4d} loss {recon_loss.item():.4f} ent {ent_loss.item():.4f} enc {enc_l} core {core_d} dec {dec_d} lat {n_lat} rho {rho:.1f} {elapsed:.1f}s")
            metrics_log.append({"step":step,"loss":recon_loss.item(),"ent_loss":ent_loss.item(),
                                "enc_loops":int(enc_l) if not isinstance(enc_l, list) else enc_l,
                                "core_loops":core_d,"dec_loops":dec_d,"latents":n_lat,"rho":rho})
            if recon_loss.item()<best_loss:
                best_loss=recon_loss.item()
            (exp_dir/"metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics_log))

            # sample
            if step % (log_every*5)==0:
                with torch.no_grad():
                    model.eval()
                    input_ids_s, _ = next(batch_iter)
                    input_ids_s=input_ids_s[:1].to(device)
                    logits_s, _ = model(input_ids_s)
                    pred=torch.argmax(logits_s, dim=-1)[0].tolist()
                    # decode few bytes
                    from src.byte_vocab import ID_TO_BYTE
                    def decode_ids(ids):
                        return "".join(chr(ID_TO_BYTE.get(i,63)) for i in ids[:100])
                    sample_text=decode_ids(pred)
                    (exp_dir/"sample.txt").write_text(f"step {step} pred: {sample_text}\n")
                    model.train()

    result={"final_loss":recon_loss.item(),"best_loss":best_loss,"params":n_params,"fusion_mode":fusion_mode,"exp_id":exp_id,"steps":steps}
    (exp_dir/"metrics.json").write_text(json.dumps(result,indent=2))
    print(f"Done exp {exp_id} final {recon_loss.item():.4f} best {best_loss:.4f} -> {exp_dir}")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--exp_id", default="inj_freq_front_001")
    parser.add_argument("--fusion_mode", default="front", choices=["front","per_layer","per_layer_nogate"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--core_layers", type=int, default=2)
    parser.add_argument("--dec_layers", type=int, default=2)
    parser.add_argument("--enc_max_loops", type=int, default=3)
    parser.add_argument("--core_depth_loops", type=int, default=2)
    parser.add_argument("--dec_max_loops", type=int, default=3)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--dynamic_patch", action="store_true")
    parser.add_argument("--patch_threshold", type=float, default=0.7)
    parser.add_argument("--text_path", type=str, default="experiments/byte_ts_001/text.txt")
    args=parser.parse_args()
    train(**vars(args))
