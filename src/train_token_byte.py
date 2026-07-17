"""Token vs byte head training.

Ready-to-go for theories/token-vs-byte-head.md

  python -m src.train_token_byte --vocab_mode byte --exp_id tok_byte_bytehead_001 --steps 2000
  python -m src.train_token_byte --vocab_mode token --exp_id tok_byte_tokenhead_001 --steps 2000
"""

import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from src.token_byte_head_model import TokenByteHeadModel
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

def train(exp_id="tok_byte_bytehead_001", vocab_mode="byte", token_vocab_size=1024, bytes_per_token=4,
          steps=2000, batch_size=8, max_len=128, lr=3e-4, dim=64, patch_size=4,
          enc_layers=2, core_layers=2, dec_layers=2, enc_max_loops=3, core_depth_loops=2, dec_max_loops=3,
          entropy_weight=0.01, log_every=100, text_path="experiments/byte_ts_001/text.txt"):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream=load_text(Path(text_path))
    print(f"{exp_id} vocab_mode={vocab_mode} device={device} bytes={len(stream):,}")

    model=TokenByteHeadModel(
        dim=dim, patch_size=patch_size, enc_layers=enc_layers, core_layers=core_layers, dec_layers=dec_layers,
        enc_max_loops=enc_max_loops, core_depth_loops=core_depth_loops, dec_max_loops=dec_max_loops,
        vocab_mode=vocab_mode, token_vocab_size=token_vocab_size, bytes_per_token=bytes_per_token
    ).to(device)

    optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
    batch_iter=make_batches(stream, max_len, batch_size)
    exp_dir=Path("experiments")/exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg=dict(exp_id=exp_id, vocab_mode=vocab_mode, token_vocab_size=token_vocab_size, bytes_per_token=bytes_per_token,
             steps=steps, dim=dim, hypothesis="byte dense supervision vs token sparse")
    (exp_dir/"config.json").write_text(json.dumps(cfg,indent=2))

    metrics=[]
    model.train()
    t0=time.time()
    best=float("inf")

    for step in range(steps):
        input_ids, targets = next(batch_iter)
        input_ids, targets = input_ids.to(device), targets.to(device)

        if vocab_mode=="byte":
            logits, info = model.base(input_ids)
            loss=F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID)
        else:
            logits, info = model.base(input_ids)  # [B,T,token_vocab]
            # downsample for token CE
            bpt=bytes_per_token
            # logits every bpt
            logits_down=logits[:, ::bpt, :]
            token_targets=model.byte_to_token_targets(targets)
            min_len=min(logits_down.shape[1], token_targets.shape[1])
            logits_down=logits_down[:,:min_len,:]
            token_targets=token_targets[:,:min_len].to(device)
            # ignore PAD? token targets derived from byte targets, which may have PAD
            loss=F.cross_entropy(logits_down.reshape(-1, logits_down.size(-1)), token_targets.reshape(-1), ignore_index=PAD_ID)

        # entropy term from base
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
        total=loss+entropy_weight*ent_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if step%log_every==0 or step==steps-1:
            elapsed=time.time()-t0
            print(f"step {step:4d} loss {loss.item():.4f} ent {ent_loss.item():.4f} mode {vocab_mode} {elapsed:.1f}s")
            metrics.append({"step":step,"loss":loss.item(),"ent":ent_loss.item(),"vocab_mode":vocab_mode})
            if loss.item()<best: best=loss.item()
            (exp_dir/"metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))
            if step%(log_every*5)==0:
                with torch.no_grad():
                    model.base.eval()
                    inp,_=next(batch_iter)
                    inp=inp[:1].to(device)
                    log,_=model.base(inp)
                    pred=torch.argmax(log, dim=-1)[0].tolist()
                    if vocab_mode=="byte":
                        dec="".join(chr(ID_TO_BYTE.get(i,63)) for i in pred[:120])
                    else:
                        # token mode decode as bytes via modulo
                        dec="".join(chr((i%256)) for i in pred[:120])
                    (exp_dir/"sample.txt").write_text(f"mode {vocab_mode} {dec}")
                    model.base.train()

    (exp_dir/"metrics.json").write_text(json.dumps({"final_loss":loss.item(),"best_loss":best,"exp_id":exp_id},indent=2))
    print(f"Done {exp_id} best {best:.4f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="tok_byte_bytehead_001")
    ap.add_argument("--vocab_mode", default="byte", choices=["byte","token"])
    ap.add_argument("--token_vocab_size", type=int, default=1024)
    ap.add_argument("--bytes_per_token", type=int, default=4)
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
    args=ap.parse_args()
    train(**vars(args))
