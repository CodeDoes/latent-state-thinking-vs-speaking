"""RWKV state carry ablation.

Ready-to-go for theories/rwkv-state-carry.md

Tests state carry vs zero init for long-horizon NIAH/logic_niiah.

Usage:
  python -m src.train_rwkv_carry --mode zero --exp_id rwkv_carry_zero_001 --steps 2000
  python -m src.train_rwkv_carry --mode stateful --exp_id rwkv_carry_stateful_001 --steps 2000
  python -m src.train_rwkv_carry --mode learned --exp_id rwkv_carry_learned_001 --steps 2000
"""

import argparse, json, time, random
from pathlib import Path
import torch
import torch.nn.functional as F

from domains.rwkv.rwkv_nano import RWKVNano, count_params
from threads.memory_growth.logic_niiah_generator import LogicNiiahGenerator

# ── Char vocab (copied from train_rwkv.py) ──
CHARS = [
    '\n', ' ', '!', ',', '-', '.', ':', '=',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
]
SPECIAL = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
VOCAB = SPECIAL + CHARS
char_to_id = {c: i for i, c in enumerate(VOCAB)}
id_to_char = {i: c for c, i in char_to_id.items()}
PAD_ID = char_to_id['<PAD>']
UNK_ID = char_to_id['<UNK>']

def encode(text: str):
    return [char_to_id.get(c, UNK_ID) for c in text]

def decode(ids):
    return ''.join(id_to_char.get(i, '<UNK>') for i in ids)

def generate_batch_tensors(generator, batch_size, max_len, gen_kwargs):
    texts, answers_list = [], []
    for _ in range(batch_size):
        sample = generator.generate(**gen_kwargs)
        texts.append(sample['text'])
        answers_list.append(sample['answers'])
    # encode
    input_ids=[]
    targets=[]
    masks=[]
    for txt in texts:
        enc=encode(txt)[:max_len]
        # target same as input for causal LM, but we will evaluate only answer positions
        # For simplicity, causal LM training: predict next token
        # We'll create input = enc[:-1], target = enc[1:]
        if len(enc)<2:
            enc=enc+[PAD_ID]*2
        inp=enc[:-1]
        tgt=enc[1:]
        # pad
        while len(inp)<max_len-1:
            inp.append(PAD_ID)
            tgt.append(PAD_ID)
        inp=inp[:max_len-1]
        tgt=tgt[:max_len-1]
        input_ids.append(inp)
        targets.append(tgt)
        mask=[1 if t!=PAD_ID else 0 for t in tgt]
        masks.append(mask)
    return torch.tensor(input_ids), torch.tensor(targets), torch.tensor(masks, dtype=torch.float)

def evaluate(model, generator, max_len, gen_kwargs, num_examples=32, device="cpu"):
    model.eval()
    correct=0
    total=0
    digit_correct=0
    digit_total=0
    with torch.no_grad():
        for _ in range(num_examples):
            sample=generator.generate(**gen_kwargs)
            enc=encode(sample['text'][:max_len])
            input_t=torch.tensor([enc], dtype=torch.long, device=device)
            logits,_=model(input_t)
            pred_ids=logits.argmax(dim=-1)[0].cpu().tolist()
            pred_text=decode(pred_ids)
            # answers is list of strings (e.g. ["42","17"])
            for ans_val in sample['answers']:
                total+=1
                if str(ans_val) in pred_text:
                    correct+=1
                for ch in str(ans_val):
                    digit_total+=1
                    if ch in pred_text:
                        digit_correct+=1
    model.train()
    return {"accuracy":correct/max(total,1), "digit_acc":digit_correct/max(digit_total,1),
            "exact_correct":correct, "total_answers":total,
            "digit_correct":digit_correct, "digit_total":digit_total}

class StatefulWrapper:
    """Manages RWKV state carry across batches."""
    def __init__(self, num_layers, dim, device):
        self.num_layers=num_layers
        self.dim=dim
        self.device=device
        self.state=None

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        # detach
        if new_state is None:
            self.state=None
        else:
            self.state=[{k: v.detach() for k,v in layer_state.items()} for layer_state in new_state]

    def reset(self):
        self.state=None

def train(exp_id="rwkv_carry_zero_001", mode="zero", steps=2000, batch_size=8, max_len=256,
          dim=128, layers=3, lr=3e-4, seed=42, log_every=100, eval_every=500,
          noise_min=1, noise_max=10, num_vars=3, device="cpu"):

    dev=torch.device(device if torch.cuda.is_available() or device=="cpu" else "cpu")
    if device!="cpu" and torch.cuda.is_available():
        dev=torch.device("cuda")

    exp_dir=Path("experiments")/exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    generator=LogicNiiahGenerator(seed=seed)
    gen_kwargs=dict(num_vars=num_vars, min_transforms=1, max_transforms=3,
                    noise_min=noise_min, noise_max=noise_max)

    model=RWKVNano(vocab_size=len(VOCAB), dim=dim, num_layers=layers, pad_token_id=PAD_ID).to(dev)
    optimizer=torch.optim.AdamW(model.parameters(), lr=lr)

    # learned init mode: add learnable initial state parameters
    learned_init_params=None
    if mode=="learned":
        # create per-layer learned init state: num, den, xx, xx2 each [dim]
        learned_init_params=[]
        for i in range(layers):
            num=torch.nn.Parameter(torch.zeros(dim))
            den=torch.nn.Parameter(torch.zeros(dim))
            xx=torch.nn.Parameter(torch.zeros(dim))
            xx2=torch.nn.Parameter(torch.zeros(dim))
            learned_init_params.extend([num, den, xx, xx2])
        learned_init_params=torch.nn.ParameterList(learned_init_params)
        # register with model? We'll just add to optimizer separately
        optimizer.add_param_group({"params": learned_init_params, "lr": lr})
        print(f"Learned init params added: {len(learned_init_params)} tensors")

    state_mgr=StatefulWrapper(layers, dim, dev)

    config=dict(exp_id=exp_id, mode=mode, steps=steps, batch_size=batch_size, max_len=max_len,
                dim=dim, layers=layers, lr=lr, noise_max=noise_max,
                hypothesis="state carry vs zero init")
    (exp_dir/"config.json").write_text(json.dumps(config,indent=2))

    print(f"Params {count_params(model):,} mode {mode} device {dev}")

    metrics_log=[]
    best_acc=0.0
    t0=time.time()

    # Warmup generator RNG to avoid repeating data seen in previous runs? Not needed.

    for step in range(1, steps+1):
        input_ids, targets, mask = generate_batch_tensors(generator, batch_size, max_len-1, gen_kwargs)
        input_ids=input_ids.to(dev)
        targets=targets.to(dev)
        mask=mask.to(dev)

        # State handling
        if mode=="zero":
            state=None
        elif mode=="stateful":
            state=state_mgr.get_state()
        elif mode=="learned":
            # build state from learned params
            state=[]
            idx=0
            for _ in range(layers):
                num=learned_init_params[idx].unsqueeze(0).expand(batch_size, -1)
                den=learned_init_params[idx+1].unsqueeze(0).expand(batch_size, -1)
                xx=learned_init_params[idx+2].unsqueeze(0).expand(batch_size, -1)
                xx2=learned_init_params[idx+3].unsqueeze(0).expand(batch_size, -1)
                idx+=4
                state.append({"num":num, "den":den, "xx":xx, "xx2":xx2})
            # If we also have carried state, we could blend – for simplicity, on step 1 use learned, then carry overrides?
            # For this ablation, learned init replaces zero init only at reset points. We will carry after first step similar to stateful but init from learned.
            if state_mgr.get_state() is not None and step>1:
                # carry dominates after first
                state=state_mgr.get_state()
        else:
            state=None

        # Forward
        if mode in ("stateful","learned"):
            logits, new_state = model(input_ids, state=state, return_state=True)
            state_mgr.update_state(new_state)
        else:
            logits, _ = model(input_ids)
            # also get new_state for logging but don't carry
            if step%10==0:
                # occasional state for decay stats
                pass

        loss=F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        loss=loss.view_as(mask)
        loss=(loss*mask).sum()/(mask.sum()+1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step%log_every==0 or step==1:
            elapsed=time.time()-t0
            # decay stats
            with torch.no_grad():
                decays=[]
                for name, p in model.named_parameters():
                    if "time_decay" in name:
                        w=torch.exp(-torch.exp(p)).mean().item()
                        decays.append(w)
                mean_w=sum(decays)/len(decays) if decays else 0
            print(f"step {step:4d}/{steps} loss {loss.item():.4f} mean_w {mean_w:.3f} elapsed {elapsed:.0f}s mode {mode}")

        if step%eval_every==0 or step==1:
            eval_metrics=evaluate(model, generator, max_len, gen_kwargs, num_examples=32, device=dev)
            acc=eval_metrics["accuracy"]
            if acc>best_acc: best_acc=acc
            print(f"  eval acc {acc:.3f} digit {eval_metrics['digit_acc']:.3f} best {best_acc:.3f}")
            metrics_log.append({"step":step,"loss":loss.item(),"accuracy":acc,"best_acc":best_acc,**eval_metrics})
            (exp_dir/"samples.json").write_text(json.dumps(metrics_log[-10:],indent=2))

        if step%500==0:
            # reset stateful occasionally to simulate new story boundary
            if mode=="stateful":
                # 10% chance reset to simulate story boundary
                if random.random()<0.1:
                    state_mgr.reset()

    final_eval=evaluate(model, generator, max_len, gen_kwargs, num_examples=64, device=dev)
    result={"final_accuracy":final_eval["accuracy"],"best_accuracy":best_acc,"exp_id":exp_id,"mode":mode,"params":count_params(model)}
    (exp_dir/"metrics.json").write_text(json.dumps(result,indent=2))
    print(f"Done {exp_id} best {best_acc:.3f} final {final_eval['accuracy']:.3f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="rwkv_carry_zero_001")
    ap.add_argument("--mode", default="zero", choices=["zero","stateful","learned"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--noise_min", type=int, default=1)
    ap.add_argument("--noise_max", type=int, default=10)
    ap.add_argument("--num_vars", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args=ap.parse_args()
    train(**vars(args))
