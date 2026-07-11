#!/usr/bin/env python3
"""
Train the separable latent-state modules, each with its OWN objective.

Curriculum (per project direction):
  Phase 0  token<->state autoencoder  -> "output sane words"
  Phase 1  latent algebra, each piece trained separately:
            make_B(A)->B, make_A(B)->A, continue(A)->A2, continue(B)->B2,
            Answer_in_format_D(A,B,C)->D  (composed state then decoded)

Each module is trained independently (own optimizer + own loss). The 'remember
the correct thing' objective is make_B(A)->B, where the TARGET B is the encoder
state of the true answer -- so the narrative state is forced to contain the
answer, not just learn to spell tokens.

Usage:
  python train_modules.py --quick                      # tiny CPU sanity
  python train_modules.py --device cuda --epochs 25     # real Kaggle run
  python train_modules.py --analyze                     # report last run
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import generate_dataset, format_for_training, build_prompt, parse_answer
from src.tokenizer import CharTokenizer
from src.modules import (
    TokenEncoder, StateDecoder, StateTransform, AnswerComposer,
    ReasoningStep, ContextManager, Tape, compose_answer,
)

QUICK = dict(d_state=48, n_samples=400, epochs=4, max_seq_len=192,
             phase0_epochs=8, phase1_epochs=3, d_model=32, batch_size=32)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = self.tokenizer.encode(self.texts[i], max_len=self.max_len)
        ids = ids[: self.max_len]
        while len(ids) < self.max_len:
            ids.append(self.tokenizer.vocab[self.tokenizer.pad_token])
        x = torch.tensor(ids, dtype=torch.long)
        return x, x  # reconstruct input from itself


def make_loaders(texts, tokenizer, batch_size, max_len):
    ds = TextDataset(texts, tokenizer, max_len)
    n = len(ds)
    train = TextDataset(texts[: int(0.8 * n)], tokenizer, max_len)
    val = TextDataset(texts[int(0.8 * n):], tokenizer, max_len)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size))


# ---------------------------------------------------------------------------
# Phase 0: autoencoder (output sane words)
# ---------------------------------------------------------------------------

def train_autoencoder(encoder, decoder, loader, opt, device, epochs, threshold=1.2):
    encoder.train(); decoder.train()
    crit = nn.CrossEntropyLoss(ignore_index=encoder.embed.padding_idx if hasattr(encoder.embed, 'padding_idx') else -100)
    last = 1e9
    for ep in range(1, epochs + 1):
        tot = 0; n = 0
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            states = encoder.states(x)              # [B,T,d_state]
            logits = decoder(states)               # [B,T,vocab]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   x.reshape(-1), ignore_index=0)
            loss.backward(); opt.step()
            tot += loss.item(); n += 1
        last = tot / max(n, 1)
        print(f"  [autoenc] ep {ep}/{epochs} recon_loss={last:.4f}")
        print(f"  STAGE: autoenc ep={ep}/{epochs} loss={last:.4f}")
        if last < threshold:
            print(f"  autoenc below threshold {threshold}; proceeding to latent ops.")
            break
    return last


# ---------------------------------------------------------------------------
# Phase 1: latent algebra (each piece separate)
# ---------------------------------------------------------------------------

def train_state_map(module, src_fn, tgt_fn, samples, encoder, opt, device, epochs, name):
    """Train a state->state map with an MSE objective against teacher states."""
    module.train(); encoder.eval()
    crit = nn.MSELoss()
    for ep in range(1, epochs + 1):
        tot = 0; n = 0
        for s in samples:
            A = encoder.state_of(_ids(s["narrative"]).unsqueeze(0).to(device)).detach()
            B = encoder.state_of(_ids(s["answer"]).unsqueeze(0).to(device)).detach()
            C = encoder.state_of(_ids(s.get("question", "")).unsqueeze(0).to(device)).detach()
            src = src_fn(A, B, C)
            tgt = tgt_fn(A, B, C)
            opt.zero_grad()
            pred = module(src)
            loss = crit(pred, tgt)
            loss.backward(); opt.step()
            tot += loss.item(); n += 1
        last = tot / max(n, 1)
        print(f"  [{name}] ep {ep}/{epochs} mse={last:.4f}")
        print(f"  STAGE: {name} ep={ep}/{epochs} mse={last:.4f}")
    return last


def train_composer(composer, dec, samples, encoder, opt, device, epochs):
    composer.train(); dec.train(); encoder.eval()
    for ep in range(1, epochs + 1):
        tot = 0; n = 0
        for s in samples:
            A = encoder.state_of(_ids(s["narrative"]).unsqueeze(0).to(device)).detach()
            B = encoder.state_of(_ids(s["answer"]).unsqueeze(0).to(device)).detach()
            C = encoder.state_of(_ids(s.get("question", "")).unsqueeze(0).to(device)).detach()
            D_target = encoder.state_of(_ids("Answer: " + s["answer"]).unsqueeze(0).to(device)).detach()
            ans_ids = _ids(s["answer"])
            opt.zero_grad()
            D = composer(A, B, C)
            mse = F.mse_loss(D, D_target)
            logits = dec(D.unsqueeze(1))            # [1,1,vocab]
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 ans_ids[:1].to(device)).detach()
            loss = mse + ce
            loss.backward(); opt.step()
            tot += loss.item(); n += 1
        last = tot / max(n, 1)
        print(f"  [composer] ep {ep}/{epochs} loss={last:.4f}")
        print(f"  STAGE: composer ep={ep}/{epochs} loss={last:.4f}")
    return last


# ---------------------------------------------------------------------------
# Eval (compositional inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_qa(encoder, make_b, composer, dec_b, dec, dataset, tokenizer, device, max_new=24):
    correct = 0; total = 0
    by_task = {}
    eos = tokenizer.vocab[tokenizer.eos_token]
    pad = tokenizer.vocab[tokenizer.pad_token]
    for s in dataset:
        if not s.get("question"):
            continue
        total += 1
        t = s["task_type"]
        by_task.setdefault(t, [0, 0])
        prompt = build_prompt(s)
        A = encoder.state_of(_ids(s["narrative"]).unsqueeze(0).to(device)).detach()
        C = encoder.state_of(_ids(s.get("question", "")).unsqueeze(0).to(device)).detach()
        ids = compose_answer(encoder, make_b, composer, dec_b, dec, A, C,
                             max_tokens=max_new, eos_id=eos, pad_id=pad)
        gen = parse_answer(tokenizer.decode(ids)).strip().lower()
        exp = s["answer"].strip().lower()
        ok = gen == exp
        correct += ok
        by_task[t][0] += ok; by_task[t][1] += 1
    acc = correct / max(total, 1)
    task_acc = {t: c / m for t, (c, m) in by_task.items()}
    return acc, task_acc, total


def _ids(text):
    # encode lazily; tokenizer is set globally. Guard empty text (story tasks
    # have no question) so the LSTM never sees a length-0 sequence.
    if not text:
        text = " "
    return torch.tensor([TOK.vocab.get(c, TOK.vocab[TOK.unk_token]) for c in text],
                        dtype=torch.long)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TOK = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--phase0_epochs", type=int, default=15)
    ap.add_argument("--phase1_epochs", type=int, default=12)
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--d_state", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true",
                   help="Retrain the autoencoder foundation even if cached")
    args = ap.parse_args()
    if args.quick:
        for k, v in QUICK.items():
            setattr(args, k, v)

    global TOK
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"STAGE: train_modules device={device} d_state={args.d_state} n={args.n_samples}")

    dataset = generate_dataset(n_samples=args.n_samples, seed=args.seed)
    texts = [format_for_training(s) for s in dataset]
    TOK = CharTokenizer(texts, max_vocab=256)

    encoder = TokenEncoder(TOK.vocab_size, d_state=args.d_state, d_model=args.d_model).to(device)
    decoder = StateDecoder(args.d_state, TOK.vocab_size).to(device)
    make_b = StateTransform(args.d_state).to(device)
    make_a = StateTransform(args.d_state).to(device)
    cont_a = StateTransform(args.d_state).to(device)
    cont_b = StateTransform(args.d_state).to(device)
    composer = AnswerComposer(args.d_state).to(device)

    train_loader, val_loader = make_loaders(texts, TOK, args.batch_size, args.max_seq_len)

    # ---- Phase 0: output sane words (the 'baseline' foundation) ----
    # Train ONCE and cache; later runs load it instead of re-running the
    # autoencoder every time (cheap iteration on the latent algebra).
    found_ckpt = Path("modules_foundation.pt")
    if found_ckpt.exists() and not args.fresh:
        print("STAGE: phase0 load cached foundation (skip retrain)")
        ck = torch.load(found_ckpt, map_location=device, weights_only=True)
        encoder.load_state_dict(ck["encoder"])
        decoder.load_state_dict(ck["decoder"])
    else:
        print("STAGE: phase0 autoencoder (train)")
        ae_opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=3e-4)
        train_autoencoder(encoder, decoder, train_loader, ae_opt, device, args.phase0_epochs)
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                   found_ckpt)
        print(f"  saved foundation -> {found_ckpt}")

    # sanity: reconstruct a validation sample
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        x = val_loader.dataset[0][0].unsqueeze(0).to(device)
        rec = decoder(encoder.states(x)).argmax(-1)[0]
        print("  sample reconstruction:", TOK.decode(rec.tolist())[:80])

    # ---- Phase 1: latent algebra, each piece separate ----
    print("STAGE: phase1 latent algebra")
    qa_samples = [s for s in dataset[int(0.8 * len(dataset)):] if s.get("question")]
    train_state_map(make_b, lambda A, B, C: A, lambda A, B, C: B, qa_samples,
                    encoder, torch.optim.AdamW(make_b.parameters(), lr=3e-4),
                    device, args.phase1_epochs, "make_B(A)->B")
    train_state_map(make_a, lambda A, B, C: B, lambda A, B, C: A, qa_samples,
                    encoder, torch.optim.AdamW(make_a.parameters(), lr=3e-4),
                    device, args.phase1_epochs, "make_A(B)->A")
    train_cont(cont_a, qa_samples, encoder, device, args.phase1_epochs, "continue(A)->A2",
               field="narrative")
    train_cont(cont_b, qa_samples, encoder, device, args.phase1_epochs, "continue(B)->B2",
               field="answer")
    train_composer(composer, decoder, qa_samples, encoder,
                   torch.optim.AdamW(list(composer.parameters()) + list(decoder.parameters()), lr=3e-4),
                   device, args.phase1_epochs)

    # ---- Eval: compositional inference ----
    print("STAGE: eval")
    acc, task_acc, n = run_qa(encoder, make_b, composer, cont_b, decoder,
                              dataset[int(0.8 * len(dataset)):], TOK, device)
    print(f"\nCompositional exact-match accuracy: {acc:.3f} (n={n})")
    for t, a in task_acc.items():
        print(f"  {t}: {a:.3f}")
    print(f"STAGE: done qa_acc={acc:.3f}")

    # save a tiny report
    report = {"qa_accuracy": acc, "task_accuracy": task_acc, "n": n,
              "d_state": args.d_state}
    Path("modules_report.json").write_text(json.dumps(report, indent=2))
    print("Saved modules_report.json")


def train_cont(module, samples, encoder, device, epochs, name, field):
    """Train continue() on consecutive encoder states of `field` text."""
    module.train(); encoder.eval()
    crit = nn.MSELoss()
    for ep in range(1, epochs + 1):
        tot = 0; n = 0
        for s in samples:
            ids = _ids(s[field])
            if ids.size(0) < 2:
                continue
            states = encoder.states(ids.unsqueeze(0).to(device)).detach()[0]  # [T, d]
            src = states[:-1]; tgt = states[1:]
            opt = torch.optim.AdamW(module.parameters(), lr=3e-4)
            opt.zero_grad()
            pred = module(src)
            loss = crit(pred, tgt)
            loss.backward(); opt.step()
            tot += loss.item(); n += 1
        last = tot / max(n, 1)
        print(f"  [{name}] ep {ep}/{epochs} mse={last:.4f}")
        print(f"  STAGE: {name} ep={ep}/{epochs} mse={last:.4f}")
    return last


if __name__ == "__main__":
    main()
