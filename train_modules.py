#!/usr/bin/env python3
"""
Train the separable latent-state modules, each with its OWN objective.

Curriculum (per project direction):
  Phase 0  token<->state autoencoder  -> "output sane words"
  Phase 1  latent algebra, each piece separate:
            make_B(A)->B, make_A(B)->A, continue(A)->A2, continue(B)->B2,
            Answer_in_format_D(A,B,C)->D  (composed state then decoded)

Each module is trained independently (own optimizer + own loss). The 'remember
the correct thing' objective is make_B(A)->B, where the TARGET B is the encoder
state of the true answer -- so the narrative state is forced to contain the
answer, not just learn to spell tokens.

FEEDBACK / DIAGNOSTICS (added): this script now records a full time series
(per-epoch autoencoder loss, per-module MSE, per-epoch compositional QA accuracy
overall + per task) and runs isolation tests that separate the three possible
failure modes:
  (a) data diversity   -> dataset_diversity()
  (b) training health  -> loss curves over time + composer reaching D_target
  (c) math / I-O       -> io_oracle_tests (decode the TRUE teacher state) and
                          make_b_recovery()
Generated outputs are histogrammed so we can see pad/garbage spam. Results go
to telemetry.json + diagnostics.txt + curves.png (+ modules_report.json for
backwards compatibility with kaggle_ctl.py).

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
    TokenEncoder, StateDecoder, StateTransform, AnswerComposer, AnswerDecoder,
    StateCrossAttn, compose_answer,
)
from src import diagnostics as diag

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

def train_autoencoder(encoder, decoder, loader, opt, device, epochs, hist, threshold=1.2):
    encoder.train(); decoder.train()
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
        hist["phase0_loss"].append(last)
        print(f"  [autoenc] ep {ep}/{epochs} recon_loss={last:.4f}")
        print(f"  STAGE: autoenc ep={ep}/{epochs} loss={last:.4f}")
        if last < threshold:
            print(f"  autoenc below threshold {threshold}; proceeding to latent ops.")
            break
    return last


# ---------------------------------------------------------------------------
# Phase 1: latent algebra (each piece separate)
# ---------------------------------------------------------------------------

def train_make_b_attn(module, samples, encoder, opt, device, epochs, hist, name="make_B(A)->B"):
    """Train make_B as CROSS-ATTENTION over the narrative's per-token states,
    conditioned on the question. (Replaces the old single-vector MLP that
    collapsed to the mean.)"""
    module.train(); encoder.eval()
    crit = nn.MSELoss()
    for ep in range(1, epochs + 1):
        tot = 0; n = 0
        for s in samples:
            A_seq = encoder.states(_ids(s["narrative"]).unsqueeze(0).to(device)).detach()
            C = encoder.state_of(_ids(s.get("question", "")).unsqueeze(0).to(device)).detach()
            B = encoder.state_of(_ids(s["answer"]).unsqueeze(0).to(device)).detach()
            opt.zero_grad()
            pred = module(A_seq, C)
            loss = crit(pred, B)
            loss.backward(); opt.step()
            tot += loss.item(); n += 1
        last = tot / max(n, 1)
        hist["phase1"].setdefault(name, []).append(last)
        print(f"  [{name}] ep {ep}/{epochs} mse={last:.4f}")
        print(f"  STAGE: {name} ep={ep}/{epochs} mse={last:.4f}")
    return last


def train_state_map(module, src_fn, tgt_fn, samples, encoder, opt, device, epochs, name, hist):
    """Train a state->state map with an MSE objective against teacher states.
    (Used for make_A(B)->A; make_B now uses the cross-attention version above.)"""
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
        hist["phase1"].setdefault(name, []).append(last)
        print(f"  [{name}] ep {ep}/{epochs} mse={last:.4f}")
        print(f"  STAGE: {name} ep={ep}/{epochs} mse={last:.4f}")
    return last


def train_composer(composer, ans_dec, dec, samples, encoder, opt, device, epochs, hist, qa_hook=None):
    composer.train(); ans_dec.train(); encoder.eval()
    for ep in range(1, epochs + 1):
        tot = 0; n = 0
        for s in samples:
            A = encoder.state_of(_ids(s["narrative"]).unsqueeze(0).to(device)).detach()
            B = encoder.state_of(_ids(s["answer"]).unsqueeze(0).to(device)).detach()
            C = encoder.state_of(_ids(s.get("question", "")).unsqueeze(0).to(device)).detach()
            D_target = encoder.state_of(_ids("Answer: " + s["answer"]).unsqueeze(0).to(device)).detach()
            ans_ids = _ids(s["answer"]).to(device)
            # Train the head to STOP: append EOS to the target. Without this the
            # head never learns an end-of-answer signal, so greedy generation
            # emits a full max_tokens garbage tail and exact-match oracle acc
            # stays 0.00 (the real cause of the previous run's oracle failure).
            eos_id = TOK.vocab[TOK.eos_token]
            tgt_ids = torch.cat([ans_ids, torch.tensor([eos_id], device=device)])
            opt.zero_grad()
            D = composer(A, B, C)
            mse = F.mse_loss(D, D_target)
            # Train the head on BOTH the clean teacher state and the noisy
            # composer output:
            #  - clean term anchors precise decoding (incl. spaces) from the
            #    TRUE answer state -- this is why the autoencoder decoder hits
            #    0.98, and why the head previously collapsed to single-word
            #    patterns (it only saw a moving, noisy D from a random composer).
            #  - noisy term adapts the head to the actual inference
            #    distribution (composer(A,B,C)), and its gradient also flows into
            #    the composer, helping it produce a decodable state.
            # D_target is detached so the clean CE never leaks into the composer;
            # the composer is driven by MSE to *reach* D_target.
            logits_clean = ans_dec.forward_teacher(D_target, tgt_ids.unsqueeze(0))
            logits_noisy = ans_dec.forward_teacher(D, tgt_ids.unsqueeze(0))
            ce_clean = F.cross_entropy(logits_clean.reshape(-1, logits_clean.size(-1)),
                                      tgt_ids.reshape(-1))
            ce_noisy = F.cross_entropy(logits_noisy.reshape(-1, logits_noisy.size(-1)),
                                      tgt_ids.reshape(-1))
            ce = ce_clean + ce_noisy
            loss = mse + ce
            loss.backward(); opt.step()
            tot += loss.item(); n += 1
        last = tot / max(n, 1)
        hist["phase1"].setdefault("composer", []).append(last)
        print(f"  [composer] ep {ep}/{epochs} loss={last:.4f}")
        print(f"  STAGE: composer ep={ep}/{epochs} loss={last:.4f}")
        if qa_hook is not None and (ep % 3 == 0 or ep == epochs):
            qa_hook(ep)
    return last


# continue(): train on consecutive encoder states. NOTE: optimizer must be
# created ONCE outside the sample loop (the old code re-created it per sample,
# which reset AdamW state every step and crippled learning).
def train_cont(module, samples, encoder, device, epochs, name, field, hist):
    module.train(); encoder.eval()
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(module.parameters(), lr=3e-4)
    for ep in range(1, epochs + 1):
        tot = 0; n = 0
        for s in samples:
            ids = _ids(s[field])
            if ids.size(0) < 2:
                continue
            states = encoder.states(ids.unsqueeze(0).to(device)).detach()[0]  # [T, d]
            src = states[:-1]; tgt = states[1:]
            opt.zero_grad()
            pred = module(src)
            loss = crit(pred, tgt)
            loss.backward(); opt.step()
            tot += loss.item(); n += 1
        last = tot / max(n, 1)
        hist["phase1"].setdefault(name, []).append(last)
        print(f"  [{name}] ep {ep}/{epochs} mse={last:.4f}")
        print(f"  STAGE: {name} ep={ep}/{epochs} mse={last:.4f}")
    return last


# ---------------------------------------------------------------------------
# Eval (compositional inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_qa(encoder, make_b, composer, ans_dec, dataset, tokenizer, device, max_new=48):
    correct = 0; total = 0
    by_task = {}
    generated_texts = []
    eos = tokenizer.vocab[tokenizer.eos_token]
    pad = tokenizer.vocab[tokenizer.pad_token]
    for s in dataset:
        if not s.get("question"):
            continue
        total += 1
        t = s["task_type"]
        by_task.setdefault(t, [0, 0])
        A_seq = encoder.states(_ids(s["narrative"]).unsqueeze(0).to(device)).detach()
        C = encoder.state_of(_ids(s.get("question", "")).unsqueeze(0).to(device)).detach()
        ids = compose_answer(encoder, make_b, composer, ans_dec, A_seq, C,
                             max_tokens=max_new, eos_id=eos, pad_id=pad)
        gen = tokenizer.decode(ids).strip().lower()
        generated_texts.append(gen)
        exp = s["answer"].strip().lower()
        ok = gen == exp
        correct += ok
        by_task[t][0] += ok; by_task[t][1] += 1
    acc = correct / max(total, 1)
    task_acc = {t: c / m for t, (c, m) in by_task.items()}
    return acc, task_acc, total, generated_texts


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
    ap.add_argument("--analyze", action="store_true", help="Report last run only")
    args = ap.parse_args()

    if args.analyze:
        report_analyze()
        return

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

    # ----- (a) DATA DIVERSITY CHECK (run first: is it the data?) -----
    div = diag.dataset_diversity(dataset, TOK)
    print("STAGE: data_diversity " + json.dumps(div))
    print(f"  >> data: {div['n_qa']} QA samples, {div['n_unique_answers']} unique "
          f"answers, majority-baseline acc={div['majority_baseline_acc']} -> {div['verdict']}")

    encoder = TokenEncoder(TOK.vocab_size, d_state=args.d_state, d_model=args.d_model).to(device)
    decoder = StateDecoder(args.d_state, TOK.vocab_size).to(device)
    make_b = StateCrossAttn(args.d_state).to(device)
    make_a = StateTransform(args.d_state).to(device)
    cont_a = StateTransform(args.d_state).to(device)
    cont_b = StateTransform(args.d_state).to(device)
    composer = AnswerComposer(args.d_state).to(device)
    ans_dec = AnswerDecoder(args.d_state, TOK.vocab_size).to(device)

    train_loader, val_loader = make_loaders(texts, TOK, args.batch_size, args.max_seq_len)
    hist = {"phase0_loss": [], "phase1": {}, "qa_acc": [], "qa_task": {}}

    # ---- Phase 0: output sane words (the 'baseline' foundation) ----
    # --quick uses its OWN cache so it never clobbers the real 256-dim
    # foundation; a d_state mismatch triggers a retrain instead of a crash.
    found_ckpt = Path("modules_foundation_quick.pt" if args.quick else "modules_foundation.pt")
    loaded = False
    if found_ckpt.exists() and not args.fresh:
        ck = torch.load(found_ckpt, map_location=device, weights_only=True)
        if ck["encoder"]["norm.weight"].shape[0] == args.d_state:
            print("STAGE: phase0 load cached foundation (skip retrain)")
            encoder.load_state_dict(ck["encoder"])
            decoder.load_state_dict(ck["decoder"])
            loaded = True
        else:
            print(f"STAGE: phase0 cached foundation d_state mismatch "
                  f"({ck['encoder']['norm.weight'].shape[0]} != {args.d_state}); retraining")
    if not loaded:
        print("STAGE: phase0 autoencoder (train)")
        ae_opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=3e-4)
        train_autoencoder(encoder, decoder, train_loader, ae_opt, device, args.phase0_epochs, hist)
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

    # QA hook: evaluated periodically during composer training so we get a
    # *learning curve* for the answer, not just a final number.
    def qa_hook(ep):
        acc, task_acc, n, _ = run_qa(encoder, make_b, composer, ans_dec,
                                     qa_samples, TOK, device)
        hist["qa_acc"].append(acc)
        for t, a in task_acc.items():
            hist["qa_task"].setdefault(t, []).append(a)
        print(f"  [qa] ep={ep} acc={acc:.3f} " +
              " ".join(f"{t}={a:.2f}" for t, a in task_acc.items()))
        print(f"  STAGE: qa ep={ep} acc={acc:.3f} " +
              " ".join(f"{t}={a:.2f}" for t, a in task_acc.items()))

    train_make_b_attn(make_b, qa_samples, encoder,
                      torch.optim.AdamW(make_b.parameters(), lr=3e-4),
                      device, args.phase1_epochs, hist)
    train_state_map(make_a, lambda A, B, C: B, lambda A, B, C: A, qa_samples,
                    encoder, torch.optim.AdamW(make_a.parameters(), lr=3e-4),
                    device, args.phase1_epochs, "make_A(B)->A", hist)
    train_cont(cont_a, qa_samples, encoder, device, args.phase1_epochs, "continue(A)->A2",
               "narrative", hist)
    train_cont(cont_b, qa_samples, encoder, device, args.phase1_epochs, "continue(B)->B2",
               "answer", hist)
    train_composer(composer, ans_dec, decoder, qa_samples, encoder,
                   torch.optim.AdamW(list(composer.parameters()) + list(ans_dec.parameters()), lr=3e-4),
                   device, args.phase1_epochs, hist, qa_hook=qa_hook)

    # ---- Final eval + isolation diagnostics ----
    print("STAGE: eval")
    acc, task_acc, n, gens = run_qa(encoder, make_b, composer, ans_dec,
                                    qa_samples, TOK, device)
    hist["qa_acc"].append(acc)
    for t, a in task_acc.items():
        hist["qa_task"].setdefault(t, []).append(a)
    print(f"\nCompositional exact-match accuracy: {acc:.3f} (n={n})")
    for t, a in task_acc.items():
        print(f"  {t}: {a:.3f}")
    print(f"STAGE: done qa_acc={acc:.3f}")

    # (b/c) isolation: is the I/O + head fine given the TRUE teacher state?
    oracle = diag.io_oracle_tests_with_decoder(
        encoder, decoder, composer, ans_dec, make_b, qa_samples, TOK, device)
    makeb = diag.make_b_recovery(encoder, make_b, qa_samples, TOK, device)
    hist_gen = diag.token_histogram(gens, TOK)
    print("STAGE: oracle " + json.dumps(oracle))
    print("STAGE: make_B " + json.dumps(makeb))
    print("STAGE: gen_hist " + json.dumps(hist_gen))
    print(f"  >> I-O/head oracle acc={oracle['oracle_answer_head_acc']} -> {oracle['interpretation']}")
    print(f"  >> generated output: {hist_gen['verdict']}")

    # save everything
    report = {
        "qa_accuracy": acc,
        "task_accuracy": task_acc,
        "n": n,
        "d_state": args.d_state,
        "data_diversity": div,
        "io_oracle": oracle,
        "make_B": makeb,
        "gen_histogram": hist_gen,
    }
    Path("modules_report.json").write_text(json.dumps(report, indent=2))
    Path("telemetry.json").write_text(json.dumps(
        {"config": vars(args), "history": hist, "final_report": report}, indent=2))
    diag.save_telemetry("diagnostics.txt", _render_diagnostics(div, oracle, makeb, hist_gen, hist, acc, task_acc))
    diag.render_curves(hist, "curves.png")
    print("Saved modules_report.json, telemetry.json, diagnostics.txt, curves.png")


def _render_diagnostics(div, oracle, makeb, hist_gen, hist, acc, task_acc) -> str:
    L = []
    L.append("=" * 70)
    L.append("DIAGNOSTICS: why did this run score the way it did?")
    L.append("=" * 70)
    L.append("\n[1] DATA DIVERSITY  (is the dataset too easy / biased?)")
    L.append(f"    QA samples={div['n_qa']}  unique answers={div['n_unique_answers']}")
    L.append(f"    majority-class baseline acc={div['majority_baseline_acc']}")
    L.append(f"    verdict: {div['verdict']}")
    L.append("\n[2] TRAINING HEALTH  (do losses move the right way?)")
    if hist["phase0_loss"]:
        L.append(f"    autoenc recon_loss: start={hist['phase0_loss'][0]:.3f} "
                 f"end={hist['phase0_loss'][-1]:.3f}")
    for name, vals in hist["phase1"].items():
        if vals:
            L.append(f"    {name}: start={vals[0]:.3f} end={vals[-1]:.3f} "
                     f"{'OK(down)' if vals[-1] < vals[0] else 'BAD(up)'}")
    L.append("\n[3] MATH / I-O  (does the architecture actually work?)")
    L.append(f"    autoenc reconstruction char-acc = {oracle['autoenc_recon_char_acc']}")
    L.append(f"    ORACLE answer-head acc (true teacher state) = {oracle['oracle_answer_head_acc']}")
    L.append(f"    composer reaches D_target MSE = {oracle['composer_D_mse']}")
    L.append(f"    make_B(A) recovery MSE = {makeb['make_B_mse']} ({makeb['verdict']})")
    L.append(f"    interpretation: {oracle['interpretation']}")
    L.append("\n[4] GENERATION OUTPUT  (what are we actually producing?)")
    L.append(f"    token fractions={hist_gen['fractions']}")
    L.append(f"    verdict: {hist_gen['verdict']}")
    L.append("\n[5] FINAL QA")
    L.append(f"    overall acc={acc:.3f}")
    for t, a in task_acc.items():
        L.append(f"      {t}: {a:.3f}")
    L.append("")
    return "\n".join(L)


def report_analyze():
    p = Path("telemetry.json")
    if not p.exists():
        print("telemetry.json not found (run first).")
        return
    data = json.loads(p.read_text())
    hist = data["history"]
    rep = data["final_report"]
    print("=== LAST RUN (from telemetry.json) ===")
    print(f"final QA acc={rep['qa_accuracy']:.3f}  task_acc={rep['task_accuracy']}")
    if hist.get("qa_acc"):
        print(f"QA accuracy over time: {['%.2f' % x for x in hist['qa_acc']]}")
    print("\nDATA:", rep["data_diversity"]["verdict"])
    print("I-O/HEAD ORACLE:", rep["io_oracle"]["interpretation"])
    print("GEN:", rep["gen_histogram"]["verdict"])


if __name__ == "__main__":
    main()
