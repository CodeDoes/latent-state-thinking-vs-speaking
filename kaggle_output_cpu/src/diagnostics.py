"""
Diagnostics for the hybrid latent-state experiments.

The previous run trained end-to-end but reported ONLY a final 0.0 QA accuracy,
so we could not tell whether failure was due to:
  (a) lack of data diversity,
  (b) a training problem (loss not moving / diverging), or
  (c) a math/architecture bug (I/O path or algebra wrong).

This module produces, on every run, the evidence to separate those three:

  * dataset_diversity()  -> is the data rich enough / biased?
  * io_oracle_tests()    -> does the I/O + answer head work given the TRUE
                            teacher state?  (isolates math from training)
  * make_b_recovery()    -> given the real narrative, how close is make_B(A) to
                            the true answer state B?  (isolates algebra training)
  * token_histogram()    -> what kind of garbage are we generating?
                           (pad spam "."/"..." => decode/I-O broken)
  * render_curves()      -> loss + accuracy-over-time PNG.

All functions are pure-ish (take modules + data, return dicts) so they can be
called from train_modules.py and bench.py.
"""
from collections import Counter
import json
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# (a) Data diversity
# ---------------------------------------------------------------------------

def dataset_diversity(dataset: list, tokenizer) -> dict:
    """How rich / biased is the data? Prints the 'is it the data?' answer."""
    per_task = Counter(s["task_type"] for s in dataset)
    qa = [s for s in dataset if s.get("question")]
    answers = Counter(s["answer"] for s in qa)
    # majority-class baseline: always predict the most common answer
    majority = answers.most_common(1)[0][1] if answers else 0
    maj_baseline = majority / max(len(qa), 1)

    lens = [len(s["answer"]) for s in qa]
    avg_len = sum(lens) / max(len(lens), 1)

    # recall passwords should be high-entropy / many unique
    recall = [s for s in qa if s["task_type"] == "recall"]
    recall_unique = len(set(s["answer"] for s in recall))

    return {
        "n_total": len(dataset),
        "per_task": dict(per_task),
        "n_qa": len(qa),
        "n_unique_answers": len(answers),
        "most_common_answer": answers.most_common(1)[0] if answers else None,
        "majority_baseline_acc": round(maj_baseline, 4),
        "avg_answer_chars": round(avg_len, 2),
        "recall_n": len(recall),
        "recall_unique_passwords": recall_unique,
        "verdict": (
            "data looks diverse (low majority baseline, many unique answers)"
            if maj_baseline < 0.2 and len(answers) > 10
            else "data may be biased toward a few answers (majority baseline high)"
        ),
    }


# ---------------------------------------------------------------------------
# (b/c) I/O + algebra isolation
# ---------------------------------------------------------------------------

@torch.no_grad()
def io_oracle_tests_with_decoder(encoder, decoder, composer, ans_dec, make_b,
                                 dataset, tokenizer, device, max_new=48) -> dict:
    """Same as io_oracle_tests but with the actual StateDecoder available."""
    eos = tokenizer.vocab[tokenizer.eos_token]
    pad = tokenizer.vocab[tokenizer.pad_token]
    qa = [s for s in dataset if s.get("question")]

    # 1. autoencoder reconstruction char-accuracy (narrative -> narrative)
    recon_hits, recon_tot = 0, 0
    for s in qa[:30]:
        ids = _enc_ids(s["narrative"], tokenizer).unsqueeze(0).to(device)
        states = encoder.states(ids)
        logits = decoder(states)
        pred = logits.argmax(-1)[0]
        tgt = ids[0]
        recon_hits += int((pred == tgt).sum())
        recon_tot += int(tgt.numel())
    autoenc_recon_char_acc = recon_hits / max(recon_tot, 1)

    # 2. oracle answer-head accuracy (teacher state D_target)
    oracle_correct, oracle_tot = 0, 0
    composer_mse_sum, composer_mse_n = 0.0, 0
    for s in qa:
        A_seq = encoder.states(_enc_ids(s["narrative"], tokenizer).unsqueeze(0).to(device)).detach()
        A = A_seq[:, -1, :]
        B = encoder.state_of(_enc_ids(s["answer"], tokenizer).unsqueeze(0).to(device)).detach()
        C = encoder.state_of(_enc_ids(s.get("question", ""), tokenizer).unsqueeze(0).to(device)).detach()
        D_target = encoder.state_of(_enc_ids("Answer: " + s["answer"], tokenizer).unsqueeze(0).to(device)).detach()
        # oracle: decode the teacher target directly
        ids = ans_dec.generate(D_target, max_tokens=max_new, eos_id=eos, pad_id=pad)
        gen = tokenizer.decode(ids).strip().lower()
        exp = s["answer"].strip().lower()
        oracle_correct += int(gen == exp)
        oracle_tot += 1
        # how far does the real pipeline get from D_target?
        D = composer(A, make_b(A_seq, C), C)
        composer_mse_sum += float(F.mse_loss(D, D_target).item())
        composer_mse_n += 1

    return {
        "autoenc_recon_char_acc": round(autoenc_recon_char_acc, 4),
        "oracle_answer_head_acc": round(oracle_correct / max(oracle_tot, 1), 4),
        "oracle_n": oracle_tot,
        "composer_D_mse": round(composer_mse_sum / max(composer_mse_n, 1), 4),
        "interpretation": _interpret_oracle(
            autoenc_recon_char_acc, oracle_correct / max(oracle_tot, 1),
            composer_mse_sum / max(composer_mse_n, 1)),
    }


def _interpret_oracle(autoenc, oracle, composer_mse):
    # The precise I/O test is `oracle` (decode the TRUE teacher state -> answer).
    # autoenc_recon_char_acc (reconstructing a long narrative perfectly) is only
    # informational, since even a good autoencoder rarely hits >0.5 char-acc on
    # long text. We gate on `oracle`, not on autoenc.
    if oracle < 0.5:
        return ("MATH: the answer head cannot decode even the TRUE teacher state "
                f"(<0.5, got {oracle:.2f}) -- the generation head / I-O mapping is wrong.")
    if composer_mse > 0.05:
        return ("TRAINING/ALGEBRA: I/O + head are fine (oracle high) but the composer "
                "does not reach the teacher state (high MSE) -- the latent algebra "
                "(make_B / composer) is not learning.")
    if makeb.get("collapsing"):
        return ("ALGEBRA COLLAPSE: make_B sits at the mean-prediction floor "
                f"(MSE={makeb['make_B_mse']} ~ var={makeb['target_var']}) -- the "
                "single pooled narrative state is too lossy to extract the answer; "
                "use attention over the narrative's token states instead.")
    return ("ALL COMPONENTS OK: I/O, head, and algebra align. Remaining errors are "
            "genuine reasoning difficulty, not a structural bug.")


@torch.no_grad()
def make_b_recovery(encoder, make_b, dataset, tokenizer, device) -> dict:
    """Given the real narrative, how close is make_B(A_seq, C) to the true B?

    Also computes the target's per-sample variance (the MSE floor a mean
    predictor would hit). If achieved MSE ~= variance floor, the module has
    COLLAPSED to the mean -> the input representation lacks the needed info.
    """
    qa = [s for s in dataset if s.get("question")]
    tot, n = 0.0, 0
    Bs = []
    for s in qa:
        A_seq = encoder.states(_enc_ids(s["narrative"], tokenizer).unsqueeze(0).to(device)).detach()
        C = encoder.state_of(_enc_ids(s.get("question", ""), tokenizer).unsqueeze(0).to(device)).detach()
        B = encoder.state_of(_enc_ids(s["answer"], tokenizer).unsqueeze(0).to(device)).detach()
        pred = make_b(A_seq, C)
        tot += float(F.mse_loss(pred, B).item())
        n += 1
        Bs.append(B[0])
    mse = tot / max(n, 1)
    var = float(torch.stack(Bs).var(0).mean().item())   # mean per-dim variance
    collapsing = bool(mse >= 0.9 * var)
    verdict = ("COLLAPSING to mean (MSE ~ variance floor: input state lacks info)"
              if collapsing else ("weak" if mse > 0.08 else "good"))
    return {"make_B_mse": round(mse, 4), "target_var": round(var, 4),
            "collapsing": collapsing, "verdict": verdict}


# ---------------------------------------------------------------------------
# (c) Generation pathology
# ---------------------------------------------------------------------------

def token_histogram(generated_texts: list, tokenizer) -> dict:
    """What are we actually producing? pad spam (".....") => decode broken."""
    pad = tokenizer.pad_token
    eos = tokenizer.eos_token
    unk = tokenizer.unk_token
    counts = Counter()
    for t in generated_texts:
        for ch in t:
            if ch == pad:
                counts["pad"] += 1
            elif ch == eos:
                counts["eos"] += 1
            elif ch == unk:
                counts["unk"] += 1
            elif ch in " \n\t":
                counts["space"] += 1
            elif ch.isalpha():
                counts["alpha"] += 1
            else:
                counts["other"] += 1
    total = sum(counts.values()) or 1
    frac = {k: round(v / total, 3) for k, v in counts.items()}
    if frac.get("pad", 0) > 0.4 or frac.get("other", 0) > 0.4:
        verdict = "GENERATION BROKEN: mostly pad/garbage tokens"
    elif frac.get("alpha", 0) > 0.6:
        verdict = "produces alphabetic text (plausible answers)"
    else:
        verdict = "mixed / unclear output"
    return {"fractions": frac, "verdict": verdict}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def render_curves(history: dict, out_path: str) -> bool:
    """Plot loss + accuracy over time. Returns True if a PNG was written.

    `history` keys: 'phase0_loss':[...], 'phase1': {name: [mse...]},
    'qa_acc':[...], 'qa_task': {task:[...]}.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (render_curves skipped, matplotlib unavailable: {e})")
        return False

    epochs = list(range(1, len(history.get("qa_acc", [])) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # left: QA accuracy over time (overall + per task)
    ax = axes[0]
    if history.get("qa_acc"):
        ax.plot(epochs, history["qa_acc"], "k-o", label="overall")
    for task, vals in history.get("qa_task", {}).items():
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, "-", label=task)
    ax.set_title("QA exact-match accuracy over time")
    ax.set_xlabel("eval epoch"); ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # right: phase-1 module MSE over time (training health)
    ax = axes[1]
    for name, vals in history.get("phase1", {}).items():
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, "-", label=name)
    if history.get("phase0_loss"):
        ax.plot(range(1, len(history["phase0_loss"]) + 1),
                history["phase0_loss"], "k--", label="autoenc")
    ax.set_title("Module training loss (MSE) over time")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _enc_ids(text, tokenizer):
    if not text:
        text = " "
    return torch.tensor([tokenizer.vocab.get(c, tokenizer.vocab[tokenizer.unk_token])
                         for c in text], dtype=torch.long)


def save_telemetry(path: str, obj: dict):
    Path(path).write_text(json.dumps(obj, indent=2))
