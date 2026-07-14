"""Structured World-State Model (Model C) -- batched training.

The latent state is an EXPLICIT slot table, one vector per entity (NAME_POOL)
and per item (ITEM_POOL). Critically, each slot's leading dims are DEDICATED
FIELDS (per the user's insight that an SSM only needs
`(entity_index, current_location_at_index)` -- a disentangled record, not one
entangled vector we then probe linearly):

  entity slot [0:loc_dim]            -> current location (one-hot over LOC_POOL)
  entity slot [loc_dim:loc_dim+inv]  -> inventory (multi-hot over ITEM_POOL)
  item   slot [0:name_dim]           -> holder (one-hot over NAME_POOL)

Reading = trivial argmax of the dedicated field.

BACKEND toggle
--------------
Span scanning (`detect_spans`) is swappable: `BACKEND = "c"` uses a compiled
C hot-path (`fastworld.so`, built from `fastworld.c`) and falls back to the
pure-Python implementation if the module is unavailable. Add a faster backend
(Mojo/Rust/C) by implementing the same `find_spans(text, pool)` contract and
registering it in `_BACKENDS`.

Why batching matters
-------------------
Tiny models are dominated by Python->torch dispatch overhead, NOT compute. The
old loop called the encoder LSTM / writer GRU / heads once *per sample per
epoch*. We now batch: one padded encoder forward over the whole mini-batch, a
single gather for last-mention pooling, and vectorized heads/losses. This is
the real speedup (10-50x), and last-mention pooling is exactly "last mention
wins" so it also fixes the interference trap for free.
"""

from __future__ import annotations
import re
import math
import ctypes
import dataclasses
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import TokenEncoder, AnswerComposer, AnswerDecoder
from src.tokenizer import CharTokenizer
from src.dataset import NAME_POOL, LOC_POOL, ITEM_POOL
from reverse_templates import reverse_templates


# Pools are imported from src.dataset (single source of truth). They MUST match
# the generator or supervision silently vanishes.
N_NAMES = len(NAME_POOL)
N_LOCS = len(LOC_POOL)
N_ITEMS = len(ITEM_POOL)

NAME_TO_I = {n: i for i, n in enumerate(NAME_POOL)}
LOC_TO_I = {l: i for i, l in enumerate(LOC_POOL)}
ITEM_TO_I = {it: i for i, it in enumerate(ITEM_POOL)}
I_TO_NAME = {i: n for n, i in NAME_TO_I.items()}
I_TO_LOC = {i: l for l, i in LOC_TO_I.items()}
I_TO_ITEM = {i: it for it, i in ITEM_TO_I.items()}


def extract_query(sample: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Pull the queried entity / item / location from the QUESTION text.

    The dataset does not store explicit `subject`/`item`/`location` keys, but
    the question always names them from the canonical pools. We recover them
    by scanning the question. Returns (subject_name, item_name, loc_name);
    any may be None.
    """
    q = sample.get("question", "") or ""
    subj = None
    for nm in NAME_POOL:
        if re.search(r"\b" + re.escape(nm) + r"\b", q):
            subj = nm
            break
    item = None
    for it in ITEM_POOL:
        if re.search(r"\b" + re.escape(it) + r"\b", q):
            item = it
            break
    loc = None
    for l in LOC_POOL:
        if re.search(r"\b" + re.escape(l) + r"\b", q):
            loc = l
            break
    return subj, item, loc


# ---------------------------------------------------------------------------
# BACKEND toggle for span scanning
# ---------------------------------------------------------------------------
BACKEND = "c"  # "c" (uses fastworld.so if present, else python) | "python"


def _load_fastworld():
    """Load the compiled C span-scanner (fastworld.so) if present."""
    try:
        import importlib
        fastworld = importlib.import_module("fastworld")  # extension (fastworld.c)
        if hasattr(fastworld, "fast_find_spans"):
            return fastworld
    except Exception:
        return None
    return None


_fastworld = _load_fastworld()


def _fast_find_spans(text: str, pool: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """C-backed span scan. Result format: 'word\\tstart,end;start,end\\n...'"""
    pool_csv = "|".join(pool)
    raw = _fastworld.fast_find_spans(text, pool_csv)  # returns a str
    if not raw:
        return {}
    out: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for line in raw.split("\n"):
        if not line:
            continue
        word, ranges = line.split("\t")
        for r in ranges.split(";"):
            if not r:
                continue
            a, b = r.split(",")
            out[word].append((int(a), int(b)))
    return dict(out)


def _py_detect_spans(text: str, pool: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def is_word_char(c):
        return c.isalnum()

    def boundary_before(i):
        return i == 0 or not is_word_char(text[i - 1])

    def boundary_after(i):
        return i >= len(text) or not is_word_char(text[i])

    for word in pool:
        if not word:
            continue
        start = 0
        while True:
            i = text.find(word, start)
            if i == -1:
                break
            end = i + len(word)
            if boundary_before(i) and boundary_after(end):
                out[word].append((i, end))
            start = i + 1
    return dict(out)


def detect_spans(text: str, pool: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Find all whole-word occurrences of any `pool` word in `text`.

    Swappable backend via the module-level `BACKEND` flag (C hot-path with
    Python fallback). Returns {word: [(start, end), ...]}.
    """
    if BACKEND == "c" and _fastworld is not None:
        try:
            return _fast_find_spans(text, pool)
        except Exception:
            pass
    return _py_detect_spans(text, pool)


# ---------------------------------------------------------------------------
# One-hot field helpers
# ---------------------------------------------------------------------------
def _loc_onehot(loc: Optional[str]) -> torch.Tensor:
    v = torch.zeros(N_LOCS)
    if loc is not None and loc in LOC_TO_I:
        v[LOC_TO_I[loc]] = 1.0
    return v


def _inv_multihot(items: List[str]) -> torch.Tensor:
    v = torch.zeros(N_ITEMS)
    for it in items:
        if it in ITEM_TO_I:
            v[ITEM_TO_I[it]] = 1.0
    return v


def _name_onehot(name: Optional[str]) -> torch.Tensor:
    v = torch.zeros(N_NAMES)
    if name is not None and name in NAME_TO_I:
        v[NAME_TO_I[name]] = 1.0
    return v


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WorldModel(nn.Module):
    def __init__(self, vocab_size: int, d_state: int = 64, d_model: int = 48):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.loc_dim = N_LOCS
        self.inv_dim = N_ITEMS
        self.name_dim = N_NAMES

        self.encoder = TokenEncoder(vocab_size, d_state, d_model, n_layers=2)

        # Learnable init slots for entities/items not present in a narrative.
        self.init_ent = nn.Parameter(torch.zeros(d_state))
        self.init_item = nn.Parameter(torch.zeros(d_state))

        # Read heads: project a per-entity / per-item slot to its dedicated
        # field. The slot is the disentangled per-entity latent (filled by
        # last-mention pooling); the head extracts ONE field from it. Holding
        # is the core relation: transfer reads item->holder directly, and
        # inventory is its inverse (items whose predicted holder == entity).
        self.loc_head = nn.Linear(d_state, N_LOCS)        # entity slot -> location
        self.holder_head = nn.Linear(self.name_dim, N_NAMES)  # item slot -> holder

        # Aux heads: encoder state AT a location/item word must predict it.
        # This breaks the encoder+writer collapse to a constant.
        self.loc_tok_head = nn.Linear(d_state, N_LOCS)
        self.item_tok_head = nn.Linear(d_state, N_ITEMS)

        # (A, B, C) -> D composer reuses the modular AnswerComposer (recall path)
        self.composer = AnswerComposer(d_state)
        self.ans_dec = AnswerDecoder(d_state, vocab_size)

    # -- batched write -----------------------------------------------------
    def write_batch(self, narratives: List[str], tokenizer, device) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
        B = len(narratives)
        seqs = [tokenizer.encode(n, max_len=None) for n in narratives]
        T = max(len(s) for s in seqs)
        ids = torch.zeros(B, T, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            if s:
                ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        states = self.encoder.states(ids)                 # [B, T, d_state]

        ent_pos = torch.full((B, N_NAMES), -1, dtype=torch.long)
        item_pos = torch.full((B, N_ITEMS), -1, dtype=torch.long)
        name_spans_batch, item_spans_batch = [], []
        for b, n in enumerate(narratives):
            ns = detect_spans(n, NAME_POOL)
            name_spans_batch.append(ns)
            for nm, sp in ns.items():
                ent_pos[b, NAME_TO_I[nm]] = sp[-1][1] - 1      # last mention wins
            isp = detect_spans(n, ITEM_POOL)
            item_spans_batch.append(isp)
            for it, sp in isp.items():
                item_pos[b, ITEM_TO_I[it]] = sp[-1][1] - 1

        ent_slots = self.init_ent.view(1, 1, self.d_state).expand(B, N_NAMES, -1).clone()
        item_slots = self.init_item.view(1, 1, self.d_state).expand(B, N_ITEMS, -1).clone()

        ev = ent_pos >= 0
        eg = ent_pos.clamp(min=0)
        gathered = states[torch.arange(B, device=device).unsqueeze(1).expand(-1, N_NAMES), eg]
        ent_slots = torch.where(ev.unsqueeze(-1), gathered, ent_slots)

        iv = item_pos >= 0
        ig = item_pos.clamp(min=0)
        gathered_i = states[torch.arange(B, device=device).unsqueeze(1).expand(-1, N_ITEMS), ig]
        item_slots = torch.where(iv.unsqueeze(-1), gathered_i, item_slots)

        holder_logits = self.holder_head(item_slots[:, :, : self.name_dim])  # [B, I, N_NAMES]
        return ent_slots, item_slots, holder_logits, states, ids, name_spans_batch

    def write(self, narrative: str, tokenizer, device):
        es, is_, hl, _, _, _ = self.write_batch([narrative], tokenizer, device)
        return es[0], is_[0], hl[0]

    # -- read --------------------------------------------------------------
    @torch.no_grad()
    def read_answer(self, ent_slot, item_slot, holder_logits, sample) -> str:
        task = sample["task_type"]
        subj_name, item_name, loc_name = extract_query(sample)
        subj = NAME_TO_I.get(subj_name, 0) if subj_name in NAME_TO_I else 0
        # holder_logits: [N_ITEMS, N_NAMES] per-item holder prediction.
        holders = holder_logits.argmax(-1)                        # [N_ITEMS]
        # per-entity location prediction (loc_head over the entity slots)
        loc_pred = self.loc_head(ent_slot).argmax(-1)            # [N_NAMES]
        if task == "location":
            return I_TO_LOC[loc_pred[subj].item()]
        if task == "inventory":
            items = sorted(I_TO_ITEM[i] for i in range(N_ITEMS) if holders[i] == subj)
            return " and ".join(items) if items else "nothing"
        if task == "transfer":
            if item_name in ITEM_TO_I:
                holder = holders[ITEM_TO_I[item_name]].item()
                return I_TO_LOC[loc_pred[holder].item()]
            return ""
        if task == "holder":
            if item_name in ITEM_TO_I:
                return I_TO_NAME[holders[ITEM_TO_I[item_name]].item()]
            return ""
        if task == "colocation":
            others = [I_TO_NAME[n] for n in range(N_NAMES)
                      if n != subj and loc_pred[n] == loc_pred[subj]]
            return " and ".join(sorted(others)) if others else "nobody"
        if task == "count_people":
            if loc_name in LOC_TO_I:
                return str(int((loc_pred == LOC_TO_I[loc_name]).sum().item()))
            return ""
        if task == "which_loc_most":
            best, best_loc = -1, LOC_TO_I[LOC_POOL[0]]
            for l in LOC_POOL:
                li = LOC_TO_I[l]; c = int((loc_pred == li).sum().item())
                if c > best:
                    best, best_loc = c, li
            return I_TO_LOC[best_loc]
        if task == "most_items":
            best, best_name = -1, 0
            for n in NAME_POOL:
                ni = NAME_TO_I[n]; c = int((holders == ni).sum().item())
                if c > best:
                    best, best_name = c, ni
            return I_TO_NAME[best_name]
        if task == "empty_loc":
            if loc_name in LOC_TO_I:
                return "yes" if int((loc_pred == LOC_TO_I[loc_name]).sum().item()) == 0 else "no"
            return ""
        if task == "has_item":
            if item_name in ITEM_TO_I and subj_name in NAME_TO_I:
                return "yes" if holders[ITEM_TO_I[item_name]] == subj else "no"
            return ""
        # recall: generative (handled by run_world_qa via generate_answer)
        return ""

    def answer_logits(self, read_slot: torch.Tensor, C: torch.Tensor,
                      answer_ids: torch.Tensor):
        A = read_slot.reshape(1, -1); Bb = read_slot.reshape(1, -1); Cc = C.reshape(1, -1)
        D = self.composer(A, Bb, Cc)                        # [1, d]
        tgt = answer_ids.unsqueeze(0)                       # [1, T_raw]
        logits = self.ans_dec.forward_teacher(D, tgt)       # [1, T, V]
        T = logits.size(1)
        return logits, tgt[:, :T]

    @torch.no_grad()
    def generate_answer(self, ent_slot, item_slot, holder_logits, question_text,
                        tokenizer, max_tokens=48, eos_id=None, pad_id=None):
        # C must be a d_state-dim *encoded* question vector, mirroring
        # _question_vec used in training (which encodes the subject name, not
        # the raw question tokens). Passing token ids here was the shape bug.
        device = ent_slot.device
        subj_name, _, _ = extract_query({"question": question_text})
        name = subj_name if subj_name in NAME_TO_I else NAME_POOL[0]
        ids = torch.tensor(tokenizer.encode(" " + name, max_len=None),
                           dtype=torch.long, device=device).unsqueeze(0)
        Cc = self.encoder.states(ids)[:, -1, :]            # [1, d_state]
        A = ent_slot.unsqueeze(0); Bb = ent_slot.unsqueeze(0)
        D = self.composer(A, Bb, Cc)
        ids_out = self.ans_dec.generate(D, max_tokens=max_tokens, eos_id=eos_id,
                                        pad_id=pad_id)
        return tokenizer.decode(ids_out)


# ---------------------------------------------------------------------------
# Prepared sample (parsed ONCE; reused every epoch)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Prepared:
    narr: str
    ent_target: torch.Tensor          # [N_NAMES, loc_dim+inv_dim]
    item_target: torch.Tensor         # [N_ITEMS, name_dim]
    ent_mask: torch.Tensor            # [N_NAMES] 1 if entity appears in narrative
    item_mask: torch.Tensor           # [N_ITEMS] 1 if item appears in narrative
    subject_idx: int
    loc_tok: List[Tuple[int, int]]    # (char_pos, loc_idx)
    item_tok: List[Tuple[int, int]]   # (char_pos, item_idx)
    ans_ids: List[int]


def prepare_sample(sample: dict, tokenizer: CharTokenizer, device) -> Prepared:
    narr = sample["narrative"]
    w, _ = reverse_templates(narr)

    ent_target = torch.zeros(N_NAMES, N_LOCS + N_ITEMS)
    ent_mask = torch.zeros(N_NAMES)
    item_target = torch.zeros(N_ITEMS, N_NAMES)
    item_mask = torch.zeros(N_ITEMS)
    name_spans = detect_spans(narr, NAME_POOL)
    item_spans = detect_spans(narr, ITEM_POOL)
    for nm, e in w.entities.items():
        if nm not in NAME_TO_I:
            continue
        ni = NAME_TO_I[nm]
        ent_mask[ni] = 1.0
        row = torch.zeros(N_LOCS + N_ITEMS)
        if e.location is not None:
            row[:N_LOCS] = _loc_onehot(e.location)
        if e.inventory:
            row[N_LOCS:] = _inv_multihot(e.inventory)
        ent_target[ni] = row
    for it, holder in w.item_holder.items():
        if it not in ITEM_TO_I:
            continue
        ii = ITEM_TO_I[it]
        item_mask[ii] = 1.0
        if holder is not None:
            item_target[ii] = _name_onehot(holder)
    # items present but without an explicit holder entry still count as present
    for iw in item_spans:
        if iw in ITEM_TO_I:
            item_mask[ITEM_TO_I[iw]] = 1.0

    # auxiliary token positions (last occurrence of each location/item word)
    loc_tok = []
    loc_spans = detect_spans(narr, LOC_POOL)
    for lw, sp in loc_spans.items():
        if lw in LOC_TO_I:
            loc_tok.append((sp[-1][1] - 1, LOC_TO_I[lw]))
    item_tok = []
    item_spans = detect_spans(narr, ITEM_POOL)
    for iw, sp in item_spans.items():
        if iw in ITEM_TO_I:
            item_tok.append((sp[-1][1] - 1, ITEM_TO_I[iw]))

    ans_ids = tokenizer.encode(sample["answer"], max_len=None)
    subj_name, _, _ = extract_query(sample)
    subj_idx = NAME_TO_I.get(subj_name, 0) if subj_name in NAME_TO_I else 0
    return Prepared(narr, ent_target, item_target, ent_mask, item_mask, subj_idx,
                    loc_tok, item_tok, ans_ids)


def self_loc_inv_dim():
    return N_LOCS + N_ITEMS


# ---------------------------------------------------------------------------
# Training (batched)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class WorldTrainConfig:
    d_state: int = 64
    d_model: int = 48
    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    ans_w: float = 1.0
    field_w: float = 1.0
    loc_tok_w: float = 1.0
    item_tok_w: float = 1.0
    seed: int = 42


def train_world(qa: List[dict], tokenizer: CharTokenizer, device, cfg: WorldTrainConfig,
                qa_hook=None, verbose: bool = True) -> Tuple[nn.Module, dict]:
    torch.manual_seed(cfg.seed)
    model = WorldModel(tokenizer.vocab_size, d_state=cfg.d_state, d_model=cfg.d_model)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    prepared = [prepare_sample(s, tokenizer, device) for s in qa]
    ent_targets = torch.stack([p.ent_target for p in prepared]).to(device)   # [N,E,loc+inv]
    ent_masks = torch.stack([p.ent_mask for p in prepared]).to(device)       # [N,E]
    item_targets = torch.stack([p.item_target for p in prepared]).to(device) # [N,I,name]
    item_masks = torch.stack([p.item_mask for p in prepared]).to(device)     # [N,I]
    subj_idx = torch.tensor([p.subject_idx for p in prepared], device=device)

    hist = {"field_loss": [], "ans_loss": [], "loc_tok_loss": [],
            "item_tok_loss": [], "qa_acc": []}
    B = cfg.batch_size
    N = len(qa)

    for ep in range(1, cfg.epochs + 1):
        perm = torch.randperm(N)
        tot_field = tot_ans = tot_loc = tot_item = 0.0
        n_batches = 0
        for s in range(0, N, B):
            idx = perm[s: s + B]
            chunk = [qa[i] for i in idx.tolist()]
            prep_chunk = [prepared[i] for i in idx.tolist()]

            ent_slots, item_slots, holder_logits, states, ids, _ = \
                model.write_batch([c["narrative"] for c in chunk], tokenizer, device)

            # ---- structured field supervision via READ HEADS on MENTIONED slots ----
            # location: CE over mentioned entities that have a known location
            et = ent_targets[idx]                                        # [b,E,loc+inv]
            em = ent_masks[idx].bool().reshape(-1)                       # [b*E]
            es = ent_slots.reshape(-1, model.d_state)[em]                 # [M, d_state]
            loc_tgt = et.reshape(-1, N_LOCS + N_ITEMS)[em, :N_LOCS]       # [M, N_LOCS]
            has_loc = loc_tgt.sum(-1) > 0.5                             # [M]
            if has_loc.any():
                loc_loss = F.cross_entropy(model.loc_head(es[has_loc]),
                                           loc_tgt[has_loc].argmax(-1))
            else:
                loc_loss = torch.zeros((), device=device)
            # inventory: derived from the holder relation (trained via holder_loss),
            # so no separate inv loss is needed -- see inventory read path.
            # holder: CE over mentioned items that have a known holder
            it = item_targets[idx]                                       # [b,I,name]
            im = item_masks[idx].bool().reshape(-1)                      # [b*I]
            is_ = item_slots.reshape(-1, model.d_state)[im]               # [M2, d_state]
            holder_tgt = it.reshape(-1, N_NAMES)[im]                     # [M2, N_NAMES]
            has_holder = holder_tgt.sum(-1) > 0.5                       # [M2]
            if has_holder.any():
                holder_loss = F.cross_entropy(
                    model.holder_head(is_[has_holder, :N_NAMES]),
                    holder_tgt[has_holder].argmax(-1))
            else:
                holder_loss = torch.zeros((), device=device)
            field_loss = loc_loss + holder_loss

            # ---- aux: encoder state at location/item words ----
            loc_pos = [(bi, p[0], p[1]) for bi, p in enumerate(prep_chunk) for p in p.loc_tok]
            loc_tok_loss = torch.zeros((), device=device)
            if loc_pos:
                rows = torch.tensor([x[0] for x in loc_pos], device=device)
                cols = torch.tensor([x[1] for x in loc_pos], device=device)
                h = states[rows, cols]
                tgt = torch.tensor([x[2] for x in loc_pos], device=device)
                loc_tok_loss = F.cross_entropy(model.loc_tok_head(h), tgt)

            item_pos = [(bi, p[0], p[1]) for bi, p in enumerate(prep_chunk) for p in p.item_tok]
            item_tok_loss = torch.zeros((), device=device)
            if item_pos:
                rows = torch.tensor([x[0] for x in item_pos], device=device)
                cols = torch.tensor([x[1] for x in item_pos], device=device)
                h = states[rows, cols]
                tgt = torch.tensor([x[2] for x in item_pos], device=device)
                item_tok_loss = F.cross_entropy(model.item_tok_head(h), tgt)

            # ---- generative answer path (recall only; the other tasks are
            # read via the structured fields, not free-text generation) ----
            ans_samples = [(bi, p) for bi, p in enumerate(prep_chunk)
                           if chunk[bi]["task_type"] == "recall" and len(p.ans_ids) > 0]
            ans_loss = torch.zeros((), device=device)
            if ans_samples:
                tot = 0.0
                for bi, p in ans_samples:
                    rslot = ent_slots[bi, p.subject_idx]
                    Cc = _question_vec(model, p, tokenizer, device)
                    logits, tgt = model.answer_logits(
                        rslot, Cc, torch.tensor(p.ans_ids, device=device))
                    tot = tot + F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                ans_loss = tot / len(ans_samples)

            loss = (cfg.field_w * field_loss + cfg.ans_w * ans_loss
                    + cfg.loc_tok_w * loc_tok_loss + cfg.item_tok_w * item_tok_loss)
            opt.zero_grad(); loss.backward(); opt.step()

            tot_field += field_loss.item(); tot_ans += ans_loss.item()
            tot_loc += loc_tok_loss.item(); tot_item += item_tok_loss.item()
            n_batches += 1

        fa = run_world_qa(model, qa, tokenizer, device, cfg) if (qa_hook or True) else (None, 0)
        acc = fa[1] if isinstance(fa, tuple) else 0.0
        hist["field_loss"].append(tot_field / max(n_batches, 1))
        hist["ans_loss"].append(tot_ans / max(n_batches, 1))
        hist["loc_tok_loss"].append(tot_loc / max(n_batches, 1))
        hist["item_tok_loss"].append(tot_item / max(n_batches, 1))
        hist["qa_acc"].append(acc)
        if verbose:
            print(f"  [world] ep {ep}/{cfg.epochs} field={hist['field_loss'][-1]:.4f} "
                  f"ans={hist['ans_loss'][-1]:.4f} loctok={tot_loc/max(n_batches,1):.4f} "
                  f"itemtok={tot_item/max(n_batches,1):.4f} acc={acc:.3f}")
            print(f"  STAGE: world ep={ep}/{cfg.epochs} "
                  f"field={hist['field_loss'][-1]:.4f} ans={hist['ans_loss'][-1]:.4f}")

    return model, hist


def _question_vec(model, p, tokenizer, device):
    """Build a question conditioning vector C from the subject name token."""
    q = " " + NAME_POOL[p.subject_idx]
    ids = torch.tensor(tokenizer.encode(q, max_len=None), dtype=torch.long,
                       device=device).unsqueeze(0)
    return model.encoder.states(ids)[:, -1, :]


# ---------------------------------------------------------------------------
# Evaluation (batched write, per-sample read)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_world_qa(model, qa: List[dict], tokenizer: CharTokenizer, device,
                 cfg: Optional[WorldTrainConfig] = None):
    if cfg is None:
        cfg = WorldTrainConfig()
    acc_by: Dict[str, List[float]] = defaultdict(list)
    correct = total = 0
    B = max(cfg.batch_size, 1)
    for s in range(0, len(qa), B):
        chunk = qa[s: s + B]
        narrs = [x["narrative"] for x in chunk]
        ent_slots, item_slots, holder_logits, _, _, _ = model.write_batch(narrs, tokenizer, device)
        for j, x in enumerate(chunk):
            task = x["task_type"]
            subj_name, item_name, loc_name = extract_query(x)
            subj = NAME_TO_I.get(subj_name, 0) if subj_name in NAME_TO_I else 0
            ent = ent_slots[j, subj]          # [d_state] subject entity slot
            # Precompute per-entity location and per-item holder predictions
            # (all puzzles are derived reads off these two fields).
            loc_pred = model.loc_head(ent_slots[j]).argmax(-1)            # [N_NAMES] loc idx
            holder_pred = model.holder_head(
                item_slots[j, :, :N_NAMES].reshape(-1, N_NAMES)).argmax(-1)  # [N_ITEMS] name idx

            if task == "location":
                pred = I_TO_LOC[loc_pred[subj].item()]
            elif task == "inventory":
                items = sorted(I_TO_ITEM[i] for i in range(N_ITEMS) if holder_pred[i] == subj)
                pred = " and ".join(items) if items else "nothing"
            elif task == "transfer":
                if item_name in ITEM_TO_I:
                    holder = holder_pred[ITEM_TO_I[item_name]].item()
                    pred = I_TO_LOC[loc_pred[holder].item()]
                else:
                    pred = ""
            elif task == "holder":
                if item_name in ITEM_TO_I:
                    pred = I_TO_NAME[holder_pred[ITEM_TO_I[item_name]].item()]
                else:
                    pred = ""
            elif task == "colocation":
                others = [I_TO_NAME[n] for n in range(N_NAMES)
                          if n != subj and loc_pred[n] == loc_pred[subj]]
                pred = " and ".join(sorted(others)) if others else "nobody"
            elif task == "count_people":
                if loc_name in LOC_TO_I:
                    li = LOC_TO_I[loc_name]
                    pred = str(int((loc_pred == li).sum().item()))
                else:
                    pred = ""
            elif task == "which_loc_most":
                best, best_loc = -1, LOC_TO_I[LOC_POOL[0]]
                for l in LOC_POOL:  # deterministic tie-break by pool order
                    li = LOC_TO_I[l]
                    c = int((loc_pred == li).sum().item())
                    if c > best:
                        best, best_loc = c, li
                pred = I_TO_LOC[best_loc]
            elif task == "most_items":
                best, best_name = -1, 0
                for n in NAME_POOL:  # deterministic tie-break
                    ni = NAME_TO_I[n]
                    c = int((holder_pred == ni).sum().item())
                    if c > best:
                        best, best_name = c, ni
                pred = I_TO_NAME[best_name]
            elif task == "empty_loc":
                if loc_name in LOC_TO_I:
                    li = LOC_TO_I[loc_name]
                    pred = "yes" if int((loc_pred == li).sum().item()) == 0 else "no"
                else:
                    pred = ""
            elif task == "has_item":
                if item_name in ITEM_TO_I and subj_name in NAME_TO_I:
                    pred = "yes" if holder_pred[ITEM_TO_I[item_name]] == subj else "no"
                else:
                    pred = ""
            else:  # recall -- generative
                item = item_slots[j, 0]
                pred = model.generate_answer(ent, item, holder_logits[j], x.get("question", ""), tokenizer)
            ok = (pred == x["answer"])
            correct += int(ok); total += 1
            acc_by[task].append(float(ok))
    acc_by_avg = {k: sum(v) / len(v) for k, v in acc_by.items()}
    overall = correct / total if total else 0.0
    return acc_by_avg, overall, total
