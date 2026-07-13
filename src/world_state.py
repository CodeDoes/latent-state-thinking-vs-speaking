"""
Structured World-State memory -- the architectural response to the complexity
that `reverse_templates.py` exposed.

WHAT reverse_templates PROVED
-----------------------------
The latent state must simultaneously track, for every entity:
    location, inventory, first_password, password_overridden
and, for every item, its current_holder. That is a *structured* table, not a
single pooled vector. The old pipeline (`make_B(A) -> B`) tries to squeeze one
specific fact out of one pooled narrative vector and diagnostically COLLAPSES to
the mean (MSE ~ variance floor) because a single vector can't hold all of them.

THIS MODULE
-----------
A `WorldModel` keeps an **explicit slot table**: one d_state-vector per entity
(in `NAME_POOL`) and one per item (in `ITEM_POOL`). The narrative is written into
these slots by a content-addressed `SlotWriter` (find each name/item's mention
spans in the text, pool the token states there, refine). A `HolderHead` maps each
item slot -> the entity that currently holds it.

TRAINING SIGNAL = reverse_templates AS TEACHER
---------------------------------------------
For every training sample we parse the narrative with `reverse_templates` to get
the *ground-truth* structured world. We then supervise each slot to match a
teacher slot = the encoder's own state of a canonical description of that
entity/item (so the slot is forced to *contain* the right structured fact, not
just to spell tokens). On top of that we train the read path
(slot -> composer -> AnswerDecoder) with the real answer string. This is dense,
fact-level supervision -- exactly the state the analysis says is necessary -- and
it is far stronger than the weak single-vector make_B regression.

Read path at inference:
    location / inventory / recall : detect target entity in question -> read slot
    transfer                      : detect item -> HolderHead -> holder entity slot
    then D = composer(read_slot, read_slot, C);  answer = AnswerDecoder(D).

This file is self-contained and integrates with bench.py via the `world` key.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import TokenEncoder, AnswerComposer, AnswerDecoder
from src.tokenizer import CharTokenizer


# ---- Fixed pools (identical to dataset.World defaults). The model only ever
#      sees these tokens, so content-addressed slot assignment by string scan
#      is legitimate -- it is exactly what a reader does when they see "John". ----
NAME_POOL = ["John", "Mary", "Alex", "Sam", "Emma", "Leo", "Zoe", "Max", "Lily", "Tom"]
LOC_POOL  = ["kitchen", "bedroom", "garden", "garage", "bathroom",
             "living room", "office", "basement", "attic", "hallway"]
ITEM_POOL = ["apple", "book", "key", "phone", "cup", "pen",
             "wallet", "watch", "bag", "umbrella"]
N_NAMES = len(NAME_POOL)
N_ITEMS = len(ITEM_POOL)
N_LOCS  = len(LOC_POOL)
NAME_TO_I = {n: i for i, n in enumerate(NAME_POOL)}
ITEM_TO_I = {i: k for k, i in enumerate(ITEM_POOL)}
LOC_TO_I = {l: i for i, l in enumerate(LOC_POOL)}
# reverse maps
I_TO_NAME = NAME_POOL
I_TO_ITEM = ITEM_POOL
I_TO_LOC = LOC_POOL


# ---------------------------------------------------------------------------
# reverse_templates import (lives at repo root; be robust to import path)
# ---------------------------------------------------------------------------
def _import_reverse():
    try:
        from reverse_templates import reverse_templates
        return reverse_templates
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from reverse_templates import reverse_templates
        return reverse_templates


# ---------------------------------------------------------------------------
# Span detection (char index == token index at char-level, max_len=None)
# ---------------------------------------------------------------------------

def detect_spans(text: str, pool: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Return {token: [(start, end), ...]} char-span (end exclusive) for each
    member of `pool` found in `text` (case-sensitive; data is Titlecase)."""
    out: Dict[str, List[Tuple[int, int]]] = {}
    for tok in pool:
        spans = []
        start = text.find(tok)
        while start != -1:
            spans.append((start, start + len(tok)))
            start = text.find(tok, start + len(tok))
        if spans:
            out[tok] = spans
    return out


# ---------------------------------------------------------------------------
# Slot writer: narrative token states -> per-entity / per-item slot table
# ---------------------------------------------------------------------------

class SlotWriter(nn.Module):
    """Content-addressed write: pool each entity/item's mentions IN NARRATIVE
    ORDER through a GRU, so the *most recent* mention wins.

    This is essential: the dataset deliberately uses interfering mentions and
    moves, so the correct state is the LAST one (e.g. 'X moved from A to B' must
    override the earlier 'X was in A'). Mean-pooling all mentions would blend
    old+new and break location tracking -- the exact 'last-mention' trap the
    data was built to expose. A GRU over mentions in order makes the final
    hidden state reflect the latest state.

    For an entity with no mentions, the slot stays near-zero (loss skips it).
    """

    def __init__(self, d_state: int, n_slots: int, mention_window: int = 24):
        super().__init__()
        self.d_state = d_state
        self.n_slots = n_slots
        self.mention_window = mention_window
        # per-mention context pooling
        self.pool_mlp = nn.Sequential(
            nn.Linear(d_state, d_state), nn.SiLU(), nn.LayerNorm(d_state))
        # recurrent summarize of mentions in order (last wins)
        self.gru = nn.GRU(d_state, d_state, num_layers=1, batch_first=True)
        # final refine
        self.refine = nn.Sequential(
            nn.Linear(d_state, d_state), nn.SiLU(), nn.LayerNorm(d_state),
            nn.Linear(d_state, d_state),
        )

    def forward(self, states: torch.Tensor, spans: Dict[str, List[Tuple[int, int]]],
                pool_to_i: Dict[str, int]) -> torch.Tensor:
        """states: [T, d] ; returns slots [n_slots, d]."""
        T, d = states.shape
        slots = states.new_zeros(self.n_slots, d)
        w = self.mention_window
        for tok, sp in spans.items():
            idx = pool_to_i.get(tok)
            if idx is None or idx >= self.n_slots:
                continue
            vecs = []
            for (s, e) in sp:                      # sp is in narrative order
                lo = max(0, s - 1)
                hi = min(T, e + w)
                if hi > lo:
                    vecs.append(states[lo:hi].mean(dim=0))
            if not vecs:
                continue
            seq = self.pool_mlp(torch.stack(vecs, dim=0)).unsqueeze(0)   # [1, M, d]
            out, h = self.gru(seq)              # h: [1, 1, d]
            slots[idx] = self.refine(h.squeeze(0).squeeze(0))
        return slots


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """Structured world-state model: narrative -> slot table -> read -> answer.

    Latent state = an explicit table of per-entity and per-item vectors -- the
    structured world that reverse_templates showed the model must track.
    """

    def __init__(self, vocab_size: int, d_state: int = 256, d_model: int = 128):
        super().__init__()
        self.d_state = d_state
        self.encoder = TokenEncoder(vocab_size, d_state=d_state, d_model=d_model)
        self.ent_writer = SlotWriter(d_state, N_NAMES)
        self.item_writer = SlotWriter(d_state, N_ITEMS)
        # DECOMPOSED SLOT FIELDS (the user's insight: an SSM tracking location
        # only needs (entity_index, current_location_at_index) -- a *disentangled*
        # record, not one entangled vector we then try to linearly read a
        # location out of). So each entity slot's first dims are EXPLICIT fields:
        #   [0:loc_dim]            -> current location (one-hot over LOC_POOL)
        #   [loc_dim:loc_dim+inv]  -> inventory (multi-hot over ITEM_POOL)
        # and each item slot's first name_dim dims are its holder (one-hot over
        # NAME_POOL). Reading = trivial argmax of the dedicated field -- no
        # fragile linear probe over an entangled vector.
        self.loc_dim = N_LOCS
        self.inv_dim = N_ITEMS
        self.name_dim = N_NAMES
        # Auxiliary: the encoder's state AT a location word must predict that
        # location. This breaks the encoder+writer degeneracy where both collapse
        # to a constant (it's cheaper to output a fixed location than to learn to
        # represent "bedroom" vs "kitchen"). Forces the encoder to actually
        # encode location words.
        self.loc_tok_head = nn.Linear(d_state, N_LOCS)
        self.item_tok_head = nn.Linear(d_state, N_ITEMS)
        # (A, B, C) -> D composer reuses the modular AnswerComposer (recall path)
        self.composer = AnswerComposer(d_state)
        self.ans_dec = AnswerDecoder(d_state, vocab_size)

    @property
    def eos_id(self):
        return self.ans_dec  # placeholder; set externally

    # --- encode + write ---
    def write(self, narrative_text: str, tokenizer: CharTokenizer,
              device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = torch.tensor(
            tokenizer.encode(narrative_text, max_len=None), dtype=torch.long,
            device=device).unsqueeze(0)                              # [1, T]
        states = self.encoder.states(ids)[0]                        # [T, d]
        ent_spans = detect_spans(narrative_text, NAME_POOL)
        item_spans = detect_spans(narrative_text, ITEM_POOL)
        ent_slots = self.ent_writer(states, ent_spans, NAME_TO_I)   # [N_NAMES, d]
        item_slots = self.item_writer(states, item_spans, ITEM_TO_I)  # [N_ITEMS, d]
        # holder_logits placeholder (holder is now read as argmax over the
        # item slot's dedicated name_dim field; kept for API compatibility)
        holder_logits = item_slots.new_zeros(N_ITEMS, N_NAMES)
        return ent_slots, item_slots, holder_logits

    # --- read a target slot given the question ---
    def read_slot(self, ent_slots, item_slots, holder_logits, question_text,
                  tokenizer, device, parsed=None):
        """Return the slot vector to decode, plus the entity/item index used.

        `parsed` (optional ReverseWorld) lets us use the GROUND-TRUTH holder /
        target at training time; at inference it is None and we use predictions.
        """
        # Encode the question to condition the decoder.
        qids = torch.tensor(tokenizer.encode(question_text or " ", max_len=None),
                            dtype=torch.long, device=device).unsqueeze(0)
        C = self.encoder.state_of(qids)[0]                          # [d]
        # Which entity is the target?
        q_names = detect_spans(question_text, NAME_POOL)
        q_items = detect_spans(question_text, ITEM_POOL)
        if q_names:
            target_name = next(iter(q_names))
            idx = NAME_TO_I[target_name]
            return ent_slots[idx], C, ("entity", idx)
        if q_items:
            item_name = next(iter(q_items))
            iidx = ITEM_TO_I[item_name]
            if parsed is not None:
                holder = parsed.item_holder.get(item_name)
                hidx = NAME_TO_I.get(holder) if holder else None
            else:
                hidx = int(holder_logits[iidx].argmax().item())
            if hidx is not None:
                return ent_slots[hidx], C, ("transfer", hidx)
            # fallback: can't resolve holder
            return ent_slots.new_zeros(self.d_state), C, ("transfer", -1)
        return ent_slots.new_zeros(self.d_state), C, ("none", -1)

    # --- compose D and decode answer logits (teacher-forced) ---
    def answer_logits(self, read_slot: torch.Tensor, C: torch.Tensor,
                      answer_ids: torch.Tensor):
        # D = composer(A=read_slot, B=read_slot, C=question)
        A = read_slot.unsqueeze(0); B = read_slot.unsqueeze(0); Cc = C.unsqueeze(0)
        D = self.composer(A, B, Cc)                                 # [1, d]
        tgt = answer_ids.unsqueeze(0)                              # [1, T_raw]
        logits = self.ans_dec.forward_teacher(D, tgt)              # [1, T, V]
        T = logits.size(1)
        # forward_teacher truncates logits to max_tokens internally; return the
        # matching truncated target so the caller's CE target aligns exactly.
        return logits, tgt[:, :T]

    @torch.no_grad()
    def generate_answer(self, ent_slots, item_slots, holder_logits, question_text,
                        tokenizer, max_tokens=24, eos_id=None, pad_id=None, parsed=None):
        read_slot, C, _ = self.read_slot(ent_slots, item_slots, holder_logits,
                                         question_text, tokenizer, ent_slots.device, parsed)
        A = read_slot.unsqueeze(0); B = read_slot.unsqueeze(0); Cc = C.unsqueeze(0)
        D = self.composer(A, B, Cc)
        return self.ans_dec.generate(D, max_tokens=max_tokens, eos_id=eos_id, pad_id=pad_id)

    @torch.no_grad()
    def read_answer(self, ent_slots, item_slots, holder_logits, sample,
                    tokenizer, max_new=24):
        """Dispatch on task. Closed-vocab reads are TRIVIAL argmax over the
        slot's dedicated field dims (location / inventory / holder) -- no
        fragile probe over an entangled vector."""
        q = sample.get("question", "")
        task = sample["task_type"]
        q_names = detect_spans(q, NAME_POOL)
        q_items = detect_spans(q, ITEM_POOL)

        if task == "location" and q_names:
            idx = NAME_TO_I[next(iter(q_names))]
            li = int(ent_slots[idx][:self.loc_dim].argmax().item())
            return I_TO_LOC[li]
        if task == "transfer" and q_items:
            iidx = ITEM_TO_I[next(iter(q_items))]
            hidx = int(item_slots[iidx][:self.name_dim].argmax().item())
            if 0 <= hidx < N_NAMES:
                li = int(ent_slots[hidx][:self.loc_dim].argmax().item())
                return I_TO_LOC[li]
            return ""
        if task == "inventory" and q_names:
            idx = NAME_TO_I[next(iter(q_names))]
            probs = ent_slots[idx][self.loc_dim:self.loc_dim + self.inv_dim]
            items = [I_TO_ITEM[i] for i, p in enumerate(probs) if p.item() > 0.5]
            return " and ".join(items) if items else "nothing"
        if task == "recall":
            eos = tokenizer.vocab[tokenizer.eos_token]
            pad = tokenizer.vocab[tokenizer.pad_token]
            ids = self.generate_answer(ent_slots, item_slots, holder_logits, q,
                                       tokenizer, max_tokens=max_new, eos_id=eos, pad_id=pad)
            return tokenizer.decode(ids).strip()
        return ""


# ---------------------------------------------------------------------------
# Teacher target builders (ground-truth structured state, from reverse_templates)
# ---------------------------------------------------------------------------

def teacher_entity_slot(model: "WorldModel", ent_name: str, e, tokenizer, device
                        ) -> torch.Tensor:
    """Canonical description of one entity -> encoder state (the teacher slot)."""
    parts = [f"{ent_name} was in the {e.location}."] if e.location else []
    if e.inventory:
        parts.append(f"{ent_name} had {' and '.join(e.inventory)}.")
    if e.first_password:
        parts.append(f"The secret code for {ent_name} is {e.first_password}.")
    desc = " ".join(parts) if parts else f"{ent_name} is unknown."
    ids = torch.tensor(tokenizer.encode(desc, max_len=None), dtype=torch.long,
                       device=device).unsqueeze(0)
    return model.encoder.state_of(ids)[0].detach()                 # [d]


def teacher_item_slot(model: "WorldModel", item_name: str, holder: Optional[str],
                      tokenizer, device) -> torch.Tensor:
    if holder:
        desc = f"{holder} held the {item_name}."
    else:
        desc = f"no one held the {item_name}."
    ids = torch.tensor(tokenizer.encode(desc, max_len=None), dtype=torch.long,
                       device=device).unsqueeze(0)
    return model.encoder.state_of(ids)[0].detach()                 # [d]


# ---------------------------------------------------------------------------
# Training + eval (self-contained; integrated into bench.py via `world`)
# ---------------------------------------------------------------------------

@dataclass
class WorldTrainConfig:
    d_state: int = 256
    d_model: int = 128
    epochs: int = 20
    batch_size: int = 16
    lr: float = 3e-4
    slot_w: float = 1.0
    ans_w: float = 1.0
    field_w: float = 1.0
    loc_tok_w: float = 1.0
    item_tok_w: float = 1.0
    seed: int = 42


def _ids(text, tok):
    if not text:
        text = " "
    return torch.tensor([tok.vocab.get(c, tok.vocab[tok.unk_token]) for c in text],
                        dtype=torch.long)


def train_world(dataset, tokenizer, device, cfg: WorldTrainConfig,
                verbose: bool = True, qa_hook=None):
    """Train the structured WorldModel. Returns (model, history)."""
    reverse_templates = _import_reverse()
    torch.manual_seed(cfg.seed)
    model = WorldModel(tokenizer.vocab_size, d_state=cfg.d_state,
                       d_model=cfg.d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    eos = tokenizer.vocab[tokenizer.eos_token]
    pad = tokenizer.vocab[tokenizer.pad_token]
    hist = {"field_loss": [], "ans_loss": [], "loc_tok_loss": [],
           "item_tok_loss": [], "qa_acc": []}

    qa = [s for s in dataset if s.get("question")]
    # NOTE: the encoder is trained end-to-end here (NOT frozen). The earlier
    # frozen-encoder design collapsed to a constant location because the
    # char-LSTM's token states represented "bedroom" and "kitchen" nearly
    # identically -- so the writer could not extract location from the narrative.
    # Training the encoder lets it learn to represent location words distinctly,
    # which is what makes the decomposed slot field (current_location_at_index)
    # actually track the right place.

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot_field = tot_ans = tot_loc_tok = tot_item_tok = 0.0
        n = 0
        for s in qa:
            narr = s["narrative"]
            q = s.get("question", "")
            ans = s["answer"]
            parsed = reverse_templates(narr)
            parsed_world = parsed[0] if isinstance(parsed, tuple) else parsed

            ent_slots, item_slots, _ = model.write(narr, tokenizer, device)

            # ---- DECOMPOSED FIELD SUPERVISION (the user's insight) ----
            # The latent record per entity = (index, current_location_at_index,
            # inventory). We supervise those as EXPLICIT slot fields so reading
            # is a trivial argmax -- no fragile linear probe over an entangled
            # vector. (a) entity slot[:loc_dim] -> one-hot(location)
            #     entity slot[loc_dim:loc_dim+inv_dim] -> multi-hot(inventory)
            # (b) item   slot[:name_dim] -> one-hot(holder)
            field_loss = torch.zeros((), device=device)
            present = 0
            for name, e in parsed_world.entities.items():
                if name not in NAME_TO_I:
                    continue
                idx = NAME_TO_I[name]
                if e.location and e.location in LOC_TO_I:
                    loc_oh = torch.zeros(model.loc_dim, device=device)
                    loc_oh[LOC_TO_I[e.location]] = 1.0
                    field_loss = field_loss + F.mse_loss(
                        ent_slots[idx][:model.loc_dim], loc_oh)
                inv_mh = torch.zeros(model.inv_dim, device=device)
                for it in e.inventory:
                    if it in ITEM_TO_I:
                        inv_mh[ITEM_TO_I[it]] = 1.0
                field_loss = field_loss + F.mse_loss(
                    ent_slots[idx][model.loc_dim:model.loc_dim + model.inv_dim], inv_mh)
                present += 1
            for item, holder in parsed_world.item_holder.items():
                if item not in ITEM_TO_I or holder not in NAME_TO_I:
                    continue
                h_oh = torch.zeros(model.name_dim, device=device)
                h_oh[NAME_TO_I[holder]] = 1.0
                field_loss = field_loss + F.mse_loss(
                    item_slots[ITEM_TO_I[item]][:model.name_dim], h_oh)
                present += 1
            if present > 0:
                field_loss = field_loss / present

            # ---- AUX: encoder state AT a location word must predict it ----
            # Breaks the encoder+writer collapse to a constant location.
            loc_tok_loss = torch.zeros((), device=device)
            loc_spans = detect_spans(narr, LOC_POOL)
            if loc_spans:
                narr_ids = torch.tensor(tokenizer.encode(narr, max_len=None),
                                         dtype=torch.long, device=device).unsqueeze(0)
                narr_states = model.encoder.states(narr_ids)[0]      # [T, d]
                Tn = narr_states.size(0)
                ntok = 0
                for lw, sp in loc_spans.items():
                    li = LOC_TO_I.get(lw)
                    if li is None:
                        continue
                    pos = min(sp[-1][1] - 1, Tn - 1)   # last token of the word
                    h = narr_states[pos]
                    loc_tok_loss = loc_tok_loss + F.cross_entropy(
                        model.loc_tok_head(h.unsqueeze(0)),
                        torch.tensor([li], device=device))
                    ntok += 1
                if ntok:
                    loc_tok_loss = loc_tok_loss / ntok

            # ---- AUX: encoder state AT an item word must predict it ----
            item_tok_loss = torch.zeros((), device=device)
            item_spans = detect_spans(narr, ITEM_POOL)
            if item_spans:
                narr_ids2 = torch.tensor(tokenizer.encode(narr, max_len=None),
                                          dtype=torch.long, device=device).unsqueeze(0)
                narr_states2 = model.encoder.states(narr_ids2)[0]
                Tn2 = narr_states2.size(0)
                nit = 0
                for iw, sp in item_spans.items():
                    ii = ITEM_TO_I.get(iw)
                    if ii is None:
                        continue
                    pos = min(sp[-1][1] - 1, Tn2 - 1)
                    h = narr_states2[pos]
                    item_tok_loss = item_tok_loss + F.cross_entropy(
                        model.item_tok_head(h.unsqueeze(0)),
                        torch.tensor([ii], device=device))
                    nit += 1
                if nit:
                    item_tok_loss = item_tok_loss / nit

            # ---- generative answer path (free-text: recall passwords) ----
            read_slot, C, kind = model.read_slot(
                ent_slots, item_slots, ent_slots.new_zeros(1, 1),
                q, tokenizer, device, parsed_world)
            ans_ids = _ids(ans, tokenizer).to(device)
            tgt_ids = torch.cat([ans_ids, torch.tensor([eos], device=device)])
            logits, tgt = model.answer_logits(read_slot, C, tgt_ids)
            ans_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1))

            loss = (cfg.field_w * field_loss + cfg.ans_w * ans_loss
                    + cfg.loc_tok_w * loc_tok_loss + cfg.item_tok_w * item_tok_loss)
            opt.zero_grad(); loss.backward(); opt.step()

            tot_field += field_loss.item(); tot_ans += ans_loss.item()
            tot_loc_tok += loc_tok_loss.item(); tot_item_tok += item_tok_loss.item()
            n += 1

        hist["field_loss"].append(tot_field / max(n, 1))
        hist["ans_loss"].append(tot_ans / max(n, 1))
        hist["loc_tok_loss"].append(tot_loc_tok / max(n, 1))
        hist["item_tok_loss"].append(tot_item_tok / max(n, 1))
        if verbose:
            print(f"  [world] ep {ep}/{cfg.epochs} "
                  f"field={hist['field_loss'][-1]:.4f} ans={hist['ans_loss'][-1]:.4f} "
                  f"loctok={hist['loc_tok_loss'][-1]:.4f} itemtok={hist['item_tok_loss'][-1]:.4f}")
            print(f"  STAGE: world ep={ep}/{cfg.epochs} "
                  f"field={hist['field_loss'][-1]:.4f} ans={hist['ans_loss'][-1]:.4f}")

        if qa_hook is not None and (ep % 3 == 0 or ep == cfg.epochs):
            acc, task_acc, _ = run_world_qa(model, qa, tokenizer, device)
            hist["qa_acc"].append(acc)
            if verbose:
                print(f"  [world-qa] ep={ep} acc={acc:.3f} " +
                      " ".join(f"{t}={a:.2f}" for t, a in task_acc.items()))
                print(f"  STAGE: world-qa ep={ep} acc={acc:.3f}")

    return model, hist


@torch.no_grad()
def run_world_qa(model, dataset, tokenizer, device, max_new=48):
    """Strict exact-match on QA tasks for the WorldModel."""
    eos = tokenizer.vocab[tokenizer.eos_token]
    pad = tokenizer.vocab[tokenizer.pad_token]
    correct = total = 0
    by_task: Dict[str, List[int]] = {}
    model.eval()
    for s in dataset:
        if not s.get("question"):
            continue
        ent_slots, item_slots, holder_logits = model.write(s["narrative"], tokenizer, device)
        gen = model.read_answer(ent_slots, item_slots, holder_logits, s,
                                tokenizer, max_new=max_new).strip().lower()
        exp = s["answer"].strip().lower()
        ok = gen == exp
        correct += ok; total += 1
        by_task.setdefault(s["task_type"], [0, 0])
        by_task[s["task_type"]][0] += ok; by_task[s["task_type"]][1] += 1
    acc = correct / max(total, 1)
    task_acc = {t: c / m for t, (c, m) in by_task.items()}
    return acc, task_acc, total
