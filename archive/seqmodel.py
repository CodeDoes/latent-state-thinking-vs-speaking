"""Clean seq2seq pivot: encoder -> latent "prepared" sequence -> decoder.

Design (the project's original thesis, minus the hand-built slot machinery):
  narrative --encoder--> memory --(latent queries cross-attend)--> Z  (think once)
  Z + "Question: ... Answer:" --decoder--> answer tokens            (speak many)

Z is built ONCE per narrative; many questions can be answered from the same Z.
No read-heads, no reverse_templates supervision, no pool-matching, no
last-mention bottleneck -- and generative recall works because it is just
token generation. All tasks (location/inventory/transfer/holder/recall/...)
share one uniform objective: predict the answer token sequence.

Reuses src.dataset (narrative/question/answer) and src.tokenizer (CharTokenizer).
"""
import math
import torch
import torch.nn as nn

PAD_ID = 0


class SeqWorldModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, n_enc=2, n_dec=2,
                 max_len=720, latent_len=16):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.latent_len = latent_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(0.1)

        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead,
                                       dim_feedforward=4 * d_model,
                                       batch_first=True, dropout=0.1),
            num_layers=n_enc)

        # latent "prepare" step: learned queries cross-attend the encoder memory
        self.latent_queries = nn.Parameter(torch.randn(latent_len, d_model) * 0.02)
        self.latent_proj = nn.Linear(d_model, d_model)

        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead,
                                       dim_feedforward=4 * d_model,
                                       batch_first=True, dropout=0.1),
            num_layers=n_dec)

        self.out = nn.Linear(d_model, vocab_size)

    # -- helpers ----------------------------------------------------------
    def _embed(self, ids):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        return self.drop(self.tok_emb(ids) + self.pos_emb(pos[:, :T]))

    @staticmethod
    def _pad_mask(ids):
        return ids == PAD_ID  # True = ignore (pad)

    # -- forward pieces ----------------------------------------------------
    def encode(self, narr_ids):
        """Narrative -> prepared latent sequence Z [B, L, d]."""
        x = self._embed(narr_ids)
        mem = self.enc(x, src_key_padding_mask=self._pad_mask(narr_ids))   # [B, Tn, d]
        B = narr_ids.size(0)
        q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)            # [B, L, d]
        z = self.dec(tgt=q, memory=mem)                                   # [B, L, d]
        return self.latent_proj(z)

    def decode_logits(self, z, tgt_ids):
        """Z + tgt tokens -> logits [B, T, V]."""
        x = self._embed(tgt_ids)
        T = tgt_ids.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)
        h = self.dec(tgt=x, memory=z, tgt_mask=causal,
                     tgt_key_padding_mask=self._pad_mask(tgt_ids))
        return self.out(h)

    def forward(self, narr_ids, tgt_ids):
        return self.decode_logits(self.encode(narr_ids), tgt_ids)


# ---------------------------------------------------------------------------
# Training / evaluation (uniform across all tasks)
# ---------------------------------------------------------------------------
def _build_prompt_answer(tokenizer, sample):
    prompt = f"Question: {sample['question']}\nAnswer: "
    prompt_ids = tokenizer.encode(prompt, max_len=None)
    nl = tokenizer.encode("\n", max_len=None)[0]
    ans_ids = tokenizer.encode(sample["answer"], max_len=None) + [nl]   # terminate
    return prompt_ids, ans_ids


def train_seq(model, qa, tokenizer, device, epochs=20, batch_size=16, lr=3e-4,
              max_narr=600, max_ans=40, verbose=True):
    from torch.nn.utils.rnn import pad_sequence
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # precompute tensors
    data = []
    for s in qa:
        nids = tokenizer.encode(s["narrative"], max_len=None)[:max_narr]
        pids, aids = _build_prompt_answer(tokenizer, s)
        if not nids or not aids:
            continue
        data.append((torch.tensor(nids, dtype=torch.long),
                     torch.tensor(pids, dtype=torch.long),
                     torch.tensor(aids, dtype=torch.long)))
    hist = []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        import random
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            nids = pad_sequence([d[0] for d in batch], batch_first=True, padding_value=PAD_ID).to(device)
            pids = [d[1] for d in batch]
            aids = [d[2] for d in batch]
            # decoder input = prompt + answer[:-1]; target = prompt + answer
            dec_in, tgt, pmask = [], [], []
            for p, a in zip(pids, aids):
                din = torch.cat([p, a[:-1]])
                tg = torch.cat([p, a])
                dec_in.append(din); tgt.append(tg)
                pmask.append(torch.tensor([False] * len(p) + [True] * len(a), dtype=torch.bool))
            L = max(t.size(0) for t in tgt)                      # tgt is 1 longer than dec_in
            dec_in = [torch.cat([d, torch.full((L - d.size(0),), PAD_ID,
                                               dtype=torch.long)]) for d in dec_in]
            dec_in = torch.stack(dec_in).to(device)
            tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD_ID).to(device)
            pmask = pad_sequence(pmask, batch_first=True, padding_value=False).to(device)
            logits = model(nids, dec_in)                       # [B, T, V]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                reduction="none").reshape(tgt.shape)
            loss = (loss * pmask).sum() / pmask.sum()          # answer-region only
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if verbose:
            print(f"  [seq] ep {ep+1}/{epochs} loss={tot/max(1,len(data)//batch_size):.4f}")
        hist.append(tot / max(1, len(data) // batch_size))
    return model, hist


@torch.no_grad()
def run_seq_qa(model, qa, tokenizer, device, max_new=40):
    model.eval()
    acc_by = {}
    correct = total = 0
    nl_id = tokenizer.encode("\n", max_len=None)[0] if tokenizer.encode("\n", max_len=None) else -1
    for s in qa:
        nids = torch.tensor(tokenizer.encode(s["narrative"], max_len=None)[:600],
                            dtype=torch.long).unsqueeze(0).to(device)
        pids = torch.tensor(tokenizer.encode(f"Question: {s['question']}\nAnswer: ",
                            max_len=None), dtype=torch.long).unsqueeze(0).to(device)
        z = model.encode(nids)
        ids = pids
        for _ in range(max_new):
            logits = model.decode_logits(z, ids)[:, -1, :]
            nxt = int(logits.argmax(-1).item())
            if nxt == nl_id or nxt == PAD_ID:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        gen = tokenizer.decode(ids[0, pids.size(1):].tolist()).strip()
        ok = (gen == s["answer"].strip())
        correct += int(ok); total += 1
        acc_by.setdefault(s["task_type"], []).append(float(ok))
    overall = correct / total if total else 0.0
    per = {k: sum(v) / len(v) for k, v in acc_by.items()}
    return per, overall, total
