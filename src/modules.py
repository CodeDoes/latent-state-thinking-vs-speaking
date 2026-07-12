"""
Separable, individually-trainable modules for the hybrid latent-state LM.

Design (per project direction): build small input->output functions, each with
its OWN training objective, instead of one monolithic end-to-end model trained
with a single next-token loss. The user's latent-space algebra:

    make_B(A)   -> B                  narrative state A -> answer state B
    make_A(B)   -> A                  inverse
    continue(A) -> A2                 advance narrative state (state evolution)
    continue(B) -> B2                 advance answer state
    Answer_in_format_D(A,B,C) -> D    compose narrative+answer+question -> D

Plus, separately trainable I/O and support modules:
    token -> state   (TokenEncoder)
    state -> token   (StateDecoder)
    state reasoning  (ReasoningStep)
    context management (ContextManager)
    tape prefix embedding / active recall (Tape)

Object mapping for the toy-world tasks:
    A = latent state of the narrative
    B = latent state of the answer
    C = latent state of the question
    D = latent state of the answer rendered in "Answer: <x>" format

Each module is a plain nn.Module with its own forward contract. train_modules.py
trains each one separately (curriculum: first the autoencoder 'output sane
words', then the latent ops), so the latent state is forced to *contain the
correct thing* rather than just learning to spell tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# I/O adapters
# ---------------------------------------------------------------------------

class TokenEncoder(nn.Module):
    """tokens -> state. Returns the final latent state AND all per-token states."""

    def __init__(self, vocab_size: int, d_state: int = 256, d_model: int = 128,
                 n_layers: int = 2):
        super().__init__()
        self.d_state = d_state
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_state)
        # Recurrent core builds the state. LSTM is simple + trainable; the
        # final hidden state is the sequence-level state A/B/C.
        self.core = nn.LSTM(d_state, d_state, num_layers=n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_state)

    def forward(self, ids: torch.Tensor, return_all: bool = False):
        # ids: [B, T]
        x = self.proj(self.embed(ids))                 # [B, T, d_state]
        out, (h, _) = self.core(x)                     # out: [B, T, d_state]
        final = self.norm(h[-1])                       # [B, d_state]
        return (final, out) if return_all else final

    def states(self, ids: torch.Tensor) -> torch.Tensor:
        """Per-token states [B, T, d_state] (for continue() training)."""
        _, out = self.forward(ids, return_all=True)
        return out

    def state_of(self, ids: torch.Tensor) -> torch.Tensor:
        """Sequence-level final state [B, d_state]."""
        return self.forward(ids)


class StateDecoder(nn.Module):
    """state -> token logits. The 'output sane words' renderer."""

    def __init__(self, d_state: int, vocab_size: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, hidden), nn.SiLU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: [B, d_state] or [B, T, d_state] -> logits same shape minus last
        return self.net(state)


# ---------------------------------------------------------------------------
# Latent-space algebra (all state->state maps)
# ---------------------------------------------------------------------------

class StateTransform(nn.Module):
    """Generic state->state map. Used for make_B, make_A, continue_A, continue_B.

    Small residual MLP so each transform is cheap and independently trainable.
    """

    def __init__(self, d_state: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, d_state),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class AnswerComposer(nn.Module):
    """(A, B, C) -> D : compose narrative + answer + question states."""

    def __init__(self, d_state: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state * 3, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, d_state),
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([A, B, C], dim=-1))


class AnswerDecoder(nn.Module):
    """Autoregressive head: starting from a composed state D, emit the ANSWER
    token sequence (NOT "Answer: ..." -- just the answer string).

    Why not an LSTM seeded from D?  We tried that and the previous Kaggle run
    finished with `oracle_answer_head_acc=0.00` -- meaning even when fed the
    TRUE teacher state, the LSTM could not decode it back to the answer, never
    mind at compositing time. The cause was that generation-time uses a
    zero-valued `start_emb` to roll the LSTM, so the only signal the LSTM sees
    is the initial hidden state h0 = proj(D); this requires the linear->LSTM
    alignment to be learned FROM SCRATCH through a single forward pass, which
    converged to a constant output.

    Instead, this is a NON-recurrent per-position MLP decoder:
      logit_t = MLP([D; pos_embed_t])                  (shape: [B, T, vocab])
    where the MLP is the same per position (weight sharing). It conditions at
    every step on the state D, so even at generation time the prediction has
    rich signal. Teacher-forcing at training time and greedy argmax at
    generation time. Maximum answer length is fixed by MAX_TOKENS to bound the
    parameters; padding past the true end is masked out by the cross-entropy
    (ignore_index=pad).
    """

    MAX_TOKENS = 24

    def __init__(self, d_state: int, vocab_size: int, hidden: int = 512, max_tokens: int = MAX_TOKENS):
        super().__init__()
        self.d_state = d_state
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        # Learnable positional embeddings -> MLP -> logits.
        self.pos_embed = nn.Parameter(torch.randn(max_tokens, d_state) * 0.02)
        self.net = nn.Sequential(
            nn.Linear(d_state + d_state, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, vocab_size),
        )
        # Independent projection of D so the decoder isn't forced to read the
        # entire state through one path -- a residual that always has the right
        # output shape, so even at the START of training the state -> vocab
        # linear at least produces a vocab-shaped output (not garbage from
        # breaking activations). This is what the old LSTM head violently
        # lacked -- an anchor path from state -> logit.
        self.state_proj = nn.Linear(d_state, vocab_size)

    def forward_teacher(self, D: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward -- produces logits for ALL T target positions.

        Args:
            D: [B, d_state] - the composed state to decode.
            tgt_ids: [B, T] - answer token ids (padded/truncated to T<=max_tokens).

        Returns:
            logits: [B, T, vocab] aligned to tgt_ids, for cross-entropy.
        """
        B, T = tgt_ids.shape
        T = min(T, self.max_tokens)
        D_e = D.unsqueeze(1).expand(-1, T, -1)            # [B, T, d]
        pos = self.pos_embed[:T].unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([D_e, pos], dim=-1)                # [B, T, 2d]
        return self.net(h) + self.state_proj(D_e)        # additive residual anchor

    @torch.no_grad()
    def generate(self, D: torch.Tensor, max_tokens: int = None,
                 eos_id=None, pad_id=None):
        """Greedy generation: returns a list of token ids (the answer).

        Stops early if eos_id is emitted; otherwise emits MAX_TOKENS tokens.
        Does NOT train anything internally -- `generate` is @torch.no_grad.
        """
        B = D.size(0)
        T = min(max_tokens or self.max_tokens, self.max_tokens)
        D_e = D.unsqueeze(1).expand(B, T, -1)
        pos = self.pos_embed[:T].unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([D_e, pos], dim=-1)
        out = self.net(h) + self.state_proj(D_e)         # [B, T, vocab]
        argmax = out.argmax(-1)
        if B == 1:
            ids = argmax[0].tolist()
        else:
            # The current pipeline only calls this with B==1 (per sample).
            # If a future caller passes B>1, return the first row.
            ids = argmax[0].tolist()
        if eos_id is not None or pad_id is not None:
            ids = [t for t in ids
                   if (eos_id is None or t != eos_id)
                   and (pad_id is None or t != pad_id)]
        return ids


# ---------------------------------------------------------------------------
# Support pieces (each separable + trainable on its own)
# ---------------------------------------------------------------------------

class ReasoningStep(nn.Module):
    """State reasoning: apply a transform K times in latent space (the 'think'
    loop). Trained by a consistency/evolution objective in train_modules.py."""

    def __init__(self, d_state: int, steps: int = 4, hidden: int = 512):
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(d_state, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, d_state),
        )

    def forward(self, s: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.steps
        for _ in range(steps):
            s = s + self.net(s)          # residual thinking
        return s


class ContextManager(nn.Module):
    """Context management: soft-pool recent states into a fixed context vector.

    Trained to predict the next state from a window of recent states (so it
    learns 'what is relevant now' instead of attending to the whole stream)."""

    def __init__(self, d_state: int, hidden: int = 256):
        super().__init__()
        self.attn = nn.Linear(d_state, 1)
        self.merge = nn.Sequential(
            nn.Linear(d_state, hidden), nn.SiLU(),
            nn.Linear(hidden, d_state),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: [B, T, d_state] -> context [B, d_state]
        w = torch.softmax(self.attn(states), dim=1)         # [B, T, 1]
        pooled = (w * states).sum(dim=1)                    # [B, d_state]
        return self.merge(pooled)


class Tape(nn.Module):
    """Prefix tape memory: exact recall. Two separable ops:
      - write(keys, vals): build a key->val memory from a sequence (prefix embed)
      - recall(query): attend over the tape to retrieve (active recall)
    Trained with a retrieval (key->val) objective, separate from reasoning."""

    def __init__(self, d_state: int, hidden: int = 256):
        super().__init__()
        self.k_proj = nn.Linear(d_state, hidden)
        self.v_proj = nn.Linear(d_state, hidden)
        self.q_proj = nn.Linear(d_state, hidden)
        self.out = nn.Linear(hidden, d_state)

    def write(self, states: torch.Tensor):
        # states: [B, T, d_state] -> (keys, vals) stored on the tape
        return self.k_proj(states), self.v_proj(states)

    def recall(self, query: torch.Tensor, tape):
        # query: [B, d_state]; tape = (keys, vals) [B, T, h]
        keys, vals = tape
        q = self.q_proj(query)                              # [B, h]
        scores = torch.einsum("bh,bth->bt", q, keys)        # [B, T]
        w = torch.softmax(scores, dim=1)
        retrieved = torch.einsum("bt,bth->bh", w, vals)     # [B, h]
        return self.out(retrieved)                          # [B, d_state]


# ---------------------------------------------------------------------------
# Compositional inference (ties the pieces together at test time)
# ---------------------------------------------------------------------------

class StateCrossAttn(nn.Module):
    """Extract the answer state B by attending over the narrative's per-token
    states A_seq, conditioned on the question state C.

    Replaces the old make_B(A_vec)->B single-vector MLP, which COLLAPSED to the
    mean (MSE ~ variance of B) because a single pooled narrative vector is too
    lossy to recover a specific fact from a long context. Attention over the
    token-level states is the mechanism that actually lets the model 'read the
    story and pull out the answer'.
    """

    def __init__(self, d_state: int, n_heads: int = 4):
        super().__init__()
        assert d_state % n_heads == 0
        self.d_state = d_state
        self.n_heads = n_heads
        self.q = nn.Linear(d_state, d_state)
        self.k = nn.Linear(d_state, d_state)
        self.v = nn.Linear(d_state, d_state)
        self.out = nn.Linear(d_state, d_state)
        self.norm = nn.LayerNorm(d_state)

    def forward(self, A_seq: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # A_seq: [B, T, d]  (per-token narrative states)
        # C:     [B, d]     (question state, used as the attention query)
        B_ = A_seq.size(0)
        dk = self.d_state // self.n_heads
        q = self.q(C).view(B_, self.n_heads, 1, dk)
        k = self.k(A_seq).view(B_, self.n_heads, -1, dk)
        v = self.v(A_seq).view(B_, self.n_heads, -1, dk)
        scores = (q * k).sum(-1) / (dk ** 0.5)          # [B, H, T]
        w = torch.softmax(scores, dim=-1)
        ctx = (w.unsqueeze(-1) * v).sum(-2)              # [B, H, dk]
        ctx = ctx.reshape(B_, self.d_state)
        return self.norm(self.out(ctx))


@torch.no_grad()
def compose_answer(encoder, make_b, composer, ans_dec, A_seq, C, max_tokens=24,
                   eos_id=None, pad_id=None):
    """Run the latent algebra end-to-end to produce an answer string.

    A_seq = narrative per-token states, C = question state.
    B = make_B(A_seq, C)  (attention over the narrative);
    D = composer(A_vec, B, C);  ans_dec generates the answer tokens from D.
    """
    A_vec = A_seq[:, -1, :]                        # [B, d] pooled narrative state
    B = make_b(A_seq, C)
    D = composer(A_vec, B, C)
    ids = ans_dec.generate(D, max_tokens=max_tokens, eos_id=eos_id, pad_id=pad_id)
    return ids
