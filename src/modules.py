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

@torch.no_grad()
def compose_answer(encoder, make_b, composer, dec_b, dec, A, C, max_tokens=24,
                   eos_id=None, pad_id=None):
    """Run the latent algebra end-to-end to produce an answer string.

    A = narrative state, C = question state.
    B = make_B(A);  D = composer(A, B, C);  decode D token-by-token using the
    answer-state advance dec_b (continue(B)) so multi-token answers work.
    """
    B = make_b(A)
    D = composer(A, B, C)
    state = D
    ids = []
    for _ in range(max_tokens):
        logit = dec(state)                      # [1, vocab]
        tok = int(torch.argmax(logit, dim=-1).item())
        if eos_id is not None and tok == eos_id:
            break
        if pad_id is not None and tok == pad_id:
            break
        ids.append(tok)
        state = dec_b(state)                    # advance answer state
    return ids
