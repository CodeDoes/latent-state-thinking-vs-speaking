"""Minimal RWKV nano model for the logic niiah task.

RWKV-4-style recurrent block with WKV time-mixing and channel-mixing.
Designed to be tiny (~100K-1M params) and run on CPU.

The core recurrence:
    wkv_t = (num_{t-1} + exp(u + k_t) * v_t) / (den_{t-1} + exp(u + k_t))

where num and den are running sums with exponential decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _cumlogsumexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable cumulative log-sum-exp over the time dim (dim=1).

    Returns log(Σ_{i<=t} exp(x[:, i])). Used to vectorize the WKV
    recurrence's running sums. Fully vectorized: cumulative max via
    torch.cummax, cumulative exp-sum via torch.cumsum.
    """
    cmax = torch.cummax(x, dim=1).values  # (B, T, C) running max over time
    shifted = x - cmax
    exp_sum = torch.cumsum(torch.exp(shifted), dim=1)
    return cmax + torch.log(exp_sum + 1e-10)

class RWKVBlock(nn.Module):
    """One RWKV layer with time-mixing (WKV) and channel-mixing (FFN)."""

    def __init__(self, dim: int, hidden_scale: int = 4):
        super().__init__()
        self.dim = dim

        # ── Layer norms ──
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # ── Time mixing ──
        # Initialize time_decay to -0.5 for slower decay, longer memory
        self.time_decay = nn.Parameter(torch.full((dim,), -0.5))
        self.time_first = nn.Parameter(torch.zeros(dim))
        # Mixing ratios
        self.time_mix_k = nn.Parameter(torch.ones(dim))
        self.time_mix_v = nn.Parameter(torch.ones(dim))
        self.time_mix_r = nn.Parameter(torch.ones(dim))

        # Projections
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        # ── Channel mixing ──
        hidden_dim = dim * hidden_scale
        self.channel_mix_k = nn.Parameter(torch.ones(dim))
        self.channel_mix_r = nn.Parameter(torch.ones(dim))
        self.fc_key = nn.Linear(dim, hidden_dim, bias=False)
        self.fc_value = nn.Linear(hidden_dim, dim, bias=False)
        self.fc_receptance = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, state: Optional[dict] = None):
        """
        Args:
            x: (B, T, dim) input
            state: dict with 'xx' (B, dim) - previous timestep's x
        Returns:
            out: (B, T, dim) output
            new_state: updated state dict
        """
        B, T, C = x.shape

        if state is None or 'xx' not in state:
            xx = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        else:
            xx = state['xx']

        # ── Time mixing ──
        x_norm = self.ln1(x)
        # Mix with previous timestep
        xk = x_norm * self.time_mix_k + xx.unsqueeze(1) * (1 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + xx.unsqueeze(1) * (1 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + xx.unsqueeze(1) * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        # ── WKV recurrence (vectorized, autograd-friendly) ──
        # wkv_t = (Σ_{i<=t} w^{t-i} e^{k_i} v_i + e^{u+k_t} v_t)
        #         / (Σ_{i<=t} w^{t-i} e^{k_i} + e^{u+k_t})
        # with w = exp(-exp(decay)) ∈ (0,1). We compute the running sums
        # via a stable cumulative-sum in log space.
        w = self.time_decay
        u = self.time_first
        w_ = torch.exp(-torch.exp(w))                 # (C,) per-channel decay
        e_k = torch.exp(k)                            # (B, T, C)
        e_u_k = torch.exp(u + k)                      # (B, T, C)

        # Weighted running sums: a_t = w a_{t-1} + e^{k_t} v_t
        # Vectorize using the cumulative product trick:
        #   num_t = w^t ( Σ_{i<=t} w^{-i} e^{k_i} v_i )
        #   den_t = w^t ( Σ_{i<=t} w^{-i} e^{k_i} )
        # To avoid w^{-t} overflow, compute in log space with a running max.
        log_w = -torch.exp(w)                          # log(w_) = -exp(w)
        # term_num_i = log(w^{-i}) + log(e^{k_i}) + log(v_i)
        t_idx = torch.arange(T, device=x.device).view(1, T, 1).float()
        log_term_num = -t_idx * log_w + k + torch.log(torch.abs(v) + 1e-8)
        log_term_den = -t_idx * log_w + k
        # sign of v matters for num (denominator is always positive)
        sign_num = torch.sign(v)

        # Stable log-cumulative-sum: cumlogsumexp
        cum_num = _cumlogsumexp(log_term_num)         # (B, T, C)
        cum_den = _cumlogsumexp(log_term_den)
        # num_t = exp(cum_num + t*log_w) * sign
        num = torch.exp(cum_num + t_idx * log_w) * sign_num
        den = torch.exp(cum_den + t_idx * log_w)

        wkv_out = (num + e_u_k * v) / (den + e_u_k + 1e-8)  # (B, T, C)

        tm_out = r * wkv_out
        tm_out = self.output(tm_out)
        x = x + tm_out

        # ── Channel mixing ──
        x_norm = self.ln2(x)
        # Mix with previous timestep
        if state is not None and 'xx2' in state:
            xx2 = state['xx2']
        else:
            xx2 = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        xk = x_norm * self.channel_mix_k + xx2.unsqueeze(1) * (1 - self.channel_mix_k)
        xr = x_norm * self.channel_mix_r + xx2.unsqueeze(1) * (1 - self.channel_mix_r)

        k_c = torch.relu(self.fc_key(xk)) ** 2
        r_c = torch.sigmoid(self.fc_receptance(xr))
        v_c = self.fc_value(k_c)
        cm_out = r_c * v_c
        x = x + cm_out

        # ── Update state ──
        new_state = {
            'xx': x[:, -1, :].detach().clone(),    # last timestep's input
            'xx2': x_norm[:, -1, :].detach().clone(),  # for channel mixing
            'num': num.detach().clone(),
            'den': den.detach().clone(),
        }

        return x, new_state


class RWKVNano(nn.Module):
    """Tiny RWKV model for the logic niiah task.

    Args:
        vocab_size: size of the token vocabulary
        dim: embedding / hidden dimension
        num_layers: number of RWKV blocks
        hidden_scale: multiplier for FFN hidden size (dim * hidden_scale)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        num_layers: int = 3,
        hidden_scale: int = 4,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([
            RWKVBlock(dim, hidden_scale) for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)
        # NOTE: separate output head (NOT tied to the embedding). The
        # embedding uses padding_idx=0 which zeros the PAD row; a tied head
        # would give PAD a constant 0 logit and let it win everywhere. A
        # dedicated head with bias breaks that degeneracy.
        self.head = nn.Linear(dim, vocab_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        # Initialize with small values for stable training
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() == 1:
                if 'time_decay' in name:
                    # Per-channel decay spread: some channels track long-range
                    # (w_ close to 1), some decay fast. Critical for our task
                    # where the answer is at the END of a long sequence.
                    nn.init.uniform_(p, -6.0, -1.0)
                elif 'time_first' in name:
                    # Bonus for current token
                    nn.init.constant_(p, 0.0)
                else:
                    nn.init.constant_(p, 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[list[dict]] = None,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[dict]]]:
        """
        Args:
            input_ids: (B, T) token IDs
            state: optional list of per-layer states for incremental decoding
            return_state: if True, return updated states
        Returns:
            logits: (B, T, vocab_size)
            new_state: (optional) list of per-layer states
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)  # (B, T, dim)

        new_state = [] if return_state else None
        for i, block in enumerate(self.blocks):
            layer_state = state[i] if state is not None else None
            x, s = block(x, layer_state)
            if return_state:
                new_state.append(s)

        x = self.ln_out(x)
        logits = self.head(x)  # (B, T, vocab)

        if return_state:
            return logits, new_state
        return logits, None

    @torch.no_grad()
    def generate_one(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generate tokens autoregressively from a prompt."""
        self.eval()
        B, T = input_ids.shape
        assert B == 1, "Only batch_size=1 for generation"

        # Run prompt to build state
        _, state = self.forward(input_ids, return_state=True)

        generated = []
        token = input_ids[:, -1:]
        for _ in range(max_new_tokens):
            logits, state = self.forward(token, state=state, return_state=True)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            generated.append(token.item())
            if token.item() == 0:  # stop on pad
                break
        return generated


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = RWKVNano(vocab_size=128, dim=64, num_layers=2)
    x = torch.randint(1, 127, (2, 32))
    logits, state = model(x, return_state=True)
    print(f"Input:  {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Params: {count_params(model):,}")
