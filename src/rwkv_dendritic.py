"""RWKV with Dendritic Channel Mixing.

Sparse, homeostatic channel-mix layer with top-k gating and duty-cycle regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _cumlogsumexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable cumulative log-sum-exp over time dim (dim=1)."""
    cmax = torch.cummax(x, dim=1).values
    shifted = x - cmax
    exp_sum = torch.cumsum(torch.exp(shifted), dim=1)
    return cmax + torch.log(exp_sum + 1e-10)


class DendriticChannelMix(nn.Module):
    """Sparse homeostatic channel mixing layer.

    Replaces dense FC-Key → ReLU² → FC-Value + FC-Receptance with:
      1. Context-gated top-k unit selection (dendritic competition)
      2. Duty-cycle homeostatic correction (boost suppressed units)
      3. Only active units compute value projection

    Args:
        dim: Input/output dimension
        hidden_scale: Multiplier for hidden dimension (default 4)
        n_active: Number of active units per token (top-k). If None, uses hidden_dim // 8.
        duty_cycle_momentum: EMA momentum for duty cycle tracking (default 0.99)
        target_duty: Target duty cycle per unit (default 1 / n_active)
        boost_strength: How strongly to boost under-firing units (default 1.0)
    """

    def __init__(
        self,
        dim: int,
        hidden_scale: int = 4,
        n_active: Optional[int] = None,
        duty_cycle_momentum: float = 0.99,
        target_duty: Optional[float] = None,
        boost_strength: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim * hidden_scale
        self.n_active = n_active or (self.hidden_dim // 8)
        self.duty_cycle_momentum = duty_cycle_momentum
        self.boost_strength = boost_strength

        # Target duty cycle: each unit should fire this often
        self.target_duty = target_duty or (self.n_active / self.hidden_dim)

        # ── Projections ──
        # Context encoder: maps current hidden state → gate logits for all hidden units
        self.context_proj = nn.Linear(dim, self.hidden_dim, bias=False)

        # Standard channel-mix projections (applied only to active units)
        self.fc_key = nn.Linear(dim, self.hidden_dim, bias=False)
        self.fc_value = nn.Linear(self.hidden_dim, dim, bias=False)
        self.fc_receptance = nn.Linear(dim, dim, bias=False)

        # Time-mixing parameters (same as standard RWKV)
        self.channel_mix_k = nn.Parameter(torch.ones(dim))
        self.channel_mix_r = nn.Parameter(torch.ones(dim))

        # Time-shift for token-shift mixing
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # ── Duty cycle tracking (buffer, not parameter) ──
        self.register_buffer('duty_cycle', torch.zeros(self.hidden_dim))
        self.register_buffer('step_count', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict] = None,
    ) -> tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (B, T, dim) - input from time-mix output
            state: dict with 'xx2' (previous ln2(x) for time-shift) and
                   duty_cycle/step_count for homeostasis

        Returns:
            out: (B, T, dim)
            new_state: updated state dict
        """
        B, T, C = x.shape
        assert C == self.dim

        # Layer norm (pre-norm for channel mix)
        x_norm = F.layer_norm(x, (C,))

        # Time-shift: mix with previous timestep
        if state is not None and 'xx2' in state:
            xx2 = state['xx2']
        else:
            xx2 = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        if T > 1:
            x_prev = torch.cat([xx2.unsqueeze(1), x_norm[:, :-1, :]], dim=1)
        else:
            x_prev = xx2.unsqueeze(1)

        # Channel-mix token shift mixing
        xk = x_norm * self.channel_mix_k + x_prev * (1 - self.channel_mix_k)
        xr = x_norm * self.channel_mix_r + x_prev * (1 - self.channel_mix_r)

        # ── Receptance gate (dense, as standard) ──
        r_c = torch.sigmoid(self.fc_receptance(xr))  # (B, T, dim)

        # ── Dendritic sparse activation ──
        # Use first token of sequence (or mean) as context for gating
        # For simplicity, use mean over sequence as context
        context = x_norm.mean(dim=1)  # (B, dim)

        # Get gate logits for all hidden units
        gate_logits = self.context_proj(context)  # (B, hidden_dim)

        # Homeostatic boost: compute boost from duty cycle
        if self.training:
            # Update duty cycle EMA
            with torch.no_grad():
                # We'll compute actual duty cycle from activations below
                pass

        # Apply duty-cycle correction (boost under-firing units)
        # boost = boost_strength * (target_duty - duty_cycle)
        boost = self.boost_strength * (self.target_duty - self.duty_cycle)  # (hidden_dim,)
        gate_logits = gate_logits + boost.unsqueeze(0)  # (B, hidden_dim)

        # Top-k gating
        topk_vals, topk_idx = torch.topk(gate_logits, self.n_active, dim=-1)  # (B, n_active)

        # Create sparse mask
        sparse_mask = torch.zeros_like(gate_logits, dtype=torch.bool)
        sparse_mask.scatter_(-1, topk_idx, True)  # (B, hidden_dim)

        # Compute key projection for ALL units (needed for ReLU² activation pattern)
        # But we only need values for active units. To keep autograd happy,
        # compute full key, then mask.
        k_full = self.fc_key(xk)  # (B, T, hidden_dim)
        k_act = F.relu(k_full) ** 2  # ReLU² activation

        # Mask inactive units
        k_act = k_act * sparse_mask.unsqueeze(1).float()  # (B, T, hidden_dim)

        # Value projection: only active units contribute
        v_c = self.fc_value(k_act)  # (B, T, dim)

        # Channel mix output
        cm_out = r_c * v_c
        out = x + cm_out

        # ── Update state ──
        new_state = {
            'xx2': x_norm[:, -1, :].detach().clone(),
        }

        # Update duty cycle during training
        if self.training:
            with torch.no_grad():
                # Average activation per unit over batch and time
                unit_activity = sparse_mask.float().mean(dim=0)  # (hidden_dim,)
                # EMA update
                self.duty_cycle.mul_(self.duty_cycle_momentum)
                self.duty_cycle.add_(unit_activity * (1 - self.duty_cycle_momentum))
                self.step_count.add_(1)

        return out, new_state

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, hidden={self.hidden_dim}, n_active={self.n_active}, '
            f'target_duty={self.target_duty:.4f}, boost={self.boost_strength}'
        )


class RWKVBlock(nn.Module):
    """RWKV block with DendriticChannelMix."""

    def __init__(self, dim: int, hidden_scale: int = 4, **dendritic_kwargs):
        super().__init__()
        self.dim = dim

        # Layer norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # Time mixing
        self.time_decay = nn.Parameter(torch.full((dim,), -0.5))
        self.time_first = nn.Parameter(torch.zeros(dim))
        self.time_mix_k = nn.Parameter(torch.ones(dim))
        self.time_mix_v = nn.Parameter(torch.ones(dim))
        self.time_mix_r = nn.Parameter(torch.ones(dim))

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        # Channel mixing (dendritic)
        self.channel_mix = DendriticChannelMix(
            dim, hidden_scale, **dendritic_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict] = None,
    ) -> tuple[torch.Tensor, Dict]:
        B, T, C = x.shape

        if state is None:
            state = {}

        # Get previous timestep's ln1(x) for time mixing
        if 'xx' in state:
            xx = state['xx']
        else:
            xx = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        # Time mixing
        x_norm = self.ln1(x)
        if T > 1:
            x_prev = torch.cat([xx.unsqueeze(1), x_norm[:, :-1, :]], dim=1)
        else:
            x_prev = xx.unsqueeze(1)

        xk = x_norm * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        # WKV recurrence (vectorized)
        w = self.time_decay
        u = self.time_first
        w_ = torch.exp(-torch.exp(w))  # decay factor per channel
        e_k = torch.exp(k)
        e_u_k = torch.exp(u + k)

        log_w = -torch.exp(w)
        t_idx = torch.arange(T, device=x.device).view(1, T, 1).float()

        log_term_num = -t_idx * log_w + k + torch.log(torch.abs(v) + 1e-8)
        log_term_den = -t_idx * log_w + k
        sign_num = torch.sign(v)

        cum_num = _cumlogsumexp(log_term_num)
        cum_den = _cumlogsumexp(log_term_den)

        S_num = torch.exp(cum_num + t_idx * log_w) * sign_num
        S_den = torch.exp(cum_den + t_idx * log_w)

        # Historical state
        if 'num' in state and 'den' in state:
            num_prev = state['num']
            den_prev = state['den']
        else:
            num_prev = torch.zeros(B, C, device=x.device, dtype=x.dtype)
            den_prev = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        decay_factor = torch.exp((t_idx + 1) * log_w)
        num = decay_factor * num_prev.unsqueeze(1) + S_num
        den = decay_factor * den_prev.unsqueeze(1) + S_den

        wkv_out = (num + e_u_k * v) / (den + e_u_k + 1e-8)

        tm_out = r * wkv_out
        tm_out = self.output(tm_out)
        x = x + tm_out

        # Channel mixing (dendritic)
        x, cm_state = self.channel_mix(x, state)

        # Update state
        new_state = {
            'xx': x_norm[:, -1, :].detach().clone(),
            'num': num[:, -1, :].detach().clone(),
            'den': den[:, -1, :].detach().clone(),
            **cm_state,
        }

        return x, new_state


class RWKVNanoDendritic(nn.Module):
    """Tiny RWKV with DendriticChannelMix."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        num_layers: int = 3,
        hidden_scale: int = 4,
        pad_token_id: int = 0,
        **dendritic_kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([
            RWKVBlock(dim, hidden_scale, **dendritic_kwargs)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() == 1:
                if 'time_decay' in name:
                    nn.init.uniform_(p, -6.0, -1.0)
                elif 'time_first' in name:
                    nn.init.constant_(p, 0.0)
                else:
                    nn.init.constant_(p, 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[List[Dict]] = None,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, Optional[List[Dict]]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)

        new_state = [] if return_state else None
        for i, block in enumerate(self.blocks):
            layer_state = state[i] if state is not None else None
            x, s = block(x, layer_state)
            if return_state:
                new_state.append(s)

        x = self.ln_out(x)
        logits = self.head(x)

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
        self.eval()
        B, T = input_ids.shape
        assert B == 1

        _, state = self.forward(input_ids, return_state=True)

        generated = []
        token = input_ids[:, -1:]
        for _ in range(max_new_tokens):
            logits, state = self.forward(token, state=state, return_state=True)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            generated.append(token.item())
            if token.item() == 0:
                break
        return generated


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test standard vs dendritic param counts
    model_std = RWKVNanoDendritic(vocab_size=128, dim=64, num_layers=2)
    print(f"Dendritic model params: {count_params(model_std):,}")

    # Test forward
    x = torch.randint(1, 127, (2, 32))
    logits, state = model_std(x, return_state=True)
    print(f"Input: {x.shape} -> Logits: {logits.shape}")
    print(f"State layers: {len(state) if state else 0}")

    # Check duty cycle buffers
    for name, buf in model_std.named_buffers():
        if 'duty_cycle' in name:
            print(f"  {name}: {buf.shape}")

    print("Dendritic model test passed!")