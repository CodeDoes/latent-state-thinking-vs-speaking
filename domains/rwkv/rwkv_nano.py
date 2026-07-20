"""Minimal RWKV-7 nano model.

RWKV-7-style recurrent block with multi-head matrix state, data-dependent
time decay, value residual, and group normalization. Matches the core
architecture of RWKV-7 "Goose" at nano scale.

Key differences from RWKV-4:
- Multi-head: state is (H, head_size, head_size) matrix per layer
- Data-dependent decay: w = f(x) via tanh + projection
- Value residual: first layer's values persist through later layers
- Group normalization instead of LayerNorm on the WKV output
- Gate (g) on the output, attention gate (a) on the key-value interaction

Designed to be tiny (~100K-1M params) and run on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class RWKV7Block(nn.Module):
    """One RWKV-7 layer with multi-head matrix state, data-dependent decay."""

    def __init__(self, dim: int, head_size: int = 64, hidden_scale: int = 4):
        super().__init__()
        self.dim = dim
        self.head_size = head_size
        assert dim % head_size == 0, f"dim={dim} must be divisible by head_size={head_size}"
        self.n_head = dim // head_size
        H = self.n_head
        N = head_size

        # ── Time mix parameters ──
        # Data-dependent shift mixing
        self.time_mix_r = nn.Parameter(torch.ones(H, N))
        self.time_mix_w = nn.Parameter(torch.ones(H, N))
        self.time_mix_k = nn.Parameter(torch.ones(H, N))
        self.time_mix_v = nn.Parameter(torch.ones(H, N))
        self.time_mix_a = nn.Parameter(torch.ones(H, N))
        self.time_mix_g = nn.Parameter(torch.ones(H, N))

        # Projections
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)

        # Data-dependent decay: w = tanh(x @ w1) @ w2
        self.w1 = nn.Linear(dim, dim, bias=False)
        self.w2 = nn.Linear(dim, dim, bias=False)
        self.w0 = nn.Parameter(torch.zeros(H, N))

        # Attention gate: a = sigmoid(a0 + (x @ a1) @ a2)
        self.a1 = nn.Linear(dim, dim, bias=False)
        self.a2 = nn.Linear(dim, dim, bias=False)
        self.a0 = nn.Parameter(torch.zeros(H, N))

        # Value residual params: v = v + (v_first - v) * sigmoid(v0 + (xv @ v1) @ v2)
        self.v1 = nn.Linear(dim, dim, bias=False)
        self.v2 = nn.Linear(dim, dim, bias=False)
        self.v0 = nn.Parameter(torch.zeros(H, N))

        # k_k (key normalization), k_a (key-attention interaction), r_k (receptance-key)
        self.k_k = nn.Parameter(torch.ones(H, N))
        self.k_a = nn.Parameter(torch.ones(H, N))
        self.r_k = nn.Parameter(torch.ones(H, N))

        # Output projection
        self.output = nn.Linear(dim, dim, bias=False)

        # Group normalization on WKV output (H groups)
        self.ln_w = nn.Parameter(torch.ones(dim))
        self.ln_b = nn.Parameter(torch.zeros(dim))

        # ── Channel mix parameters ──
        hidden_dim = dim * hidden_scale
        self.time_mix_k_c = nn.Parameter(torch.ones(H, N))
        self.time_mix_r_c = nn.Parameter(torch.ones(H, N))
        self.fc_key = nn.Linear(dim, hidden_dim, bias=False)
        self.fc_value = nn.Linear(hidden_dim, dim, bias=False)
        self.fc_receptance = nn.Linear(dim, dim, bias=False)
        self.ln_c = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, state: Optional[dict] = None,
                v_first: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, dim) input
            state: dict with:
                'xx' — (B, dim) previous timestep's x
                'state' — (B, H, N, N) per-head matrix state
            v_first: (B, dim) value residual from layer 0 (propagated)
        Returns:
            out: (B, T, dim)
            new_state: updated state dict
            new_v_first: updated v_first
        """
        B, T, C = x.shape
        H = self.n_head
        N = self.head_size
        device = x.device
        dtype = x.dtype

        # ── State initialization ──
        if state is None:
            xx = torch.zeros(B, C, device=device, dtype=dtype)
            mat_state = torch.zeros(B, H, N, N, device=device, dtype=torch.float)
        else:
            xx = state.get('xx', torch.zeros(B, C, device=device, dtype=dtype))
            mat_state = state.get('state', torch.zeros(B, H, N, N, device=device, dtype=torch.float))

        if v_first is None:
            v_first = torch.zeros(B, C, device=device, dtype=dtype)

        # ── Time mixing ──
        # Shift: mix current with previous timestep
        if T > 1:
            x_prev = torch.cat([xx.unsqueeze(1), x[:, :-1, :]], dim=1)
        else:
            x_prev = xx.unsqueeze(1)

        # Data-dependent interpolation
        def reshape_to_head(t):
            """Reshape (B, T, C) → (B, T, H, N)."""
            return t.view(B, T, H, N)

        xx_diff = x_prev - x  # (B, T, C)

        # Mix each head separately using per-head time_mix params
        x_diff_h = reshape_to_head(xx_diff)  # (B, T, H, N)
        x_h = reshape_to_head(x)             # (B, T, H, N)

        xr_h = x_h + x_diff_h * self.time_mix_r  # (B, T, H, N)
        xw_h = x_h + x_diff_h * self.time_mix_w
        xk_h = x_h + x_diff_h * self.time_mix_k
        xv_h = x_h + x_diff_h * self.time_mix_v
        xa_h = x_h + x_diff_h * self.time_mix_a
        xg_h = x_h + x_diff_h * self.time_mix_g

        # Flatten back for linear projections
        xr = xr_h.reshape(B, T, C)
        xw = xw_h.reshape(B, T, C)
        xk = xk_h.reshape(B, T, C)
        xv = xv_h.reshape(B, T, C)
        xa = xa_h.reshape(B, T, C)
        xg = xg_h.reshape(B, T, C)

        r = self.receptance(xr)  # (B, T, C)
        w_raw = torch.tanh(self.w1(xw)) @ self.w2.weight  # (B, T, C)
        k = self.key(xk)
        v = self.value(xv)
        a_raw = self.a1(xa) @ self.a2.weight  # (B, T, C)
        g = torch.sigmoid(self.gate(xg))

        # Reshape to heads
        r_h = r.view(B, T, H, N)       # (B, T, H, N)
        w_h = w_raw.view(B, T, H, N)   # (B, T, H, N)
        k_h = k.view(B, T, H, N)
        v_h = v.view(B, T, H, N)
        a_h = (self.a0 + a_raw.view(B, T, H, N))  # (B, T, H, N), sigmoid below
        g_h = g.view(B, T, H, N)
        a_h = torch.sigmoid(a_h)

        # Normalize k with k_k
        kk_h = F.normalize(k_h * self.k_k, dim=-1, p=2.0)  # (B, T, H, N)
        k_h = k_h * (1 + (a_h - 1) * self.k_a)

        # Value residual: first layer stores v_first, later layers blend
        is_first_layer = (v_first is None or (isinstance(v_first, torch.Tensor) and v_first.numel() == 1 and v_first.item() == 0))
        if is_first_layer:
            ret_v_first = v.detach()  # layer 0: store as v_first
        else:
            v_first_h = v_first.view(B, H, N)
            blend = torch.sigmoid(self.v0 + (self.v1(xv) @ self.v2.weight).view(B, T, H, N))
            v_h = v_h + (v_first_h.unsqueeze(1) - v_h) * blend
            ret_v_first = v_first

        # Data-dependent decay: w ∈ (0, 1)
        # w = exp(-0.606531 * sigmoid(w0 + w))
        w_decay = torch.exp(-0.606531 * torch.sigmoid((self.w0 + w_h).float()))  # (B, T, H, N)

        # ── WKV recurrence per head ──
        # For each token position, update the matrix state:
        # state = state * w + state @ ab + vk
        # where vk = v @ k^T (outer product)
        #       ab = (-kk) @ (kk * a)^T
        # output = state @ r

        output_h = torch.zeros(B, T, H, N, device=device, dtype=dtype)

        for t in range(T):
            r_t = r_h[:, t, :, :]          # (B, H, N)
            w_t = w_decay[:, t, :, :]      # (B, H, N) — per-element decay
            k_t = k_h[:, t, :, :]          # (B, H, N)
            v_t = v_h[:, t, :, :]          # (B, H, N)
            kk_t = kk_h[:, t, :, :]       # (B, H, N)
            a_t = a_h[:, t, :, :]           # (B, H, N)

            # vk = v @ k^T  → (B, H, N, N)
            vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)

            # ab = (-kk) @ (kk * a)^T → (B, H, N, N)
            ab = (-kk_t).unsqueeze(-1) @ (kk_t * a_t).unsqueeze(-2)

            # state = state * w + state @ ab + vk
            # w_t (B, H, N) → (B, H, 1, N) for elementwise multiply with (B, H, N, N)
            mat_state = (mat_state * w_t.unsqueeze(-2).float()
                         + (mat_state @ ab.float())
                         + vk.float())

            # output = state @ r  → (B, H, N, N) @ (B, H, N, 1) = (B, H, N, 1)
            out_t = (mat_state.to(dtype=dtype) @ r_t.unsqueeze(-1)).squeeze(-1)  # (B, H, N)
            output_h[:, t, :, :] = out_t

        # Group normalization on output
        output = output_h.reshape(B, T, C)  # (B, T, C)
        output = F.group_norm(
            output.view(B * T, C), num_groups=H,
            weight=self.ln_w, bias=self.ln_b, eps=64e-5,
        ).view(B, T, C)

        # Residual shortcut: (r * k * r_k).sum(dim=-1) * v
        shortcut = (r_h * k_h * self.r_k).sum(dim=-1, keepdim=True) * v_h  # (B, T, H, N)
        output = output + shortcut.reshape(B, T, C)

        # Gate
        output = output * g

        # Output projection
        tm_out = self.output(output)
        x = x + tm_out

        # ── Channel mixing (RWKV-7 style) ──
        x_norm = self.ln_c(x)
        if T > 1:
            x_prev_c = torch.cat([xx.unsqueeze(1), x_norm[:, :-1, :]], dim=1)
        else:
            x_prev_c = xx.unsqueeze(1)

        xx_diff_c = x_prev_c - x_norm
        x_diff_c_h = xx_diff_c.view(B, T, H, N)
        x_norm_h = x_norm.view(B, T, H, N)

        xk_c_h = x_norm_h + x_diff_c_h * self.time_mix_k_c
        xr_c_h = x_norm_h + x_diff_c_h * self.time_mix_r_c

        xk_c = xk_c_h.reshape(B, T, C)
        xr_c = xr_c_h.reshape(B, T, C)

        k_c = torch.relu(self.fc_key(xk_c)) ** 2
        r_c = torch.sigmoid(self.fc_receptance(xr_c))
        v_c = self.fc_value(k_c)
        cm_out = r_c * v_c
        x = x + cm_out

        # ── Update state ──
        new_state = {
            'xx': x[:, -1, :].detach().clone(),
            'state': mat_state[:, :, :, :].detach().clone(),  # (B, H, N, N)
        }

        return x, new_state, ret_v_first


class RWKV7Nano(nn.Module):
    """Tiny RWKV-7 model matching the core architecture of RWKV-7 Goose.

    Args:
        vocab_size: size of the token vocabulary
        dim: embedding / hidden dimension
        head_size: dimension per head (must divide dim)
        num_layers: number of RWKV-7 blocks
        hidden_scale: multiplier for FFN hidden size
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        head_size: int = 32,
        num_layers: int = 3,
        hidden_scale: int = 4,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.head_size = head_size
        self.n_head = dim // head_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([
            RWKV7Block(dim, head_size, hidden_scale) for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() == 1:
                if 'time_mix' in name or 'k_k' in name or 'k_a' in name or 'r_k' in name:
                    nn.init.ones_(p)
                elif 'w0' in name or 'a0' in name or 'v0' in name:
                    nn.init.zeros_(p)
                elif 'ln_w' in name or 'weight' in name:
                    nn.init.ones_(p)
                else:
                    nn.init.zeros_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[list[dict]] = None,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[dict]]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)

        new_state = [] if return_state else None
        v_first = None

        for i, block in enumerate(self.blocks):
            layer_state = state[i] if state is not None else None
            x, s, v_first = block(x, layer_state, v_first)
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
        assert B == 1, "Only batch_size=1 for generation"

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


# ── Backward compatibility aliases ──
# New code uses RWKV7Block/RWKV7Nano (RWKV-7 architecture).
# Old RWKV-4 code was migrated to the new architecture.
# For compatibility with old checkpoints, the legacy classes are preserved
# in theories/archive/.
RWKVBlock = RWKV7Block
RWKVNano = RWKV7Nano


# ── Legacy RWKV-4 block (for loading old checkpoints) ──
# Kept here so from domains.rwkv.rwkv_nano import RWKVBlock still works.
# The old RWKV-4 code was moved to theories/archive/.

class LegacyRWKVBlock(nn.Module):
    """Original RWKV-4 block. Preserved for loading old checkpoints."""
    def __init__(self, dim: int, hidden_scale: int = 4):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_decay = nn.Parameter(torch.full((dim,), -0.5))
        self.time_first = nn.Parameter(torch.zeros(dim))
        self.time_mix_k = nn.Parameter(torch.ones(dim))
        self.time_mix_v = nn.Parameter(torch.ones(dim))
        self.time_mix_r = nn.Parameter(torch.ones(dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        hidden_dim = dim * hidden_scale
        self.channel_mix_k = nn.Parameter(torch.ones(dim))
        self.channel_mix_r = nn.Parameter(torch.ones(dim))
        self.fc_key = nn.Linear(dim, hidden_dim, bias=False)
        self.fc_value = nn.Linear(hidden_dim, dim, bias=False)
        self.fc_receptance = nn.Linear(dim, dim, bias=False)

    def forward(self, x, state=None):
        # Minimal forward stub for parameter loading only
        B, T, C = x.shape
        if state is None:
            xx = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        else:
            xx = state.get('xx', torch.zeros(B, C, device=x.device, dtype=x.dtype))
        new_state = {'xx': x[:, -1, :].detach().clone()}
        return x, new_state

class LegacyRWKVNano(nn.Module):
    """Original RWKV-4 model. Preserved for loading old checkpoints."""
    def __init__(self, vocab_size, dim=128, num_layers=3, hidden_scale=4, pad_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([LegacyRWKVBlock(dim, hidden_scale) for _ in range(num_layers)])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=True)

    def forward(self, input_ids, state=None, return_state=False):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        new_state = []
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


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = RWKV7Nano(vocab_size=258, dim=64, head_size=32, num_layers=2)
    x = torch.randint(1, 257, (2, 32))
    logits, state = model(x, return_state=True)
    print(f"Input:  {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Params: {count_params(model):,}")
    print(f"Heads:  {model.n_head} × head_size={model.head_size}")
    print(f"State:  {state[0]['state'].shape}")  # (B, H, N, N)
    print("OK")
