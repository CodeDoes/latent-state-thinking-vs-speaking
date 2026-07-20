"""Delta-Rule State Memory and RWKV-State Memory in pure PyTorch.

Adapts the Delta-Mem and RWKV memory concept for low-rank query, key, value,
and output delta-injections into attention modules.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_DELTA_MEM_HEADS = ("q", "k", "v", "o")
VALID_DELTA_MEM_STATE_UPDATE_MODES = ("standard", "lambda_outside", "no_lambda")
VALID_DELTA_MEM_OUTPUT_INITS = ("zero", "small", "random")


def normalize_delta_mem_heads(heads: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(heads, str):
        items = tuple(part.strip().lower() for part in heads.split(",") if part.strip())
    else:
        items = tuple(str(part).strip().lower() for part in heads if str(part).strip())
    if not items or items == ("none",):
        return ()
    invalid = sorted(set(items) - set(VALID_DELTA_MEM_HEADS))
    if invalid:
        raise ValueError(f"Unsupported delta-memory heads: {invalid}; expected subset of {VALID_DELTA_MEM_HEADS}")
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return tuple(out)


class DeltaRuleStateMemory(nn.Module):
    """Delta-Mem-style online associative memory for attention modules.

    Hidden states are projected to low-rank memory q/k/v. The state is read
    before each token write, then projected into attention q/k/v/o deltas.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        query_size: int,
        key_size: int,
        value_size: int,
        output_size: int,
        rank: int = 8,
        num_state_heads: int = 1,
        alpha: float = 16.0,
        beta_bias_init: float = -1.5,
        normalize_qk: bool = True,
        couple_lambda: bool = True,
        state_update_mode: str = "standard",
        rankwise_gates: bool = True,
        delta_heads: Sequence[str] | str = ("q", "k", "v", "o"),
        output_init: str = "zero",
        output_init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if num_state_heads < 1:
            raise ValueError("num_state_heads must be >= 1")
        if state_update_mode not in VALID_DELTA_MEM_STATE_UPDATE_MODES:
            raise ValueError(
                f"Unsupported state_update_mode={state_update_mode!r}; "
                f"expected one of {VALID_DELTA_MEM_STATE_UPDATE_MODES}"
            )
        if output_init not in VALID_DELTA_MEM_OUTPUT_INITS:
            raise ValueError(
                f"Unsupported output_init={output_init!r}; "
                f"expected one of {VALID_DELTA_MEM_OUTPUT_INITS}"
            )

        self.hidden_size = hidden_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.rank = rank
        self.num_state_heads = num_state_heads
        self.state_read_dim = rank * num_state_heads
        self.alpha = alpha
        self.delta_scaling = alpha / rank
        self.beta_bias_init = beta_bias_init
        self.normalize_qk = normalize_qk
        self.couple_lambda = couple_lambda
        self.state_update_mode = state_update_mode
        self.rankwise_gates = rankwise_gates
        self.delta_heads = normalize_delta_mem_heads(delta_heads)

        gate_dim_per_head = rank if rankwise_gates else 1
        self.gate_dim = gate_dim_per_head * num_state_heads

        # Projections to memory space
        self.memory_q_proj = nn.Linear(hidden_size, self.state_read_dim, bias=False)
        self.memory_k_proj = nn.Linear(hidden_size, self.state_read_dim, bias=False)
        self.memory_v_proj = nn.Linear(hidden_size, self.state_read_dim, bias=False)

        # Gates
        self.beta_proj = nn.Linear(hidden_size, self.gate_dim, bias=False)
        self.beta_bias = nn.Parameter(torch.full((self.gate_dim,), beta_bias_init))
        if self.couple_lambda:
            self.lambda_proj = None
            self.lambda_bias = None
        else:
            self.lambda_proj = nn.Linear(hidden_size, self.gate_dim, bias=False)
            self.lambda_bias = nn.Parameter(torch.full((self.gate_dim,), -beta_bias_init))

        # Projections from memory space to attention deltas
        self.delta_q_proj = nn.Linear(self.state_read_dim, query_size, bias=False)
        self.delta_k_proj = nn.Linear(self.state_read_dim, key_size, bias=False)
        self.delta_v_proj = nn.Linear(self.state_read_dim, value_size, bias=False)
        self.delta_o_proj = nn.Linear(self.state_read_dim, output_size, bias=False)

        self.reset_parameters(output_init, output_init_scale)

    def reset_parameters(self, output_init: str, output_init_scale: float) -> None:
        for proj in (self.memory_q_proj, self.memory_k_proj, self.memory_v_proj):
            nn.init.kaiming_uniform_(proj.weight, a=5**0.5)
        nn.init.zeros_(self.beta_proj.weight)
        if self.lambda_proj is not None:
            nn.init.zeros_(self.lambda_proj.weight)

        # Reset delta projections based on init choice
        for proj in (self.delta_q_proj, self.delta_k_proj, self.delta_v_proj, self.delta_o_proj):
            if output_init == "zero":
                nn.init.zeros_(proj.weight)
            elif output_init == "small":
                nn.init.trunc_normal_(
                    proj.weight,
                    mean=0.0,
                    std=output_init_scale,
                    a=-3 * output_init_scale,
                    b=3 * output_init_scale,
                )
            elif output_init == "random":
                nn.init.kaiming_uniform_(proj.weight, a=5**0.5)

    def _reshape_gate(self, gate: torch.Tensor) -> torch.Tensor:
        if self.rankwise_gates:
            return gate.view(*gate.shape[:-1], self.num_state_heads, self.rank)
        return gate.view(*gate.shape[:-1], self.num_state_heads, 1).expand(
            *gate.shape[:-1], self.num_state_heads, self.rank
        )

    def _normalize_memory_projection(self, projected: torch.Tensor) -> torch.Tensor:
        if self.normalize_qk:
            original_dtype = projected.dtype
            projected = projected.float().view(*projected.shape[:-1], self.num_state_heads, self.rank)
            projected = F.normalize(torch.tanh(projected), dim=-1, eps=1e-6)
            projected = projected.reshape(*projected.shape[:-2], self.state_read_dim).to(dtype=original_dtype)
        return projected

    def _project_memory(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        memory_q = self._normalize_memory_projection(self.memory_q_proj(x))
        memory_k = self._normalize_memory_projection(self.memory_k_proj(x))
        memory_v = self.memory_v_proj(x)

        beta = torch.sigmoid(self.beta_proj(x) + self.beta_bias.to(device=x.device, dtype=x.dtype))
        beta = self._reshape_gate(beta)

        if self.state_update_mode == "no_lambda":
            lam = torch.ones_like(beta)
        elif self.couple_lambda:
            lam = 1.0 - beta
        else:
            assert self.lambda_proj is not None and self.lambda_bias is not None
            lam = torch.sigmoid(self.lambda_proj(x) + self.lambda_bias.to(device=x.device, dtype=x.dtype))
            lam = self._reshape_gate(lam)

        return memory_q, memory_k, memory_v, beta, lam

    def _update_coefficients(self, beta: torch.Tensor, lam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.state_update_mode == "standard":
            return lam, beta, beta
        if self.state_update_mode == "lambda_outside":
            return lam, lam * beta, beta
        if self.state_update_mode == "no_lambda":
            return torch.ones_like(beta), beta, beta
        raise ValueError(f"Unsupported state_update_mode: {self.state_update_mode}")

    def _scan_batched_torch(
        self,
        memory_q: torch.Tensor,
        memory_k: torch.Tensor,
        memory_v: torch.Tensor,
        keep: torch.Tensor,
        erase: torch.Tensor,
        write: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = memory_q.shape
        if initial_state is None:
            state = torch.zeros(batch_size, self.rank, self.rank, device=memory_q.device, dtype=torch.float32)
        else:
            state = initial_state.float()

        reads: list[torch.Tensor] = []
        q_seq = memory_q.float()
        k_seq = memory_k.float()
        v_seq = memory_v.float()

        for token_idx in range(seq_len):
            q_t = q_seq[:, token_idx, :]
            k_t = k_seq[:, token_idx, :]
            v_t = v_seq[:, token_idx, :]
            keep_t = keep[:, token_idx, :].unsqueeze(-1)
            erase_t = erase[:, token_idx, :].unsqueeze(-1)
            write_t = write[:, token_idx, :].unsqueeze(-1)

            # Read-before-write timing
            read_t = torch.einsum("bij,bj->bi", state, q_t)
            pred_t = torch.einsum("bij,bj->bi", state, k_t)

            state = (
                keep_t * state
                - erase_t * pred_t.unsqueeze(-1) * k_t.unsqueeze(1)
                + write_t * v_t.unsqueeze(-1) * k_t.unsqueeze(1)
            )

            reads.append(read_t.to(dtype=memory_q.dtype))

        return state, torch.stack(reads, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Processes sequence x [B, T, D] or [T, D] and returns delta heads."""
        is_flat = x.dim() == 2
        if is_flat:
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.shape
        memory_q, memory_k, memory_v, beta, lam = self._project_memory(x)
        keep, erase, write = self._update_coefficients(beta, lam)

        # Reshape to scan over head and batch dimensions
        flat_batch = batch_size * self.num_state_heads
        q_seq = memory_q.float().view(batch_size, seq_len, self.num_state_heads, self.rank)
        k_seq = memory_k.float().view(batch_size, seq_len, self.num_state_heads, self.rank)
        v_seq = memory_v.float().view(batch_size, seq_len, self.num_state_heads, self.rank)

        q_seq = q_seq.permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        k_seq = k_seq.permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        v_seq = v_seq.permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)

        keep_seq = keep.float().permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        erase_seq = erase.float().permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        write_seq = write.float().permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)

        _, reads_flat = self._scan_batched_torch(
            memory_q=q_seq,
            memory_k=k_seq,
            memory_v=v_seq,
            keep=keep_seq,
            erase=erase_seq,
            write=write_seq,
        )

        reads = reads_flat.reshape(batch_size, self.num_state_heads, seq_len, self.rank)
        reads = reads.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.state_read_dim).to(dtype=x.dtype)

        # Project reads to deltas
        deltas = {}
        if "q" in self.delta_heads:
            deltas["q"] = self.delta_q_proj(reads) * self.delta_scaling
        if "k" in self.delta_heads:
            deltas["k"] = self.delta_k_proj(reads) * self.delta_scaling
        if "v" in self.delta_heads:
            deltas["v"] = self.delta_v_proj(reads) * self.delta_scaling
        if "o" in self.delta_heads:
            deltas["o"] = self.delta_o_proj(reads) * self.delta_scaling

        if is_flat:
            for k in deltas:
                deltas[k] = deltas[k].squeeze(0)

        return deltas


class RWKVStateMemory(nn.Module):
    """RWKV-4/7 style state reader used to produce attention deltas."""

    def __init__(
        self,
        *,
        hidden_size: int,
        query_size: int,
        key_size: int,
        value_size: int,
        output_size: int,
        delta_heads: Sequence[str] | str = ("q", "k", "v", "o"),
        output_init: str = "zero",
        output_init_scale: float = 0.02,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.scale = scale
        self.delta_heads = normalize_delta_mem_heads(delta_heads)

        # Standard linear layers for RWKV mixing and recurrence
        self.ln = nn.LayerNorm(hidden_size)
        self.time_mix_k = nn.Parameter(torch.ones(hidden_size))
        self.time_mix_v = nn.Parameter(torch.ones(hidden_size))
        self.time_mix_r = nn.Parameter(torch.ones(hidden_size))

        self.time_decay = nn.Parameter(torch.full((hidden_size,), -0.5))
        self.time_first = nn.Parameter(torch.zeros(hidden_size))

        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)

        # Outputs
        self.delta_q_proj = nn.Linear(hidden_size, query_size, bias=False)
        self.delta_k_proj = nn.Linear(hidden_size, key_size, bias=False)
        self.delta_v_proj = nn.Linear(hidden_size, value_size, bias=False)
        self.delta_o_proj = nn.Linear(hidden_size, output_size, bias=False)

        self.reset_parameters(output_init, output_init_scale)

    def reset_parameters(self, output_init: str, output_init_scale: float) -> None:
        nn.init.uniform_(self.time_decay, -6.0, -1.0)
        nn.init.constant_(self.time_first, 0.0)

        for proj in (self.delta_q_proj, self.delta_k_proj, self.delta_v_proj, self.delta_o_proj):
            if output_init == "zero":
                nn.init.zeros_(proj.weight)
            elif output_init == "small":
                nn.init.trunc_normal_(
                    proj.weight,
                    mean=0.0,
                    std=output_init_scale,
                    a=-3 * output_init_scale,
                    b=3 * output_init_scale,
                )
            elif output_init == "random":
                nn.init.kaiming_uniform_(proj.weight, a=5**0.5)

    def _cumlogsumexp(self, x: torch.Tensor) -> torch.Tensor:
        cmax = torch.cummax(x, dim=1).values
        shifted = x - cmax
        exp_sum = torch.cumsum(torch.exp(shifted), dim=1)
        return cmax + torch.log(exp_sum + 1e-10)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        is_flat = x.dim() == 2
        if is_flat:
            x = x.unsqueeze(0)

        B, T, C = x.shape
        x_norm = self.ln(x)

        # Time mixing (recurrent-style shift-free simulation over the sequence)
        xx = torch.zeros_like(x_norm)
        xx[:, 1:] = x_norm[:, :-1]

        xk = x_norm * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        # WKV calculation (read-before-write timing, i.e., token t reads past state S_{t-1})
        w = self.time_decay
        u = self.time_first
        log_w = -torch.exp(w)

        t_idx = torch.arange(T, device=x.device).view(1, T, 1).float()
        log_term_num = -t_idx * log_w + k + torch.log(torch.abs(v) + 1e-8)
        log_term_den = -t_idx * log_w + k
        sign_num = torch.sign(v)

        cum_num = self._cumlogsumexp(log_term_num)
        cum_den = self._cumlogsumexp(log_term_den)

        num = torch.exp(cum_num + t_idx * log_w) * sign_num
        den = torch.exp(cum_den + t_idx * log_w)

        # Shift to achieve read-before-write (current step does not see its own write yet)
        num_shifted = torch.zeros_like(num)
        den_shifted = torch.zeros_like(den)
        num_shifted[:, 1:] = num[:, :-1]
        den_shifted[:, 1:] = den[:, :-1]

        wkv_out = num_shifted / (den_shifted + 1e-8)
        out_state = r * wkv_out * self.scale

        # Project to delta heads
        deltas = {}
        if "q" in self.delta_heads:
            deltas["q"] = self.delta_q_proj(out_state)
        if "k" in self.delta_heads:
            deltas["k"] = self.delta_k_proj(out_state)
        if "v" in self.delta_heads:
            deltas["v"] = self.delta_v_proj(out_state)
        if "o" in self.delta_heads:
            deltas["o"] = self.delta_o_proj(out_state)

        if is_flat:
            for k in deltas:
                deltas[k] = deltas[k].squeeze(0)

        return deltas
