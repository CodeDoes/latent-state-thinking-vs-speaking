"""Adaptive-loop byte model: Encoder → RWKV-7 Core → Decoder.

Full pipeline from the theory doc:
  Bytes → [RNN Encoder w/ surprise loop] → latents
        → [RWKV-7 Core over latents] → core hidden states
        → [Projection] → decoder input
        → [RNN Decoder w/ adaptive-exit loop] → byte logits

Three recurrence mechanisms:
  1. Encoder time-loop: RNN over bytes, exits via surprise threshold.
  2. RWKV-7 time-loop: recurrent over latents (linear attention).
  3. Decoder depth-loop: iterates R' times per patch with learned exit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from domains.rwkv.rwkv_nano import RWKVBlock
from threads.unsorted.simple_rnn_receptance import SimpleRNNReceptance
from threads.g1g_frontend.surprise_patcher import surprise_per_step, surprise_to_patch_lengths, variable_pool_by_patch


# ── Adaptive Exit Gate ────────────────────────────────────────────────

class AdaptiveExitGate(nn.Module):
    """Learned exit probability for depth loops.

    At loop step r, outputs λ ∈ (0,1): probability of exiting NOW.
    Cumulative exit probability: π_r = λ_r * ∏_{k<r} (1 - λ_k).

    Training: sample r from {1..R} proportional to π_r, or use
    expected loss ∑_r π_r * L_r. We use the expected loss for stability.
    Inference: exit when cumulative π_r > threshold.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(
        self, h: torch.Tensor, max_r: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [batch, seq_len, dim] hidden state at current loop step
            max_r: maximum number of loop steps

        Returns:
            exit_probs: [batch, seq_len] probability of exiting at each step
            cum_exit: [batch, seq_len] cumulative exit probability up to now
        """
        # λ_r = sigmoid(W · h)
        lam = torch.sigmoid(self.proj(h).squeeze(-1))  # [B, T]

        # For expected loss, we need π_r = λ_r * ∏_{k<r} (1 - λ_k)
        # But during forward we just return λ and let the caller compute cumprod.
        # For the cumulative exit check, we accumulate (1-λ) products.
        return lam


def expected_exit_loss(
    exit_lambdas: list[torch.Tensor],
    losses_per_step: list[torch.Tensor],
    entropy_weight: float = 0.01,
) -> torch.Tensor:
    """Compute expected loss over loop depths with entropy regularization.

    Args:
        exit_lambdas: list of λ_r tensors, each [B, T]
        losses_per_step: list of loss tensors L_r, each [B, T] (per-position)
        entropy_weight: weight for entropy bonus

    Returns:
        scalar loss
    """
    R = len(exit_lambdas)
    # Compute π_r for each step
    # π_r = λ_r * ∏_{k<r} (1 - λ_k)
    cum_survive = torch.ones_like(exit_lambdas[0])  # ∏_{k<r} (1 - λ_k)
    expected_loss = torch.zeros_like(exit_lambdas[0])

    pis = []
    for r in range(R):
        pi_r = exit_lambdas[r] * cum_survive  # [B, T]
        pis.append(pi_r)
        expected_loss = expected_loss + pi_r * losses_per_step[r]
        cum_survive = cum_survive * (1 - exit_lambdas[r])

    # Entropy of the exit distribution: H = -∑ π_r log(π_r)
    # Encourages exploration of loop depths
    eps = 1e-7
    entropy = torch.zeros_like(exit_lambdas[0])
    for pi_r in pis:
        entropy = entropy - pi_r * (pi_r + eps).log()
    # Negate: we want to MAXIMIZE entropy, so subtract
    return expected_loss.mean() - entropy_weight * entropy.mean()


# ── Encoder ───────────────────────────────────────────────────────────

class ByteEncoder(nn.Module):
    """RNN byte encoder with surprise-based loop.

    Processes bytes through a stacked RNN. After each pass, checks a
    surprise router. If surprise < threshold, exits; otherwise loops
    the hidden states back for another pass.
    """

    def __init__(self, dim: int, n_layers: int = 2, max_loops: int = 3):
        super().__init__()
        self.dim = dim
        self.max_loops = max_loops

        self.embed = nn.Embedding(258, dim, padding_idx=0)
        self.layers = nn.ModuleList([
            SimpleRNNReceptance(input_dim=dim, hidden_dim=dim)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.surprise_proj = nn.Linear(dim, 1)
        self.ln_out = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: torch.Tensor,
        states: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
        """
        Returns:
            output: [B, T, dim]
            final_states: list of [B, dim] per layer
            info: dict with loop_count, surprise_history
        """
        B, T = tokens.shape
        x = self.embed(tokens)

        if states is None:
            states = [None] * len(self.layers)

        surprise_history = []
        layer_states = list(states) if states is not None else [None] * len(self.layers)

        for loop_idx in range(self.max_loops):
            h = x
            new_states = []
            for i, (layer, ln) in enumerate(zip(self.layers, self.layer_norms)):
                out, h_final, receptance = layer(ln(h), layer_states[i])
                h = h + out  # residual
                new_states.append(h_final)
            layer_states = new_states

            h = self.ln_out(h)

            # Surprise check: mean surprise across sequence
            surprise = torch.sigmoid(self.surprise_proj(h).squeeze(-1))  # [B, T]
            surprise_history.append(surprise.detach().mean().item())

            if surprise.mean().item() < 0.5:
                break

            # Loop: feed hidden states back as input
            x = h.detach()  # detach to prevent gradient through loop

        return h, layer_states, {
            "loop_count": loop_idx + 1,
            "surprise_history": surprise_history,
        }


# ── Pooling ───────────────────────────────────────────────────────────

def pool_bytes_to_latents(
    h: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """Fixed-size mean pooling: [B, T, D] → [B, T//patch_size, D]."""
    B, T, D = h.shape
    k = patch_size
    T_trunc = (T // k) * k
    if T_trunc == 0:
        return h.new_zeros(B, 0, D)
    return h[:, :T_trunc, :].view(B, T_trunc // k, k, D).mean(dim=2)


def dynamic_pool_bytes_to_latents(
    h: torch.Tensor,
    threshold: float = 0.7,
    min_patch: int = 2,
    max_patch: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Surprise-based dynamic pooling: [B, T, D] → [B, J_max, D].

    Returns:
        latents: [B, J_max, D] mean-pooled per patch (padded)
        patch_lengths: [B, J_max] actual patch lengths (0 for padding)
        patch_counts: [B] number of real patches per batch element
    """
    B, T, D = h.shape
    surprise = surprise_per_step(h)  # [B, T-1]
    patch_lengths = surprise_to_patch_lengths(
        surprise, threshold=threshold, min_patch=min_patch, max_patch=max_patch
    )  # [B, J_max]
    J_max = patch_lengths.shape[1]

    # Mean-pool within variable patches
    latents = h.new_zeros(B, J_max, D)
    actual_counts = []
    for b in range(B):
        cur = 0
        count = 0
        for j in range(J_max):
            L = int(patch_lengths[b, j].item())
            if L == 0:
                break
            latents[b, j] = h[b, cur:cur+L].mean(dim=0)
            cur += L
            count += 1
        actual_counts.append(count)

    return latents, patch_lengths, torch.tensor(actual_counts, dtype=torch.long, device=h.device)


def broadcast_variable_patches(
    core_out: torch.Tensor,
    patch_lengths: torch.Tensor,
    target_T: int,
) -> torch.Tensor:
    """Broadcast variable-length patch outputs back to byte positions.

    Args:
        core_out: [B, J_max, D] core output per patch
        patch_lengths: [B, J_max] patch lengths (0 for padding)
        target_T: target sequence length (original byte count)

    Returns:
        [B, target_T, D] broadcasted (truncated to target_T)
    """
    B, J_max, D = core_out.shape
    out = core_out.new_zeros(B, target_T, D)
    for b in range(B):
        cur = 0
        for j in range(J_max):
            L = int(patch_lengths[b, j].item())
            if L == 0:
                break
            end = min(cur + L, target_T)
            out[b, cur:end] = core_out[b, j]
            cur = end
            if cur >= target_T:
                break
    return out


# ── RWKV-7 Core ───────────────────────────────────────────────────────

class RWKV7Core(nn.Module):
    """Stacked RWKV-7 blocks operating on latent sequence.

    This is the "global slow dynamics" layer: processes the compressed
    latent sequence with linear-complexity recurrence.
    """

    def __init__(self, dim: int, n_layers: int = 2, hidden_scale: int = 4):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            RWKVBlock(dim, hidden_scale) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)

    def forward(
        self,
        z: torch.Tensor,
        states: Optional[list[dict]] = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        Args:
            z: [B, J, dim] latent sequence
            states: optional list of per-layer RWKV states

        Returns:
            h: [B, J, dim] core output
            new_states: list of updated states
        """
        new_states = []
        h = z
        for i, block in enumerate(self.blocks):
            layer_state = states[i] if states is not None else None
            h, s = block(h, layer_state)
            new_states.append(s)
        h = self.ln_out(h)
        return h, new_states


# ── Looped RWKV-7 Core ───────────────────────────────────────────────

class LoopedRWKV7Core(nn.Module):
    """RWKV-7 core with optional depth-looping + adaptive exit.

    For each latent step, can iterate the RWKV block R times.
    The adaptive exit gate decides when to stop iterating.

    Two modes:
      - shared_state: state accumulates across loops (iterative refinement)
      - reset_state: state resets each loop (treats loops as depth layers)
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 2,
        max_depth_loops: int = 1,
        hidden_scale: int = 4,
        shared_state: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_depth_loops = max_depth_loops
        self.shared_state = shared_state

        self.core = RWKV7Core(dim, n_layers, hidden_scale)
        self.exit_gate = AdaptiveExitGate(dim) if max_depth_loops > 1 else None
        # Projection to align state between loops if not shared
        self.loop_proj = nn.Linear(dim, dim) if not shared_state else None

    def forward(
        self,
        z: torch.Tensor,
        states: Optional[list[dict]] = None,
    ) -> tuple[torch.Tensor, list[dict], dict]:
        """
        Returns:
            h: [B, J, dim] output
            new_states: list of per-layer states
            info: dict with exit_lambdas, depth_loop_count
        """
        if self.max_depth_loops == 1:
            h, new_states = self.core(z, states)
            return h, new_states, {"exit_lambdas": [], "depth_loop_count": 1}

        exit_lambdas = []
        h = z
        current_states = states
        depth_count = self.max_depth_loops

        for r in range(self.max_depth_loops):
            h, new_states = self.core(h, current_states)

            # Check exit gate (collect lambdas for training loss)
            lam = self.exit_gate(h, r)
            exit_lambdas.append(lam)

            # Early exit only during inference (no grad)
            if not self.training and lam.mean().item() > 0.5 and r > 0:
                depth_count = r + 1
                break

            if r < self.max_depth_loops - 1:
                if self.shared_state:
                    current_states = new_states
                else:
                    current_states = None
                    if self.loop_proj is not None:
                        h = self.loop_proj(h)

        return h, new_states, {
            "exit_lambdas": exit_lambdas,
            "depth_loop_count": depth_count,
        }


# ── Decoder ───────────────────────────────────────────────────────────

class ByteDecoder(nn.Module):
    """RNN byte decoder with adaptive-exit depth loop.

    Takes core hidden states broadcast to byte positions, plus encoder
    output. Iterates R' times with learned exit per byte position.
    """

    def __init__(self, dim: int, n_layers: int = 2, max_loops: int = 3):
        super().__init__()
        self.dim = dim
        self.max_loops = max_loops

        self.layers = nn.ModuleList([
            SimpleRNNReceptance(input_dim=dim, hidden_dim=dim)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, 258)
        self.ln_out = nn.LayerNorm(dim)

        # Adaptive exit (only if max_loops > 1)
        self.exit_gate = AdaptiveExitGate(dim) if max_loops > 1 else None

    def forward(
        self,
        encoder_out: torch.Tensor,
        core_out_broadcast: torch.Tensor,
        states: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
        """
        Args:
            encoder_out: [B, T, dim] from byte encoder
            core_out_broadcast: [B, T, dim] core output broadcast to byte positions
            states: previous decoder states

        Returns:
            logits: [B, T, 258]
            final_states: list of [B, dim]
            info: dict with exit_lambdas, depth_loop_count, losses_per_step
        """
        B, T, D = encoder_out.shape
        x = encoder_out + core_out_broadcast  # combine local + global

        if states is None:
            states = [None] * len(self.layers)

        layer_states = list(states)
        exit_lambdas = []
        losses_per_step = []
        best_logits = None
        best_loss = torch.tensor(float('inf'), device=x.device)
        depth_count = 1

        for r in range(self.max_loops):
            h = x
            new_states = []
            for i, (layer, ln) in enumerate(zip(self.layers, self.layer_norms)):
                out, h_final, receptance = layer(ln(h), layer_states[i])
                h = h + out
                new_states.append(h_final)
            layer_states = new_states
            h = self.ln_out(h)

            logits = self.head(h)  # [B, T, 258]
            losses_per_step.append(logits.detach())

            # Track best logits by loss (for expected loss computation)
            if best_logits is None:
                best_logits = logits

            if self.exit_gate is not None and r < self.max_loops - 1:
                lam = self.exit_gate(h, r)
                exit_lambdas.append(lam)

                # Early exit only during inference
                if not self.training and lam.mean().item() > 0.5 and r > 0:
                    best_logits = logits
                    depth_count = r + 1
                    break

            # Loop: feed output back
            if r < self.max_loops - 1:
                x = h.detach()  # detach for depth loop stability

        return best_logits, layer_states, {
            "exit_lambdas": exit_lambdas,
            "depth_loop_count": depth_count,
        }


# ── Full Model ────────────────────────────────────────────────────────

class AdaptiveLoopModel(nn.Module):
    """Full pipeline: Encoder → RWKV-7 Core → Decoder.

    Architecture:
        bytes → [RNN Encoder w/ surprise loop] → latents (mean-pooled)
              → [RWKV-7 Core over latents] → core hidden states
              → [broadcast to byte positions] + encoder_out
              → [RNN Decoder w/ adaptive-exit loop] → byte logits
    """

    def __init__(
        self,
        dim: int = 64,
        patch_size: int = 4,
        enc_layers: int = 2,
        core_layers: int = 2,
        dec_layers: int = 2,
        enc_max_loops: int = 3,
        core_depth_loops: int = 1,
        dec_max_loops: int = 3,
        core_hidden_scale: int = 4,
        shared_core_state: bool = True,
        dynamic_patch: bool = False,
        patch_threshold: float = 0.7,
        min_patch: int = 2,
        max_patch: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.dynamic_patch = dynamic_patch
        self.patch_threshold = patch_threshold
        self.min_patch = min_patch
        self.max_patch = max_patch

        self.encoder = ByteEncoder(dim, enc_layers, enc_max_loops)
        self.core = LoopedRWKV7Core(
            dim, core_layers, core_depth_loops, core_hidden_scale, shared_core_state
        )
        self.decoder = ByteDecoder(dim, dec_layers, dec_max_loops)
        self.to_core = nn.Linear(dim, dim)  # project encoder latents to core dim

    def forward(
        self,
        tokens: torch.Tensor,
        enc_states: Optional[list[torch.Tensor]] = None,
        core_states: Optional[list[dict]] = None,
        dec_states: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            tokens: [B, T] byte token ids (1..257)

        Returns:
            logits: [B, T, 258]
            info: dict with all sub-component info
        """
        B, T = tokens.shape

        # 1. Encode bytes → hidden states
        encoder_out, enc_final_states, enc_info = self.encoder(tokens, enc_states)

        # 2. Pool to latents (fixed or dynamic)
        if self.dynamic_patch:
            latents, patch_lengths, patch_counts = dynamic_pool_bytes_to_latents(
                encoder_out, self.patch_threshold, self.min_patch, self.max_patch
            )
        else:
            latents = pool_bytes_to_latents(encoder_out, self.patch_size)
            patch_lengths = None
            patch_counts = None

        latents_proj = self.to_core(latents)

        # 3. RWKV-7 core over latents
        core_out, core_final_states, core_info = self.core(latents_proj, core_states)

        # 4. Project core output back to byte dim and broadcast
        J = core_out.shape[1]
        if self.dynamic_patch:
            T_trunc = int(patch_lengths.sum(dim=1).max().item())
            enc_trunc = encoder_out[:, :T_trunc, :]
            core_broadcast = broadcast_variable_patches(
                core_out, patch_lengths, T_trunc
            )
        else:
            k = self.patch_size
            T_trunc = J * k
            enc_trunc = encoder_out[:, :T_trunc, :]
            core_broadcast = core_out.repeat_interleave(k, dim=1)[:, :T_trunc, :]

        # 5. Decode bytes
        dec_out, dec_final_states, dec_info = self.decoder(enc_trunc, core_broadcast, dec_states)

        # Pad logits back to original T if needed
        if dec_out.shape[1] < T:
            pad = dec_out.new_zeros(B, T - dec_out.shape[1], 258)
            logits = torch.cat([dec_out, pad], dim=1)
        else:
            logits = dec_out[:, :T, :]

        # Compute effective patch count for stats
        if self.dynamic_patch:
            n_latents = patch_counts.float().mean().item() if patch_counts is not None else J
        else:
            n_latents = J

        info = {
            "encoder": enc_info,
            "core": core_info,
            "decoder": dec_info,
            "n_latents": n_latents,
            "compression_ratio": T / max(n_latents, 1),
            "dynamic_patch": self.dynamic_patch,
            "states": {
                "enc": enc_final_states,
                "core": core_final_states,
                "dec": dec_final_states,
            },
        }

        return logits, info

    def get_exit_stats(self, info: dict) -> dict:
        """Extract adaptive exit statistics for logging."""
        stats = {
            "enc_loops": info["encoder"]["loop_count"],
            "core_depth_loops": info["core"]["depth_loop_count"],
            "dec_loops": info["decoder"]["depth_loop_count"],
            "n_latents": info["n_latents"],
            "compression_ratio": info["compression_ratio"],
        }
        if info["core"]["exit_lambdas"]:
            core_lams = torch.stack(info["core"]["exit_lambdas"])
            stats["core_exit_mean"] = core_lams.mean().item()
        if info["decoder"]["exit_lambdas"]:
            dec_lams = torch.stack(info["decoder"]["exit_lambdas"])
            stats["dec_exit_mean"] = dec_lams.mean().item()
        return stats


# ── Smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T = 2, 64
    tokens = torch.randint(1, 256, (B, T))
    targets = torch.randint(1, 256, (B, T))

    for dynamic in [False, True]:
        label = "dynamic" if dynamic else "fixed"
        print(f"\n{'='*50}")
        print(f"Mode: {label} patches")
        print(f"{'='*50}")

        model = AdaptiveLoopModel(
            dim=64,
            patch_size=4,
            enc_layers=2,
            core_layers=2,
            dec_layers=2,
            enc_max_loops=3,
            core_depth_loops=2,
            dec_max_loops=3,
            dynamic_patch=dynamic,
            patch_threshold=0.7,
            min_patch=2,
            max_patch=16,
        )

        logits, info = model(tokens)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {n_params:,}")
        print(f"  encoder:  {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  core:     {sum(p.numel() for p in model.core.parameters()):,}")
        print(f"  decoder:  {sum(p.numel() for p in model.decoder.parameters()):,}")

        stats = model.get_exit_stats(info)
        print(f"  logits: {logits.shape}")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Gradient check
        loss = torch.nn.functional.cross_entropy(
            logits[:, :T-1, :].reshape(-1, 258), targets[:, :T-1].reshape(-1)
        )
        loss.backward()

        n_zero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0)
        n_total = sum(1 for _ in model.parameters())
        print(f"Gradient: {n_zero}/{n_total} zero-grad params")
        print("OK" if n_zero == 0 else "ISSUE")
        model.zero_grad()
