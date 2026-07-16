"""Byte-level adaptive encoder↔decoder with PATCH compression.

BYTES → [RNN Encoder] → LATENTS (patch-level)
      → (byte_pred | TRIGGER) per latent
      → [RNN Decoder at triggered latents] → BYTES

Encoder runs at patch stride. Decoder only runs where encoder triggers.
Trigger cost in loss → compression pressure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TriggerHead(nn.Module):
    """Outputs: byte logits (258) + trigger logit (1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.byte_head = nn.Linear(dim, 258)
        self.trigger_head = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.byte_head(h), self.trigger_head(h).squeeze(-1)


class ByteRNN(nn.Module):
    """RNN with receptance gating."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W_in = nn.Linear(dim, dim, bias=False)
        self.W_rec = nn.Linear(dim, dim, bias=False)
        self.W_recept = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        if state is None:
            state = x.new_zeros(B, D)

        h = state
        outs = []
        for t in range(T):
            h_in = self.W_in(x[:, t])
            h_rec = self.W_rec(h)
            r = torch.sigmoid(self.W_recept(x[:, t]))
            h = r * h_in + (1 - r) * h_rec
            outs.append(h)
        return torch.stack(outs, dim=1), h


class PatchEncoder(nn.Module):
    """RNN encoder operating at patch level.

    Input: [B, T] bytes → embed → pool to patches → RNN over patches
    Output per patch: (byte_logits_for_patch, trigger_logit)
    """

    def __init__(
        self,
        dim: int = 64,
        patch_size: int = 8,
        n_layers: int = 2,
        min_patches_before_trigger: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.min_patches = min_patches_before_trigger

        self.embed = nn.Embedding(258, dim, padding_idx=0)
        self.patch_pool = nn.AvgPool1d(patch_size, stride=patch_size)  # [B, dim, T] -> [B, dim, J]

        self.layers = nn.ModuleList([ByteRNN(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = TriggerHead(dim)

        # Positional trigger bias: suppress early patches
        self.register_buffer("pos_bias", torch.zeros(256))
        with torch.no_grad():
            self.pos_bias[:min_patches_before_trigger] = -3.0

    def forward(
        self,
        tokens: torch.Tensor,
        states: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict]:
        """
        Returns:
            byte_logits: [B, J, 258] — predicted byte distribution per patch (mean over patch)
            trigger_logits: [B, J] — trigger per patch
            final_states: list of [B, D]
            info: dict
        """
        B, T = tokens.shape
        J = T // self.patch_size
        T_trunc = J * self.patch_size

        # Embed bytes: [B, T] -> [B, T, D]
        x = self.embed(tokens[:, :T_trunc])

        # Pool to patches: [B, T, D] -> [B, D, T] -> pool -> [B, D, J] -> [B, J, D]
        x = x.transpose(1, 2)
        x = self.patch_pool(x)  # [B, D, J]
        x = x.transpose(1, 2)   # [B, J, D]

        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        h = x
        for i, (layer, state) in enumerate(zip(self.layers, states)):
            h, h_final = layer(h, state)
            new_states.append(h_final)

        h = self.ln(h)
        byte_logits, trigger_logits = self.head(h)  # [B, J, 258], [B, J]

        # Positional trigger bias
        seq_len = min(J, self.pos_bias.shape[0])
        trigger_logits[:, :seq_len] = trigger_logits[:, :seq_len] + self.pos_bias[:seq_len]

        return byte_logits, trigger_logits, new_states, {
            "trigger_prob": torch.sigmoid(trigger_logits).mean().item(),
            "n_patches": J,
        }


class SparseDecoder(nn.Module):
    """Decoder that only runs at triggered patch positions.

    Takes triggered patch latents, autoregressively decodes bytes within patch.
    """

    def __init__(self, dim: int = 64, patch_size: int = 8, n_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

        self.embed = nn.Embedding(258, dim, padding_idx=0)
        self.layers = nn.ModuleList([ByteRNN(dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 258)

    def forward(
        self,
        patch_latents: torch.Tensor,  # [B, K, D] — latents at K triggered patches
        target_bytes: Optional[torch.Tensor] = None,  # [B, K, patch_size] for teacher forcing
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            byte_logits: [B, K, patch_size, 258] — predictions per byte in patch
            info: dict
        """
        B, K, D = patch_latents.shape

        if target_bytes is not None:
            # Teacher forcing: [B, K, patch_size] -> [B, K*patch_size]
            target_flat = target_bytes.reshape(B, K * self.patch_size)
            # Embed and pool each patch's bytes to condition
            # For simplicity: decode each patch independently from its latent
            byte_logits = []
            for k in range(K):
                h = patch_latents[:, k]  # [B, D]
                states = [None] * len(self.layers)
                patch_logits = []
                # Start with patch latent as initial state for each layer
                for _ in range(self.patch_size):
                    x = h.unsqueeze(1)  # [B, 1, D]
                    new_states = []
                    for i, (layer, state) in enumerate(zip(self.layers, states)):
                        out, h_final = layer(x, state)
                        h = out[:, 0]  # [B, D]
                        new_states.append(h_final)
                    states = new_states
                    h = self.ln(h)
                    patch_logits.append(self.head(h))  # [B, 258]
                byte_logits.append(torch.stack(patch_logits, dim=1))  # [B, patch_size, 258]
            byte_logits = torch.stack(byte_logits, dim=1)  # [B, K, patch_size, 258]
        else:
            # Inference: autoregressive per patch
            byte_logits = patch_latents.new_zeros(B, K, self.patch_size, 258)
            for k in range(K):
                h = patch_latents[:, k]
                states = [None] * len(self.layers)
                for p in range(self.patch_size):
                    x = h.unsqueeze(1)
                    new_states = []
                    for i, (layer, state) in enumerate(zip(self.layers, states)):
                        out, h_final = layer(x, state)
                        h = out[:, 0]
                        new_states.append(h_final)
                    states = new_states
                    h = self.ln(h)
                    byte_logits[:, k, p] = self.head(h)

        return byte_logits, {"n_triggered": K}


class PatchLoopModel(nn.Module):
    """Patch-level encoder + sparse decoder with trigger cost."""

    def __init__(
        self,
        dim: int = 64,
        patch_size: int = 8,
        enc_layers: int = 2,
        dec_layers: int = 2,
        min_patches_before_trigger: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

        self.encoder = PatchEncoder(dim, patch_size, enc_layers, min_patches_before_trigger)
        self.decoder = SparseDecoder(dim, patch_size, dec_layers)
        self.latent_proj = nn.Linear(258, dim)  # encoder byte_logits -> decoder latent

    def forward(
        self,
        tokens: torch.Tensor,
        enc_states: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            byte_logits: [B, T, 258] — full sequence predictions
            info: dict with trigger stats, latent count
        """
        B, T = tokens.shape

        # 1. Encode at patch level
        enc_byte_logits, enc_trigger_logits, enc_states, enc_info = self.encoder(tokens, enc_states)
        # enc_byte_logits: [B, J, 258], enc_trigger_logits: [B, J]

        trigger_prob = torch.sigmoid(enc_trigger_logits)  # [B, J]

        # 2. Select triggered patches
        triggered_mask = trigger_prob > 0.5  # [B, J]
        n_triggered = triggered_mask.sum(dim=1)  # [B]

        # 3. Get latents at triggered positions
        # Project encoder byte_logits to dim as proxy latent
        patch_latents = self.latent_proj(enc_byte_logits.detach())  # [B, J, D]

        # Gather triggered latents per batch
        max_triggered = n_triggered.max().item()
        if max_triggered == 0:
            # No triggers: use all patches, decoder decodes everything
            triggered_latents = patch_latents
            triggered_mask = torch.ones_like(triggered_mask)
            max_triggered = J
        else:
            triggered_latents = torch.zeros(B, max_triggered, self.dim, device=tokens.device)
            for b in range(B):
                lat = patch_latents[b, triggered_mask[b]]  # [K, D]
                triggered_latents[b, :lat.shape[0]] = lat

        # 4. Target bytes for triggered patches (teacher forcing)
        J = enc_trigger_logits.shape[1]
        T_trunc = J * self.patch_size
        target_patches = tokens[:, :T_trunc].reshape(B, J, self.patch_size)  # [B, J, P]
        triggered_targets = torch.zeros(B, max_triggered, self.patch_size, dtype=torch.long, device=tokens.device)
        for b in range(B):
            tgt = target_patches[b, triggered_mask[b]]  # [K, P]
            if tgt.shape[0] > 0:
                triggered_targets[b, :tgt.shape[0]] = tgt

        # 5. Decode triggered patches
        dec_byte_logits, dec_info = self.decoder(triggered_latents, triggered_targets)
        # dec_byte_logits: [B, K, P, 258]

        # 6. Scatter back to full sequence
        full_logits = torch.zeros(B, T_trunc, 258, device=tokens.device)
        for b in range(B):
            k_idx = 0
            for j in range(J):
                if triggered_mask[b, j]:
                    full_logits[b, j*self.patch_size:(j+1)*self.patch_size] = dec_byte_logits[b, k_idx]
                    k_idx += 1
                else:
                    # Use encoder prediction for non-triggered patches (broadcast to patch)
                    full_logits[b, j*self.patch_size:(j+1)*self.patch_size] = enc_byte_logits[b, j].unsqueeze(0).repeat(self.patch_size, 1)

        # Pad to original T
        if T_trunc < T:
            pad = full_logits.new_zeros(B, T - T_trunc, 258)
            full_logits = torch.cat([full_logits, pad], dim=1)

        info = {
            "encoder": enc_info,
            "decoder": dec_info,
            "trigger_rate": trigger_prob.mean().item(),
            "n_triggered": n_triggered.float().mean().item(),
            "n_patches": J,
            "states": {"enc": enc_states},
        }

        return full_logits, info


# ── Smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T = 2, 64
    tokens = torch.randint(1, 256, (B, T))

    model = PatchLoopModel(dim=64, patch_size=8, enc_layers=2, dec_layers=2)

    logits, info = model(tokens)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    print(f"  encoder:  {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  decoder:  {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"  logits: {logits.shape}")
    print(f"  trigger_rate: {info['trigger_rate']:.3f}")
    print(f"  n_triggered: {info['n_triggered']:.1f}/{info['n_patches']}")

    # Gradient check
    loss = F.cross_entropy(logits.view(-1, 258), torch.randint(1, 256, (B * T,)))
    loss.backward()
    n_zero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0)
    print(f"Gradient: {n_zero}/{sum(1 for _ in model.parameters())} zero-grad")
    print("OK" if n_zero == 0 else "ISSUE")