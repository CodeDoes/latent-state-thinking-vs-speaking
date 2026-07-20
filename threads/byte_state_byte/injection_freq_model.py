"""Injection-frequency variants of AdaptiveLoopModel."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

from threads.adaptive_compute.adaptive_loop_model import (
    ByteEncoder, LoopedRWKV7Core, AdaptiveExitGate,
    pool_bytes_to_latents, dynamic_pool_bytes_to_latents,
    broadcast_variable_patches, ByteDecoder
)
from threads.unsorted.simple_rnn_receptance import SimpleRNNReceptance


class PerLayerFusionDecoder(nn.Module):
    def __init__(self, dim: int, n_layers: int = 2, max_loops: int = 3, use_gate: bool = True):
        super().__init__()
        self.dim = dim
        self.max_loops = max_loops
        self.use_gate = use_gate
        self.layers = nn.ModuleList([
            SimpleRNNReceptance(input_dim=dim, hidden_dim=dim) for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, 258)
        self.ln_out = nn.LayerNorm(dim)
        self.exit_gate = AdaptiveExitGate(dim) if max_loops > 1 else None
        if use_gate:
            self.fusion_gates = nn.ModuleList([
                nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid()) for _ in range(n_layers)
            ])
        else:
            self.fusion_gates = nn.ModuleList([None for _ in range(n_layers)])

    def forward(self, encoder_out: torch.Tensor, core_out_broadcast: torch.Tensor, states=None):
        B, T, D = encoder_out.shape
        x = encoder_out + core_out_broadcast
        if states is None:
            states = [None] * len(self.layers)
        layer_states = list(states)
        exit_lambdas = []
        best_logits = None
        depth_count = 1
        for r in range(self.max_loops):
            h = x
            new_states = []
            for i, (layer, ln) in enumerate(zip(self.layers, self.layer_norms)):
                if self.use_gate and self.fusion_gates[i] is not None:
                    gate_input = torch.cat([ln(h), core_out_broadcast], dim=-1)
                    gate = self.fusion_gates[i](gate_input)
                    fused = ln(h) + gate * core_out_broadcast
                else:
                    fused = ln(h) + core_out_broadcast
                out, h_final, _ = layer(fused, layer_states[i])
                h = h + out
                new_states.append(h_final)
            layer_states = new_states
            h = self.ln_out(h)
            logits = self.head(h)
            if best_logits is None:
                best_logits = logits
            if self.exit_gate is not None and r < self.max_loops - 1:
                lam = self.exit_gate(h, r)
                exit_lambdas.append(lam)
                if not self.training and lam.mean().item() > 0.5 and r > 0:
                    best_logits = logits
                    depth_count = r + 1
                    break
            if r < self.max_loops - 1:
                x = h.detach()
        return best_logits, layer_states, {"exit_lambdas": exit_lambdas, "depth_loop_count": depth_count}


class InjectionFreqAdaptiveModel(nn.Module):
    def __init__(self, dim=64, patch_size=4, enc_layers=2, core_layers=2, dec_layers=2,
                 enc_max_loops=3, core_depth_loops=2, dec_max_loops=3,
                 fusion_mode="front", dynamic_patch=False, patch_threshold=0.7, min_patch=2, max_patch=16,
                 core_hidden_scale=4, shared_core_state=True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.dynamic_patch = dynamic_patch
        self.patch_threshold = patch_threshold
        self.min_patch = min_patch
        self.max_patch = max_patch
        self.fusion_mode = fusion_mode
        self.encoder = ByteEncoder(dim, enc_layers, enc_max_loops)
        self.core = LoopedRWKV7Core(dim, core_layers, core_depth_loops, core_hidden_scale, shared_core_state)
        self.to_core = nn.Linear(dim, dim)
        if fusion_mode == "front":
            self.decoder = ByteDecoder(dim, dec_layers, dec_max_loops)
        elif fusion_mode == "per_layer":
            self.decoder = PerLayerFusionDecoder(dim, dec_layers, dec_max_loops, use_gate=True)
        elif fusion_mode == "per_layer_nogate":
            self.decoder = PerLayerFusionDecoder(dim, dec_layers, dec_max_loops, use_gate=False)
        else:
            raise ValueError(fusion_mode)

    def forward(self, tokens: torch.Tensor, enc_states=None, core_states=None, dec_states=None):
        B, T = tokens.shape
        encoder_out, enc_final_states, enc_info = self.encoder(tokens, enc_states)
        if self.dynamic_patch:
            latents, patch_lengths, patch_counts = dynamic_pool_bytes_to_latents(
                encoder_out, self.patch_threshold, self.min_patch, self.max_patch)
        else:
            latents = pool_bytes_to_latents(encoder_out, self.patch_size)
            patch_lengths = None
            patch_counts = None
            J = latents.shape[1]
        latents_proj = self.to_core(latents)
        core_out, core_final_states, core_info = self.core(latents_proj, core_states)

        if self.dynamic_patch:
            T_trunc = int(patch_lengths.sum(dim=1).max().item())
            enc_trunc = encoder_out[:, :T_trunc, :]
            core_broadcast = broadcast_variable_patches(core_out, patch_lengths, T_trunc)
        else:
            k = self.patch_size
            J = core_out.shape[1]
            T_trunc = J * k
            enc_trunc = encoder_out[:, :T_trunc, :]
            core_broadcast = core_out.repeat_interleave(k, dim=1)[:, :T_trunc, :]

        dec_out, dec_final_states, dec_info = self.decoder(enc_trunc, core_broadcast, dec_states)

        if dec_out.shape[1] < T:
            pad = dec_out.new_zeros(B, T - dec_out.shape[1], 258)
            logits = torch.cat([dec_out, pad], dim=1)
        else:
            logits = dec_out[:, :T, :]

        if self.dynamic_patch:
            n_latents = patch_counts.float().mean().item() if patch_counts is not None else core_out.shape[1]
        else:
            n_latents = core_out.shape[1]

        info = {
            "encoder": enc_info,
            "core": core_info,
            "decoder": dec_info,
            "n_latents": n_latents,
            "compression_ratio": T / max(n_latents, 1),
            "dynamic_patch": self.dynamic_patch,
            "states": {"enc": enc_final_states, "core": core_final_states, "dec": dec_final_states},
            "patch_lengths": patch_lengths,
            "patch_counts": patch_counts,
            "enc_loops": enc_info.get("loop_count", 1),
        }
        return logits, info

    def get_exit_stats(self, info: dict):
        return {
            "enc_loops": info["encoder"]["loop_count"],
            "core_depth_loops": info["core"]["depth_loop_count"],
            "dec_loops": info["decoder"]["depth_loop_count"],
            "n_latents": info["n_latents"],
            "compression_ratio": info["compression_ratio"],
        }

def count_params(m):
    return sum(p.numel() for p in m.parameters())
