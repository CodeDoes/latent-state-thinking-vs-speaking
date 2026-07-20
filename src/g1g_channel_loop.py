"""Loopy byte-level channel-mix replacement for the frozen g1g 2.9B RWKV-7.

Replaces selected RWKV-7 FFN (channel-mix) sublayers with a trainable
loopy byte encoder↔decoder pathway:

    input (dim hidden state)
      ├── time-mix: FROZEN (multi-head matrix state WKV)
      └── channel-mix: REPLACED
            ln2 → [project to bytes] → loopy encoder RNN
              → (byte|trigger) → loopy decoder RNN → [pool → project back]
            → residual add

The encoder projects the dim-dim hidden state to N byte-token IDs (258 vocab),
processes them through an RNN with adaptive triggering, and the decoder
converts the result back to the model dimension. This replaces a fixed FFN
with variable-compute byte-level reasoning.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from minGRU_pytorch import minGRU

# ── Constants ──────────────────────────────────────────────────────────────

BYTE_VOCAB_SIZE = 258
BYTE_PAD = 0
BYTE_TO_ID = {b: 2 + b for b in range(256)}

MODEL_DIR = Path.home() / "Documents" / "models" / "rwkv7-g1g-byte-iface"
DEFAULT_MODEL_PATH = MODEL_DIR / "model.pth"


# ── ByteEncoderRNN (parallel minGRU) ──────────────────────────────────────


class ByteEncoderRNN(nn.Module):
    """Loopy byte encoder: reads byte IDs, accumulates state, emits trigger.

    Uses minGRU for parallel processing of all N bytes in one scan.
    """

    def __init__(self, byte_dim: int, expansion_factor: float = 1.0):
        super().__init__()
        self.byte_dim = byte_dim
        self.embed = nn.Embedding(BYTE_VOCAB_SIZE, byte_dim, padding_idx=BYTE_PAD)
        self.gru = minGRU(byte_dim, expansion_factor=expansion_factor)
        self.byte_head = nn.Linear(byte_dim, BYTE_VOCAB_SIZE, bias=False)
        self.trigger_head = nn.Linear(byte_dim, 1)

    def forward(
        self,
        byte_ids: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        soft_logits: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = byte_ids.shape

        if soft_logits is not None:
            x = soft_logits @ self.embed.weight  # (B, T, byte_dim)
        else:
            x = self.embed(byte_ids)

        h = self.gru(x, prev_hidden)  # (B, T, byte_dim)

        byte_logits = self.byte_head(h)  # (B, T, 258)
        trigger_logits = self.trigger_head(h).squeeze(-1)  # (B, T)
        avg_trigger = torch.sigmoid(trigger_logits).mean()

        # Return final hidden for state carry
        final_hidden = h[:, -1:, :]  # (B, 1, byte_dim)

        return byte_logits, trigger_logits, final_hidden, avg_trigger


class ByteDecoderRNN(nn.Module):
    """Loopy byte decoder: reads encoder trigger signal, produces bytes.

    Uses minGRU for parallel processing.
    """

    def __init__(self, byte_dim: int, expansion_factor: float = 1.0):
        super().__init__()
        self.byte_dim = byte_dim
        self.input_proj = nn.Linear(BYTE_VOCAB_SIZE + 1, byte_dim, bias=False)
        self.gru = minGRU(byte_dim, expansion_factor=expansion_factor)
        self.byte_head = nn.Linear(byte_dim, BYTE_VOCAB_SIZE, bias=False)
        self.trigger_head = nn.Linear(byte_dim, 1)

    def forward(
        self,
        enc_byte_logits: torch.Tensor,
        enc_trigger_logits: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trigger_3d = torch.sigmoid(enc_trigger_logits).unsqueeze(-1)
        dec_in = torch.cat([enc_byte_logits, trigger_3d], dim=-1)
        x = self.input_proj(dec_in)  # (B, T, byte_dim)

        h = self.gru(x, prev_hidden)  # (B, T, byte_dim)

        byte_logits = self.byte_head(h)  # (B, T, 258)
        trigger_logits = self.trigger_head(h).squeeze(-1)  # (B, T)
        avg_trigger = torch.sigmoid(trigger_logits).mean()

        final_hidden = h[:, -1:, :]

        return byte_logits, trigger_logits, final_hidden, avg_trigger

# ── BlackGooseChannelMix: single-linear-layer FFN replacement ────────────
# Based on BlackGoose_Rimer (https://github.com/Alic-Li/BlackGoose_Rimer)
# Replaces the standard RWKV-7 channel-mix (time-mix + ReLU² + key/value)
# with a single linear layer: x → nn.Linear(dim, dim)(x)
# No time-mix, no activation, no gating — just a learned projection.


class BlackGooseChannelMix(nn.Module):
    """Simplest possible RWKV-7 channel-mix replacement.

    Takes the layer-normed input and passes it through a single linear
    layer. No time-mix, no squared ReLU, no receptance gating — just a
    learned dim→dim projection, exactly as in BlackGoose_Rimer's CMix.

    Compared to LoopyChannelMix (byte encoder↔decoder RNN), this has:
    - Far simpler forward pass (one matmul)
    - Fewer hyperparameters (just dim)
    - dim² parameters per layer (vs n_bytes×byte_dim + byte_dim² for loopy)
    - A single linear projection with no recurrent state to carry
    """

    def __init__(self, dim: int):
        super().__init__()
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        loop_state: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, D) normalized hidden state
            loop_state: ignored (no recurrent state needed)
        Returns:
            out: (B, D) output (residual is added by caller)
            info: empty dict (no triggers to report)
        """
        return self.value(x), {}


# ── LoopyChannelMix: the actual FFN replacement ─────────────────────────


class LoopyChannelMix(nn.Module):
    """Replaces one RWKV-7 FFN with a loopy byte encoder↔decoder.

    ln2(x) → project to N bytes → encoder RNN → (byte|trigger)
      → decoder RNN → pool → project back to dim → output
    """

    def __init__(
        self,
        dim: int,
        n_bytes: int = 32,
        byte_dim: int = 64,
        expansion_factor: float = 1.0,
        min_encoder_steps: int = 4,
        max_loops: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.n_bytes = n_bytes
        self.byte_dim = byte_dim
        self.max_loops = max_loops
        self.min_encoder_steps = min_encoder_steps

        self.hidden_to_byte_logits = nn.Linear(dim, n_bytes * BYTE_VOCAB_SIZE, bias=False)
        self.encoder = ByteEncoderRNN(byte_dim, expansion_factor=expansion_factor)
        self.decoder = ByteDecoderRNN(byte_dim, expansion_factor=expansion_factor)
        self.byte_to_hidden = nn.Linear(byte_dim, dim, bias=False)

        self.register_buffer("pos_bias", torch.zeros(n_bytes))
        with torch.no_grad():
            self.pos_bias[:min_encoder_steps] = -5.0

    def _hidden_to_bytes(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project (B, dim) → (B, n_bytes) byte IDs + (B, n_bytes, 258) logits."""
        B = h.shape[0]
        logits = self.hidden_to_byte_logits(h).view(B, self.n_bytes, BYTE_VOCAB_SIZE)
        if self.training:
            # Gumbel-softmax with straight-through: use soft samples as
            # embeddings via the straight-through one-hot
            probs = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
            byte_ids = probs.argmax(dim=-1)
            # Return soft probs so encoder can use them differentiably
            return byte_ids, probs
        else:
            byte_ids = logits.argmax(dim=-1)
            return byte_ids, logits

    def _pool_decode(self, dec_hidden: torch.Tensor) -> torch.Tensor:
        """Pool decoder hidden state → project to dim."""
        # dec_hidden: (B, 1, byte_dim) — final token's hidden
        return self.byte_to_hidden(dec_hidden.squeeze(1))  # (B, dim)

    def forward(
        self,
        x: torch.Tensor,
        loop_state: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        B = x.shape[0]
        byte_ids, byte_probs = self._hidden_to_bytes(x)

        # Pass soft probs to encoder during training for differentiability
        soft_logits = byte_probs if self.training else None

        enc_hidden = loop_state.get("enc_hidden") if loop_state else None
        enc_byte_logits, enc_trigger_logits, enc_hidden, enc_tr = \
            self.encoder(byte_ids, enc_hidden, soft_logits=soft_logits)
        seq_len = min(self.n_bytes, self.pos_bias.shape[0])
        enc_trigger_logits[:, :seq_len] += self.pos_bias[:seq_len]

        dec_hidden = loop_state.get("dec_hidden") if loop_state else None
        dec_byte_logits, dec_trigger_logits, dec_hidden, dec_tr = \
            self.decoder(enc_byte_logits, enc_trigger_logits.detach(), dec_hidden)

        out = self._pool_decode(dec_hidden)
        info = {
            "enc_trigger_rate": enc_tr.item() if torch.is_tensor(enc_tr) else enc_tr,
            "dec_trigger_rate": dec_tr.item() if torch.is_tensor(dec_tr) else dec_tr,
            "enc_hidden": enc_hidden,
            "dec_hidden": dec_hidden,
        }
        return out, info


# ── Frozen g1g wrapper with loopy channel mixes ──────────────────────────


class G1GWithLoopyChannel(nn.Module):
    """Frozen g1g 2.9B RWKV-7 with trainable channel-mix replacements.

    Loads frozen weights from the byte-interface checkpoint and replaces
    the FFN sublayer of selected layers with a trainable module.

    Supported channel types:
    - 'loopy': LoopyChannelMix — byte encoder↔decoder RNN pathway
    - 'blackgoose': BlackGooseChannelMix — single linear layer
      (based on BlackGoose_Rimer: https://github.com/Alic-Li/BlackGoose_Rimer)

    Only the replacement modules' parameters are trainable. All original
    parameters (time-mix, layer norms, embed, head) are frozen.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        layers_to_replace: Optional[list[int]] = None,
        channel_type: str = 'loopy',
        n_bytes: int = 32,
        byte_dim: int = 64,
        expansion_factor: float = 1.0,
        min_encoder_steps: int = 4,
        max_loops: int = 2,
        verbose: bool = True,
    ):
        assert channel_type in ('loopy', 'blackgoose'), \
            f"channel_type must be 'loopy' or 'blackgoose', got '{channel_type}'"
        self.channel_type = channel_type
        super().__init__()

        t0 = time.time()

        # Load frozen state dict
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        self._sd = sd

        D = sd["ln_out.weight"].shape[0]
        head_size = sd["blocks.0.att.r_k"].shape[-1]
        n_heads = D // head_size
        n_layers = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".ln1.weight"))

        self.dim = D
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_size = head_size
        self.device = None
        self.dtype = torch.bfloat16

        if verbose:
            print(f"G1GWithLoopyChannel: {n_layers} layers, {D} dim, {n_heads}×{head_size} heads")

        # Trainable loopy channels
        if layers_to_replace is None:
            layers_to_replace = list(range(min(6, n_layers)))
        self.layers_to_replace = set(layers_to_replace)

        self.loopy_channels = nn.ModuleDict()
        for lid in layers_to_replace:
            if channel_type == 'loopy':
                self.loopy_channels[str(lid)] = LoopyChannelMix(
                    dim=D, n_bytes=n_bytes, byte_dim=byte_dim,
                    expansion_factor=expansion_factor,
                    min_encoder_steps=min_encoder_steps, max_loops=max_loops,
                )
            else:  # 'blackgoose'
                self.loopy_channels[str(lid)] = BlackGooseChannelMix(dim=D)

        # Frozen output params
        self.ln_out_weight = nn.Parameter(sd["ln_out.weight"].clone(), requires_grad=False)
        self.ln_out_bias = nn.Parameter(sd["ln_out.bias"].clone(), requires_grad=False)
        self.byte_head_weight = nn.Parameter(sd["byte_head.weight"].clone(), requires_grad=False)
        self.byte_embed_weight = nn.Parameter(sd["byte_embed.weight"].clone(), requires_grad=False)

        trainable = sum(p.numel() for p in self.loopy_channels.parameters())
        if verbose:
            print(f"  Trainable {channel_type} channels: {trainable:,} params ({trainable/1e6:.2f}M)")

    def to_device(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        """Move frozen weights to device/dtype."""
        self.device = device
        self.dtype = dtype
        for k, v in self._sd.items():
            if isinstance(v, torch.Tensor):
                # Keep mat state as float32, everything else as dtype
                if k.endswith('.mat'):
                    self._sd[k] = v.to(device=device, dtype=torch.float)
                else:
                    self._sd[k] = v.to(device=device, dtype=dtype)
        self.ln_out_weight.data = self.ln_out_weight.data.to(device=device, dtype=dtype)
        self.ln_out_bias.data = self.ln_out_bias.data.to(device=device, dtype=dtype)
        self.byte_head_weight.data = self.byte_head_weight.data.to(device=device, dtype=dtype)
        self.byte_embed_weight.data = self.byte_embed_weight.data.to(device=device, dtype=dtype)
        self.loopy_channels.to(device=device, dtype=dtype)

    def _time_mix(self, layer_id: int, ln1: torch.Tensor, state: dict) -> torch.Tensor:
        """Frozen time-mixing for one layer.  Core RWKV-7 WKV computation."""
        sd = self._sd
        D, H, N = self.dim, self.n_heads, self.head_size
        att = f"blocks.{layer_id}.att."

        # Squeeze 3D shift params (1,1,D) to (D,) for 2D inputs
        x_r = sd[att + "x_r"].squeeze()
        x_w = sd[att + "x_w"].squeeze()
        x_k = sd[att + "x_k"].squeeze()
        x_v = sd[att + "x_v"].squeeze()
        x_a = sd[att + "x_a"].squeeze()
        x_g = sd[att + "x_g"].squeeze()

        xx = state["xx"] - ln1
        xr = ln1 + xx * x_r
        xw = ln1 + xx * x_w
        xk = ln1 + xx * x_k
        xv = ln1 + xx * x_v
        xa = ln1 + xx * x_a
        xg = ln1 + xx * x_g

        r = xr @ sd[att + "receptance.weight"]
        w = torch.tanh(xw @ sd[att + "w1"]) @ sd[att + "w2"]
        k = xk @ sd[att + "key.weight"]
        v = xv @ sd[att + "value.weight"]
        # Squeeze (1,1,D) params to (D,)
        a0 = sd[att + "a0"].squeeze()
        a = torch.sigmoid(a0 + (xa @ sd[att + "a1"]) @ sd[att + "a2"])
        g = torch.sigmoid(xg @ sd[att + "g1"]) @ sd[att + "g2"]

        r_h, k_h, v_h, a_h, g_h, w_h = \
            (t.view(-1, H, N) for t in [r, k, v, a, g, w])

        kk = F.normalize(k_h * sd[att + "k_k"].view(1, H, N), dim=-1, p=2.0)
        k_adj = k_h * (1 + (a_h - 1) * sd[att + "k_a"].view(1, H, N))

        v_first = state.get("v_first")
        if v_first is None or v_first.numel() != H * N:
            state["v_first"] = v_h.clone()
        else:
            v0 = sd[att + "v0"].squeeze()
            v0_h = v0.view(H, N) if v0.dim() == 2 else v0.view(1, H, N)
            blend_raw = (xv @ sd[att + "v1"]) @ sd[att + "v2"]
            blend = torch.sigmoid(v0_h + blend_raw.view(-1, H, N))
            v_h = v_h + (state["v_first"] - v_h) * blend

        w0 = sd[att + "w0"].squeeze()
        w0_h = w0.view(H, N) if w0.dim() == 2 else w0.view(1, H, N)
        w_decay = torch.exp(-0.606531 * torch.sigmoid((w0_h + w_h).float()))

        mat = state["mat"]
        vk = v_h.unsqueeze(-1) @ k_adj.unsqueeze(-2)
        ab = (-kk).unsqueeze(-1) @ (kk * a_h).unsqueeze(-2)
        mat = (mat * w_decay.unsqueeze(-2).float()
               + (mat @ ab.float())
               + vk.float())

        out_h = (mat.to(dtype=ln1.dtype) @ r_h.unsqueeze(-1)).squeeze(-1)

        out_flat = out_h.reshape(-1, D)
        out_flat = F.group_norm(
            out_flat, num_groups=H,
            weight=sd[att + "ln_x.weight"],
            bias=sd[att + "ln_x.bias"], eps=64e-5,
        ).view(-1, D)

        shortcut = (r_h * k_h * sd[att + "r_k"].unsqueeze(0)).sum(dim=-1, keepdim=True) * v_h
        out_flat = out_flat + shortcut.reshape(-1, D)
        out_flat = out_flat * g
        tm_out = out_flat @ sd[att + "output.weight"]
        state["mat"] = mat.detach()
        return tm_out

    def _forward_token(self, h: torch.Tensor, states: list[dict]) -> torch.Tensor:
        """Process one token position through all layers."""
        for i in range(self.n_layers):
            s = states[i]

            ln1 = F.layer_norm(
                h, (self.dim,),
                weight=self._sd[f"blocks.{i}.ln1.weight"],
                bias=self._sd[f"blocks.{i}.ln1.bias"],
            )
            tm_out = self._time_mix(i, ln1, s)
            h = h + tm_out

            ln2 = F.layer_norm(
                h, (self.dim,),
                weight=self._sd[f"blocks.{i}.ln2.weight"],
                bias=self._sd[f"blocks.{i}.ln2.bias"],
            )

            if str(i) in self.loopy_channels:
                cm_out, _ = self.loopy_channels[str(i)](ln2)
            else:
                att = f"blocks.{i}.ffn."
                xx_c = s.get("xx_c", torch.zeros_like(h))
                x_diff = xx_c - ln2
                xk_shift = self._sd[att + "x_k"].squeeze()
                xk = ln2 + x_diff * xk_shift
                k_c = F.relu(xk @ self._sd[att + "key.weight"].T) ** 2
                cm_out = k_c @ self._sd[att + "value.weight"].T

            h = h + cm_out
            s["xx"] = ln1.detach().clone()
            s["xx_c"] = ln2.detach().clone()

        return h

    def forward(
        self,
        byte_ids: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        B, T = byte_ids.shape
        if self.device is None or self.device != byte_ids.device:
            self.to_device(byte_ids.device, self.dtype)

        byte_ids = byte_ids.to(self.device)
        outputs = []

        for t in range(T):
            byte_t = byte_ids[:, t]
            if (byte_t == BYTE_PAD).all():
                break

            h = F.embedding(byte_t, self.byte_embed_weight)

            if t == 0:
                states = [
                    {
                        "xx": torch.zeros(B, self.dim, device=self.device, dtype=self.dtype),
                        "xx_c": torch.zeros(B, self.dim, device=self.device, dtype=self.dtype),
                        "mat": torch.zeros(B, self.n_heads, self.head_size, self.head_size,
                                           device=self.device, dtype=torch.float),
                        "v_first": None,
                    }
                    for _ in range(self.n_layers)
                ]

            h = self._forward_token(h, states)
            outputs.append(h)

        if not outputs:
            return torch.zeros(B, 0, BYTE_VOCAB_SIZE, device=self.device)

        h_out = torch.stack(outputs, dim=1)

        if not return_logits:
            return h_out

        h_out = F.layer_norm(h_out, (self.dim,),
                             weight=self.ln_out_weight, bias=self.ln_out_bias)
        logits = h_out @ self.byte_head_weight.T
        return logits

    def get_trainable_params(self):
        return list(self.loopy_channels.parameters())


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== G1G Channel Loop — smoke test ===")

    # Test with a tiny synthetic model instead of the real g1g
    D, H, N = 128, 4, 32
    n_layers = 3

    # Create a fake frozen state dict matching the g1g structure
    sd = {}
    sd["byte_embed.weight"] = torch.randn(258, D)
    sd["byte_head.weight"] = torch.randn(258, D)
    sd["ln_out.weight"] = torch.randn(D)
    sd["ln_out.bias"] = torch.randn(D)
    for i in range(n_layers):
        for name in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"]:
            sd[f"blocks.{i}.{name}"] = torch.randn(D)
        for pfx in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]:
            sd[f"blocks.{i}.att.{pfx}"] = torch.randn(1, 1, D)
        for w in ["receptance.weight", "key.weight", "value.weight",
                   "w1", "w2", "a1", "a2", "v1", "v2", "output.weight",
                   "g1", "g2"]:
            sd[f"blocks.{i}.att.{w}"] = torch.randn(D, D) if "g2" not in w and "w2" not in w and "a2" not in w and "v2" not in w else torch.randn(D, D)
        sd[f"blocks.{i}.att.a0"] = torch.randn(1, 1, D)
        sd[f"blocks.{i}.att.v0"] = torch.randn(1, 1, D)
        sd[f"blocks.{i}.att.w0"] = torch.randn(1, 1, D)
        sd[f"blocks.{i}.att.k_k"] = torch.randn(1, 1, D)
        sd[f"blocks.{i}.att.k_a"] = torch.randn(1, 1, D)
        sd[f"blocks.{i}.att.r_k"] = torch.randn(H, N)
        sd[f"blocks.{i}.att.ln_x.weight"] = torch.randn(D)
        sd[f"blocks.{i}.att.ln_x.bias"] = torch.randn(D)
        sd[f"blocks.{i}.ffn.key.weight"] = torch.randn(D * 4, D)
        sd[f"blocks.{i}.ffn.value.weight"] = torch.randn(D, D * 4)
        sd[f"blocks.{i}.ffn.x_k"] = torch.randn(1, 1, D)

    if DEFAULT_MODEL_PATH.exists():
        print("Real g1g model found. Quick forward test...")
        model = G1GWithLoopyChannel(
            model_path=DEFAULT_MODEL_PATH,
            layers_to_replace=[0],
            n_bytes=8,
            byte_dim=16,
        )
        byte_ids = torch.randint(2, 258, (1, 2))
        logits = model(byte_ids)
        assert logits.shape == (1, 2, 258), f"Expected (1, 2, 258), got {logits.shape}"
        model.train()
        logits = model(byte_ids)
        loss = logits.sum()
        loss.backward()
        n_grad = sum(1 for p in model.loopy_channels.parameters()
                    if p.grad is not None and p.grad.abs().sum() > 0)
        n_total = sum(1 for p in model.loopy_channels.parameters())
        hidden_grad = model.loopy_channels['0'].hidden_to_byte_logits.weight.grad
        assert hidden_grad is not None, "hidden_to_byte_logits must receive gradient"
        print(f"  Output: {logits.shape}, Gradients: {n_grad}/{n_total}, "
              f"h2b_grad: {hidden_grad.norm().item():.2f}")
        print("OK")
    else:
        print(f"No g1g weights at {DEFAULT_MODEL_PATH}")
        print("Verifying module construction...")
        cm = LoopyChannelMix(dim=128, n_bytes=16, byte_dim=32)
        x = torch.randn(2, 128)
        out, info = cm(x)
        print(f"  LoopyChannelMix: {x.shape} → {out.shape}")
        print(f"  Enc trigger: {info['enc_trigger_rate']:.3f}")
        print(f"  Dec trigger: {info['dec_trigger_rate']:.3f}")
        n = sum(p.numel() for p in cm.parameters())
        print(f"  Params: {n:,}")
        print("OK")
