"""g1g 2.9B with BlackGoose linear channel-mix.

Frozen backbone loaded via NF4 cache (~1.5GB on GPU).  Weights stay
quantized and are dequantized per-layer during forward, keeping peak
memory under 4GB even on small GPUs.

Only the BlackGooseChannelMix linear layers are trainable.  All backbone
weights are frozen.

Usage:
    from src.g1g_blackgoose_nf4 import G1GBlackGooseNF4
    model = G1GBlackGooseNF4(layers_to_replace=[0, 1], device='cuda')
    logits = model(byte_ids)  # forward, gradients through BG cmix only
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ── Constants ──────────────────────────────────────────────────────────────

BYTE_VOCAB_SIZE = 258
BYTE_PAD = 0
BYTE_TO_ID = {b: 2 + b for b in range(256)}

MODEL_DIR = Path.home() / "Documents" / "models" / "rwkv7-g1g-byte-iface"
DEFAULT_MODEL_PATH = MODEL_DIR / "model.pth"
NF4_CACHE_DIR = MODEL_DIR / "nf4_cache"
NF4_INDEX_PATH = NF4_CACHE_DIR / "index.json"


# ── BlackGoose channel-mix ────────────────────────────────────────────────


class BlackGooseChannelMix(nn.Module):
    """Single Linear(dim, dim) channel-mix replacement.

    Based on BlackGoose_Rimer's CMix.
    """

    def __init__(self, dim: int, dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = torch.device("cuda")):
        super().__init__()
        self.value = nn.Linear(dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, loop_state=None):
        return self.value(x), {}


# ── NF4 state dict with per-layer dequantization ──────────────────────────


class NF4StateDict:
    """Holds NF4 quantized weights + small bf16 tensors.

    Quantized weights stay as (q, QuantState) tuples on GPU (0.5 bytes/param).
    Small tensors are stored in bf16 directly.
    During forward, call dequantize_layer(lid) to get a temporary dict of
    dequantized bf16 tensors for that layer's time-mix + channel-mix.
    """

    def __init__(self, model_path: Path, device: torch.device,
                 dtype: torch.dtype):
        from bitsandbytes.functional import QuantState

        t0 = time.time()
        sd = torch.load(model_path, map_location="cpu", weights_only=True)

        with open(NF4_INDEX_PATH) as f:
            index: dict = json.load(f)

        # Load NF4 shards (keep quantized)
        self._index = index
        self._quant: dict[str, tuple] = {}
        self._bf16: dict[str, torch.Tensor] = {}
        self._device = device
        self._dtype = dtype
        self._layer_weights: dict[int, list[str]] = {}
        self._layer_nf4_count = 0

        # Group NF4 entries by layer
        for key in index:
            parts = key.split(".")
            lid = int(parts[1])
            self._layer_weights.setdefault(lid, []).append(key)

        for key, cache_path in index.items():
            data = torch.load(cache_path, map_location=device, weights_only=True)
            qs = QuantState(
                absmax=data["absmax"], shape=data["shape"], code=data["code"],
                blocksize=data["blocksize"], dtype=data["dtype"],
                quant_type=data["quant_type"],
            )
            self._quant[key] = (data["q"], qs)
            del data

        # Move small (non-NF4) tensors to GPU as bf16
        for key, v in sd.items():
            if key in index:
                continue  # handled by NF4
            if isinstance(v, torch.Tensor):
                self._bf16[key] = v.to(device=device, dtype=dtype)
            else:
                self._bf16[key] = v

        del sd
        torch.cuda.empty_cache() if device.type == "cuda" else None

        nf4_mem = sum(v[0].numel() * v[0].element_size() / 2
                      for v in self._quant.values())
        bf16_mem = sum(v.numel() * v.element_size()
                       for v in self._bf16.values())
        total = nf4_mem + bf16_mem
        gpu = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(f"NF4 state dict: {total/1e9:.2f}GB "
              f"({bf16_mem/1e9:.2f}GB bf16 + {nf4_mem/1e9:.2f}GB NF4) "
              f"on GPU in {time.time()-t0:.1f}s [GPU={gpu:.2f}GB]")

    def dequantize_layer(self, lid: int) -> dict[str, torch.Tensor]:
        """Dequantize layer `lid` into fresh bf16 tensors.

        During eval these are temporary (discarded after use).
        During train, each layer creates distinct tensors so autograd
        can correctly backprop through all 32 layers (no buffer overwrite).
        """
        from bitsandbytes.functional import dequantize_4bit
        out = {}
        for key in self._layer_weights.get(lid, []):
            q, qs = self._quant[key]
            out[key] = dequantize_4bit(q, qs).to(dtype=self._dtype)
        return out

    def get_bf16(self, key: str) -> torch.Tensor:
        return self._bf16[key]

    def has(self, key: str) -> bool:
        return key in self._bf16 or key in self._quant


# ── G1GBlackGooseNF4 ──────────────────────────────────────────────────────


class G1GBlackGooseNF4(nn.Module):
    """Frozen g1g 2.9B (NF4 quantized) with trainable BlackGoose channel-mix.

    Args:
        layers_to_replace: layer indices to replace FFN with BlackGoose linear.
        device: 'cuda' or 'cpu'.
        dtype: computation dtype (bfloat16 recommended).
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        layers_to_replace: Optional[list[int]] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.dtype = dtype

        t0 = time.time()

        # Load NF4 backbone
        self._sd = NF4StateDict(model_path, self.device, dtype)

        # Infer dimensions
        D = self._sd.get_bf16("ln_out.weight").shape[0]
        head_size = self._sd.get_bf16("blocks.0.att.r_k").shape[-1]
        n_heads = D // head_size
        n_layers = sum(1 for k in self._sd._bf16
                       if k.startswith("blocks.") and k.endswith(".ln1.weight"))

        self.dim = D
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_size = head_size

        if verbose:
            print(f"G1GBlackGooseNF4: {n_layers} layers, {D} dim, "
                  f"{n_heads}×{head_size} heads")

        # Trainable BlackGoose layers
        if layers_to_replace is None:
            layers_to_replace = list(range(min(4, n_layers)))
        self.layers_to_replace = set(layers_to_replace)

        self.trainable_channels = nn.ModuleDict()
        for lid in layers_to_replace:
            self.trainable_channels[str(lid)] = BlackGooseChannelMix(
                dim=D, dtype=dtype, device=self.device,
            )

        # Extract output buffers and drop from _sd to save memory
        self.register_buffer("_ln_out_w", self._sd._bf16.pop("ln_out.weight").detach().clone())
        self.register_buffer("_ln_out_b", self._sd._bf16.pop("ln_out.bias").detach().clone())
        self.register_buffer("_head_w", self._sd._bf16.pop("byte_head.weight").detach().clone())
        self.register_buffer("_embed_w", self._sd._bf16.pop("byte_embed.weight").detach().clone())
        self._sd._bf16.pop("emb.weight", None)  # full 65K vocab — unused in byte mode

        trainable = sum(p.numel() for p in self.trainable_channels.parameters())
        if verbose:
            print(f"  Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
            if self.device.type == "cuda":
                print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── Time-mix (frozen, uses per-layer dequantized weights) ──────────

    def _time_mix(self, lid: int, ln1: torch.Tensor, s: dict,
                  w: dict[str, torch.Tensor]) -> torch.Tensor:
        """RWKV-7 time-mixing for layer lid.

        w = dequantized NF4 weights for this layer (att receptance/key/value/output,
        ffn key/value).  Other projection weights (w1, w2, a1, a2, v1, v2, g1, g2)
        are NOT quantized and accessed directly from _sd._bf16.
        """
        bf16 = self._sd._bf16
        D, H, N = self.dim, self.n_heads, self.head_size
        att = f"blocks.{lid}.att."

        xx = s["xx"] - ln1
        xr = ln1 + xx * bf16[att + "x_r"].squeeze()
        xw = ln1 + xx * bf16[att + "x_w"].squeeze()
        xk = ln1 + xx * bf16[att + "x_k"].squeeze()
        xv = ln1 + xx * bf16[att + "x_v"].squeeze()
        xa = ln1 + xx * bf16[att + "x_a"].squeeze()
        xg = ln1 + xx * bf16[att + "x_g"].squeeze()

        # NF4 weights (in `w`)
        r = xr @ w[att + "receptance.weight"]
        k = xk @ w[att + "key.weight"]
        v = xv @ w[att + "value.weight"]

        # Non-NF4 weights (stored directly as bf16)
        w_t = torch.tanh(xw @ bf16[att + "w1"]) @ bf16[att + "w2"]
        a_t = torch.sigmoid(
            bf16[att + "a0"].squeeze()
            + (xa @ bf16[att + "a1"]) @ bf16[att + "a2"])
        g_t = torch.sigmoid(xg @ bf16[att + "g1"]) @ bf16[att + "g2"]

        def th(t): return t.view(H, N)
        def fh(t): return t.reshape(D)
        r_h, k_h, v_h, a_h, w_h = map(th, [r, k, v, a_t, w_t])
        # g stays flat (D,) for the elementwise gate
        g = g_t

        kk = F.normalize(k_h * th(bf16[att + "k_k"].squeeze()), dim=-1, p=2.0)
        k_adj = k_h * (1 + (a_h - 1) * th(bf16[att + "k_a"].squeeze()))

        if s.get("v_first") is None:
            s["v_first"] = v_h.clone()
        else:
            blend = torch.sigmoid(
                th(bf16[att + "v0"].squeeze())
                + (xv @ bf16[att + "v1"] @ bf16[att + "v2"]).view(H, N))
            v_h = v_h + (s["v_first"] - v_h) * blend

        w0 = bf16[att + "w0"].squeeze()
        wd = torch.exp(-0.606531 * torch.sigmoid((th(w0) + w_h).float()))

        mat = s["mat"]
        vk = v_h.unsqueeze(-1) @ k_adj.unsqueeze(-2)
        ab = (-kk).unsqueeze(-1) @ (kk * a_h).unsqueeze(-2)
        mat = (mat * wd.unsqueeze(-2).float()
               + (mat @ ab.float()) + vk.float())

        out_h = (mat.to(dtype=ln1.dtype) @ r_h.unsqueeze(-1)).squeeze(-1)
        out = fh(out_h)
        out = F.group_norm(out.view(1, D), H,
                           weight=bf16[att + "ln_x.weight"],
                           bias=bf16[att + "ln_x.bias"], eps=64e-5).view(D)
        out = out + fh((r_h * k_h * bf16[att + "r_k"]).sum(-1, keepdim=True) * v_h)
        out = out * g
        tm = out @ w[att + "output.weight"]
        s["mat"] = mat.detach().clone()
        return tm

    # ── Single layer block (checkpointed) ─────────────────────────────

    @staticmethod
    def _pack_state(xx, xx_c, mat, v_first_valid, v_first_h, v_first_n):
        """Pack state dict into a flat tensor tuple for checkpoint."""
        return (xx, xx_c, mat, v_first_valid,
                v_first_h if v_first_valid else xx.new_zeros(0),
                torch.tensor(v_first_n, device=xx.device))

    @staticmethod
    def _unpack_state(packed):
        """Unpack checkpoint-compatible tuple back to dict."""
        xx, xx_c, mat, vf_valid = packed[:4]
        if vf_valid:
            return {"xx": xx, "xx_c": xx_c, "mat": mat,
                    "v_first": packed[4]}
        return {"xx": xx, "xx_c": xx_c, "mat": mat, "v_first": None}

    def _layer_block(self, lid: torch.Tensor, h: torch.Tensor,
                     xx: torch.Tensor, xx_c: torch.Tensor,
                     mat: torch.Tensor,
                     vf_state: torch.Tensor) -> tuple:
        """Forward for one frozen layer (all tensor args, for checkpoint).
        Dequantizes weights inside so autograd doesn't accumulate
        references across layers.
        vf_state: [v_first.reshape(-1), valid_flag] where valid_flag is 0 or 1.
        """
        i = int(lid.item())
        D = self.dim
        H, N = self.n_heads, self.head_size
        bf16 = self._sd._bf16

        v_first = vf_state[:-1].reshape(H, N) if vf_state[-1] > 0.5 else None
        s = {"xx": xx, "xx_c": xx_c, "mat": mat, "v_first": v_first}

        ln1 = F.layer_norm(h, (D,),
                           weight=bf16[f"blocks.{i}.ln1.weight"],
                           bias=bf16[f"blocks.{i}.ln1.bias"])

        w = self._sd.dequantize_layer(i)
        h = h + self._time_mix(i, ln1, s, w)

        ln2 = F.layer_norm(h, (D,),
                           weight=bf16[f"blocks.{i}.ln2.weight"],
                           bias=bf16[f"blocks.{i}.ln2.bias"])

        if str(i) in self.trainable_channels:
            cm_out, _ = self.trainable_channels[str(i)](ln2)
        else:
            ffn = f"blocks.{i}.ffn."
            xx_c_s = s.get("xx_c", torch.zeros_like(h))
            xk = ln2 + (xx_c_s - ln2) * bf16[ffn + "x_k"].squeeze()
            k_c = F.relu(xk @ w[ffn + "key.weight"].T) ** 2
            cm_out = k_c @ w[ffn + "value.weight"].T

        h = h + cm_out
        del w

        # Pack output state for checkpoint compatibility
        new_vf = s.get("v_first")
        if new_vf is not None:
            new_vf_state = torch.cat([new_vf.reshape(-1), h.new_ones(1)])
        else:
            new_vf_state = torch.cat([h.new_zeros(H * N), h.new_zeros(1)])

        return (h,
                ln1.detach().clone(),  # new_xx
                ln2.detach().clone(),  # new_xx_c
                s["mat"].detach().clone(),
                new_vf_state)

    # ── Eval buffer management (pre-allocated to avoid allocator fragmentation) ─

    def _ensure_eval_bufs(self):
        """Pre-allocate one set of shared dequant buffers for eval.
        Buffers are keyed by weight type (e.g., "att.receptance.weight")
        and are reused across all layers."""
        if hasattr(self, "_eval_bufs"):
            return
        # Get shapes from QuantState for layer 0 weights
        layer0_keys = self._sd._layer_weights.get(0, [])
        bufs = {}
        for key in layer0_keys:
            type_key = key.split(".", 2)[2]  # e.g., "att.receptance.weight"
            _, qs = self._sd._quant[key]
            bufs[type_key] = torch.empty(qs.shape, device=self.device,
                                         dtype=self.dtype)
        self._eval_bufs = bufs
        # Populate with layer 0 weights
        self._dequantize_to_bufs(0)

    def _dequantize_to_bufs(self, lid: int):
        """Dequantize layer `lid` into shared eval buffers."""
        from bitsandbytes.functional import dequantize_4bit
        for key in self._sd._layer_weights.get(lid, []):
            type_key = key.split(".", 2)[2]
            q, qs = self._sd._quant[key]
            deq = dequantize_4bit(q, qs)
            self._eval_bufs[type_key].copy_(deq)

    # ── Single token forward ──────────────────────────────────────────

    def _forward_token(self, h: torch.Tensor, states: list) -> torch.Tensor:
        training = self.training
        Z = self.n_heads * self.head_size
        for i in range(self.n_layers):
            if training:
                # Use checkpoint during training to avoid autograd
                # accumulating dequantized weight references across layers
                s = states[i]
                xx = s.get("xx", torch.zeros_like(h))
                xx_c = s.get("xx_c", torch.zeros_like(h))
                mat = s.get("mat", torch.zeros(self.n_heads, self.head_size,
                                                 self.head_size, device=h.device,
                                                 dtype=torch.float32))
                vf = s.get("v_first")
                vf_state = torch.cat([
                    vf.reshape(-1) if vf is not None else h.new_zeros(Z),
                    h.new_ones(1) if vf is not None else h.new_zeros(1)
                ])

                lid_t = torch.tensor(i, device=h.device)

                h, new_xx, new_xx_c, new_mat, new_vf_state = checkpoint(
                    self._layer_block, lid_t, h, xx, xx_c, mat, vf_state,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

                states[i]["xx"] = new_xx
                states[i]["xx_c"] = new_xx_c
                states[i]["mat"] = new_mat
                if new_vf_state[-1] > 0.5:
                    states[i]["v_first"] = new_vf_state[:-1].reshape(
                        self.n_heads, self.head_size)
                else:
                    states[i]["v_first"] = None
            else:
                # Eval: use pre-allocated buffers (no autograd) for speed
                # and to avoid CUDA allocator fragmentation
                if not hasattr(self, "_eval_bufs"):
                    self._ensure_eval_bufs()
                self._dequantize_to_bufs(i)

                s = states[i]
                bf16 = self._sd._bf16
                ln1 = F.layer_norm(h, (self.dim,),
                                   weight=bf16[f"blocks.{i}.ln1.weight"],
                                   bias=bf16[f"blocks.{i}.ln1.bias"])

                # Build a temporary dict with full keys pointing to shared bufs
                # _time_mix expects keys like "blocks.{i}.att.receptance.weight"
                w = {}
                for key in self._sd._layer_weights.get(i, []):
                    type_key = key.split(".", 2)[2]
                    w[key] = self._eval_bufs[type_key]

                h = h + self._time_mix(i, ln1, s, w)
                ln2 = F.layer_norm(h, (self.dim,),
                                   weight=bf16[f"blocks.{i}.ln2.weight"],
                                   bias=bf16[f"blocks.{i}.ln2.bias"])
                if str(i) in self.trainable_channels:
                    cm_out, _ = self.trainable_channels[str(i)](ln2)
                else:
                    ffn = f"blocks.{i}.ffn."
                    xx_c = s.get("xx_c", torch.zeros_like(h))
                    xk = ln2 + (xx_c - ln2) * bf16[ffn + "x_k"].squeeze()
                    k_c = F.relu(xk @ w[ffn + "key.weight"].T) ** 2
                    cm_out = k_c @ w[ffn + "value.weight"].T
                h = h + cm_out
                s["xx"] = ln1.detach().clone()
                s["xx_c"] = ln2.detach().clone()

        return h

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, byte_ids: torch.Tensor,
                return_logits: bool = True,
                states: Optional[List[dict]] = None) -> torch.Tensor:
        """Forward pass.
        If states is provided, resume from cached states.
        Returns (logits, new_states) when states is not None or T==1.
        """
        B, T = byte_ids.shape
        outputs = []
        return_states = states is not None or T == 1

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if not self.training and self.device.type == "cuda":
            self._ensure_eval_bufs()

        for t in range(T):
            bt = byte_ids[:, t]
            if (bt == BYTE_PAD).all():
                break

            h = F.embedding(bt, self._embed_w)

            if states is None:
                states = [{
                    "xx": torch.zeros(self.dim, device=bt.device, dtype=self.dtype),
                    "xx_c": torch.zeros(self.dim, device=bt.device, dtype=self.dtype),
                    "mat": torch.zeros(self.n_heads, self.head_size, self.head_size,
                                       device=bt.device, dtype=torch.float),
                    "v_first": None,
                } for _ in range(self.n_layers)]

            h = self._forward_token(h, states)
            outputs.append(h)

        if not outputs:
            empty = torch.zeros(B, 0, BYTE_VOCAB_SIZE, device=byte_ids.device)
            return (empty, states) if return_states else empty

        h_out = torch.stack(outputs, dim=1)
        if not return_logits:
            return (h_out, states) if return_states else h_out
        h_out = F.layer_norm(h_out, (self.dim,),
                             weight=self._ln_out_w, bias=self._ln_out_b)
        logits = h_out @ self._head_w.T
        return (logits, states) if return_states else logits

    def get_trainable_params(self):
        return list(self.trainable_channels.parameters())


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=== G1GBlackGooseNF4 ===")
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    layers = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [0]

    model = G1GBlackGooseNF4(layers_to_replace=layers, device=device)

    x = torch.randint(2, 258, (1, 4))
    if device == "cuda":
        x = x.cuda()
    logits = model(x)
    print(f"Forward: {x.shape} -> {logits.shape}")

    if device == "cuda":
        model.train()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        n_grad = sum(1 for p in model.get_trainable_params()
                     if p.grad is not None and p.grad.abs().sum() > 0)
        n_total = sum(1 for p in model.get_trainable_params())
        print(f"Gradients: {n_grad}/{n_total}, "
              f"GPU: {torch.cuda.memory_allocated()/1e9:.3f}GB")
    print("OK")
