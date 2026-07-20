#!/usr/bin/env python3
"""Byte-level inference engine for the g1g 2.9B RWKV-7 model.

Stateful token-by-token generation with optional NF4 quantization
for fitting on small GPUs.

Usage:
    from domains.g1g.auto_tokenizer import ByteG1GInference

    # GPU with NF4 quantization
    model = ByteG1GInference(device='cuda', quant='nf4')
    out = model.generate("Once upon a time", max_new_bytes=200)
    print(out)

    # CPU (no quantization)
    model = ByteG1GInference(device='cpu')
    out = model.generate("Hello", max_new_bytes=50)
    print(out)


[meta]
status: active
[/meta]
"""

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ── Byte vocabulary ──────────────────────────────────────────────────────

BYTE_VOCAB_SIZE = 258
BYTE_PAD = 0
BYTE_UNK = 1

BYTE_TO_ID = {b: 2 + b for b in range(256)}
ID_TO_BYTE = {v: k for k, v in BYTE_TO_ID.items()}


def encode(text: str, max_len: int = 2048) -> torch.Tensor:
    raw = text.encode("utf-8")
    ids = [BYTE_TO_ID[b] for b in raw]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)


def decode(byte_ids: torch.Tensor) -> str:
    ids = byte_ids.tolist()
    raw = bytes(ID_TO_BYTE[i] for i in ids if i >= 2)
    return raw.decode("utf-8", errors="replace")


# ── Model paths ───────────────────────────────────────────────────────────

MODEL_DIR = Path.home() / "Documents" / "models" / "rwkv7-g1g-byte-iface"
DEFAULT_MODEL_PATH = MODEL_DIR / "model.pth"
NF4_CACHE_DIR = MODEL_DIR / "nf4_cache"
NF4_INDEX_PATH = NF4_CACHE_DIR / "index.json"


# ── Weight helpers ────────────────────────────────────────────────────────

def _is_big_linear(key: str) -> bool:
    """Heuristic: big = 2D weight tensor in att or ffn, > 100K elements."""
    return (key.endswith('.weight')
            and ('att.' in key or 'ffn.' in key))


def _load_nf4_cache(device: torch.device) -> dict:
    """Load pre-computed NF4 quantized weights from disk cache.

    Returns dict mapping weight key -> (quantized_tensor, QuantState)
    already on device.
    """
    from bitsandbytes.functional import QuantState

    index = json.loads(NF4_INDEX_PATH.read_text())
    result = {}
    for k, cache_path in index.items():
        data = torch.load(cache_path, map_location=device, weights_only=True)
        qs = QuantState(
            absmax=data['absmax'],
            shape=data['shape'],
            code=data['code'],
            blocksize=data['blocksize'],
            dtype=data['dtype'],
            quant_type=data['quant_type'],
        )
        result[k] = (data['q'], qs)
    return result


# ── ByteG1GInference ──────────────────────────────────────────────────────

class ByteG1GInference:
    """Stateful byte-level inference engine for g1g 2.9B.

    Args:
        model_path: path to model.pth
        device: 'cpu' or 'cuda'
        dtype: default bfloat16
        quant: None for bf16, 'nf4' for 4-bit quantization using cache
        verbose: print status
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quant: Optional[str] = None,
        verbose: bool = True,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.quant = quant

        t0 = time.time()
        sd = torch.load(model_path, map_location="cpu", weights_only=True)

        D = sd["ln_out.weight"].shape[0]
        head_size = sd["blocks.0.att.r_k"].shape[-1]
        n_heads = D // head_size
        n_layers = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".ln1.weight"))

        self.dim = D
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_size = head_size

        # Load weights to device
        is_cuda = device.startswith('cuda')

        if quant == 'nf4' and is_cuda:
            from bitsandbytes.functional import dequantize_4bit
            self._dequantize = dequantize_4bit

            qcache = _load_nf4_cache(self.device)
            for k, v in list(sd.items()):
                if k in qcache:
                    # Quantize middle layers (4-27), keep first/last in bf16
                    layer_num = None
                    if k.startswith('blocks.'):
                        try:
                            layer_num = int(k.split('.')[1])
                        except (IndexError, ValueError):
                            pass
                    if layer_num is not None and 4 <= layer_num <= 27:
                        # NF4 entry already on device — drop the fp32 copy now
                        # so the full 5.5GB model never stays resident.
                        sd[k] = qcache[k]
                        del v
                        continue
                    else:
                        sd[k] = v.to(device=self.device, dtype=self.dtype)
                        del v
                elif isinstance(v, torch.Tensor):
                    sd[k] = v.to(device=self.device, dtype=self.dtype)
                    del v
                else:
                    sd[k] = v
            self._sd = sd
            total = sum(v.numel() if isinstance(v, torch.Tensor) else v[0].numel() * 2
                       for v in sd.values())
        else:
            # Load everything in bf16
            for k, v in list(sd.items()):
                if isinstance(v, torch.Tensor):
                    sd[k] = v.to(device=self.device, dtype=self.dtype)
            self._sd = sd
            total = sum(v.numel() for v in sd.values())

        self.reset()
        torch.cuda.empty_cache() if is_cuda else None

        if verbose:
            gpu_mem = ""
            if is_cuda:
                gpu_mem = f", GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB"
            print(f"ByteG1GInference: {n_layers} layers, {D} dim, "
                  f"{n_heads}×{head_size} heads, {total:,} params "
                  f"({time.time()-t0:.1f}s{gpu_mem})")

    def reset(self):
        """Zero all recurrent states."""
        D, H, N = self.dim, self.n_heads, self.head_size
        dev, dt = self.device, self.dtype
        self.states = [
            {
                "xx": torch.zeros(D, device=dev, dtype=dt),
                "xx_c": torch.zeros(D, device=dev, dtype=dt),
                "mat": torch.zeros(H, N, N, device=dev, dtype=torch.float),
                "v_first": None,
            }
            for _ in range(self.n_layers)
        ]

    def get_w(self, key: str) -> torch.Tensor:
        """Get weight tensor, dequantizing if stored as NF4 tuple."""
        v = self._sd[key]
        if isinstance(v, tuple):
            return self._dequantize(v[0], v[1]).to(dtype=self.dtype)
        return v

    @torch.no_grad()
    def step(self, byte_id: int) -> torch.Tensor:
        """Process one byte, return logits for next byte."""
        sd = self._sd
        D, H, N = self.dim, self.n_heads, self.head_size
        dev, dt = self.device, self.dtype

        h = F.embedding(
            torch.tensor([byte_id], device=dev, dtype=torch.long),
            sd["byte_embed.weight"],
        ).squeeze(0)

        for i in range(self.n_layers):
            s = self.states[i]

            # ── LN1 ──
            ln1 = F.layer_norm(
                h, (D,),
                weight=sd[f"blocks.{i}.ln1.weight"],
                bias=sd[f"blocks.{i}.ln1.bias"],
            )
            att = f"blocks.{i}.att."

            # ── Shift ──
            xx = s["xx"] - ln1
            xr = ln1 + xx * sd[att + "x_r"].squeeze()
            xw = ln1 + xx * sd[att + "x_w"].squeeze()
            xk = ln1 + xx * sd[att + "x_k"].squeeze()
            xv = ln1 + xx * sd[att + "x_v"].squeeze()
            xa = ln1 + xx * sd[att + "x_a"].squeeze()
            xg = ln1 + xx * sd[att + "x_g"].squeeze()

            # ── Proj ──
            r = xr @ self.get_w(att + "receptance.weight")
            w = torch.tanh(xw @ self.get_w(att + "w1")) @ self.get_w(att + "w2")
            k = xk @ self.get_w(att + "key.weight")
            v = xv @ self.get_w(att + "value.weight")
            a = torch.sigmoid(
                sd[att + "a0"].squeeze()
                + (xa @ self.get_w(att + "a1")) @ self.get_w(att + "a2")
            )
            g = torch.sigmoid(xg @ self.get_w(att + "g1")) @ self.get_w(att + "g2")

            # ── Multi-head ──
            def to_h(t): return t.view(H, N)
            def from_h(t): return t.reshape(D)

            r_h, k_h, v_h, a_h, g_h, w_h = map(to_h, [r, k, v, a, g, w])

            # ── Key norm ──
            kk = F.normalize(
                k_h * to_h(sd[att + "k_k"].squeeze()), dim=-1, p=2.0
            )
            k_adj = k_h * (1 + (a_h - 1) * to_h(sd[att + "k_a"].squeeze()))

            # ── Value residual ──
            if s["v_first"] is None:
                s["v_first"] = v_h.clone()
            else:
                blend = torch.sigmoid(
                    to_h(sd[att + "v0"].squeeze())
                    + (xv @ self.get_w(att + "v1") @ self.get_w(att + "v2")).view(H, N)
                )
                v_h = v_h + (s["v_first"] - v_h) * blend

            # ── Decay ──
            w0 = sd[att + "w0"].squeeze()
            w_decay = torch.exp(
                -0.606531 * torch.sigmoid((to_h(w0) + w_h).float())
            )

            # ── Matrix state ──
            mat = s["mat"]
            vk = v_h.unsqueeze(-1) @ k_adj.unsqueeze(-2)
            ab = (-kk).unsqueeze(-1) @ (kk * a_h).unsqueeze(-2)
            mat = (mat * w_decay.unsqueeze(-2).float()
                   + (mat @ ab.float())
                   + vk.float())

            out_h = (mat.to(dtype=ln1.dtype) @ r_h.unsqueeze(-1)).squeeze(-1)
            out_flat = from_h(out_h)

            out_flat = F.group_norm(
                out_flat.view(1, D), num_groups=H,
                weight=sd[att + "ln_x.weight"],
                bias=sd[att + "ln_x.bias"], eps=64e-5,
            ).view(D)

            shortcut = (r_h * k_h * sd[att + "r_k"]).sum(dim=-1, keepdim=True) * v_h
            out_flat = out_flat + from_h(shortcut)
            out_flat = out_flat * g
            tm_out = out_flat @ self.get_w(att + "output.weight")
            h = h + tm_out

            # ── Channel-mix ──
            ln2 = F.layer_norm(
                h, (D,),
                weight=sd[f"blocks.{i}.ln2.weight"],
                bias=sd[f"blocks.{i}.ln2.bias"],
            )
            ffn = f"blocks.{i}.ffn."
            xx_c = s["xx_c"] - ln2
            xk_c = ln2 + xx_c * sd[ffn + "x_k"].squeeze()
            k_c = F.relu(xk_c @ self.get_w(ffn + "key.weight").T) ** 2
            v_c = k_c @ self.get_w(ffn + "value.weight").T
            h = h + v_c

            # ── Update state ──
            s["xx"] = ln1.detach().clone()
            s["xx_c"] = ln2.detach().clone()
            s["mat"] = mat.detach().clone()

        h = F.layer_norm(h, (D,),
                         weight=sd["ln_out.weight"],
                         bias=sd["ln_out.bias"])
        logits = h @ sd["byte_head.weight"].T
        return logits

    def feed_prefix(self, text: str):
        """Feed a prompt string, updating state. Returns last logits."""
        byte_ids = encode(text)
        logits = None
        for bid in byte_ids.tolist():
            logits = self.step(bid)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_new_bytes: int = 500,
        temperature: float = 0.8,
        top_p: float = 0.9,
        verbose: bool = False,
    ) -> str:
        """Generate text from a prompt."""
        self.reset()
        if prompt:
            self.feed_prefix(prompt)

        generated_ids = []
        next_byte_id = BYTE_TO_ID.get(ord("\n"), BYTE_UNK)

        for i in range(max_new_bytes):
            logits = self.step(next_byte_id)

            probs = F.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()

            next_byte_id = int(torch.multinomial(sorted_probs, 1).item())
            next_byte_id = int(sorted_indices[next_byte_id].item())

            if next_byte_id == BYTE_PAD:
                break
            generated_ids.append(next_byte_id)

            if verbose and i % 50 == 0:
                partial = decode(torch.tensor(generated_ids))
                print(f"  [{i:4d}] {partial[-40:]}")

        return decode(torch.tensor(generated_ids))


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    quant = sys.argv[2] if len(sys.argv) > 2 else ("nf4" if device == "cuda" else None)

    model = ByteG1GInference(device=device, quant=quant, verbose=True)

    if device == "cpu":
        # Quick sanity on CPU
        model.feed_prefix("Hi")
        logits = model.step(32)
        print(f"CPU test OK: logits shape={logits.shape}")
    else:
        prompt = "Once upon a time,"
        print(f"\nPrompt: {prompt!r}")
        t0 = time.time()
        out = model.generate(prompt, max_new_bytes=500, temperature=0.8, verbose=True)
        elapsed = time.time() - t0
        print(f"\n--- Generated in {elapsed:.1f}s ---")
        print(prompt + out)
