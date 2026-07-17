"""Dendrite RWKV - Frozen backbone + independent LoRA adapters + routing.

Simplified working version: inject LoRA into backbone, swap adapter weights for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from pathlib import Path
import json
import hashlib
import shutil
import random
import numpy as np
from safetensors.torch import save_file, load_file
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from src.rwkv_nano import RWKVNano


# ── LoRA Linear ──────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank

        in_dim = base.in_features
        out_dim = base.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

        # Freeze base
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    def get_lora_params(self):
        return [self.lora_A, self.lora_B]

    def merge(self):
        self.base.weight.data += (self.lora_B @ self.lora_A) * self.scale

    def unmerge(self):
        self.base.weight.data -= (self.lora_B @ self.lora_A) * self.scale


def inject_lora(model: RWKVNano, rank: int = 8, alpha: float = 16):
    """Inject LoRA into all projection layers of RWKV blocks."""
    lora_map = {}

    for i, block in enumerate(model.blocks):
        for name in ['key', 'value', 'receptance', 'output',
                     'fc_key', 'fc_value', 'fc_receptance']:
            layer = getattr(block, name)
            lora = LoRALinear(layer, rank, alpha)
            setattr(block, name, lora)
            lora_map[f'block{i}.{name}'] = lora

    return lora_map


# ── Address Head & PPCA Verifier ────────────────────────────────────

class AddressHead:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        self.fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.full((X.shape[0],), 0.5)
        return self.clf.predict_proba(X)[:, 1]


class PPCAVerifier:
    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.class_means = {}
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.pca.fit(X)
        X_red = self.pca.transform(X)
        for c in [0, 1]:
            mask = (y == c)
            self.class_means[c] = X_red[mask].mean(axis=0) if mask.any() else np.zeros(self.n_components)
        self.fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.full((X.shape[0],), 0.5)
        X_red = self.pca.transform(X)
        # Gaussian likelihood ratio
        logits = []
        for x in X_red:
            ll0 = -0.5 * np.sum((x - self.class_means[0]) ** 2)
            ll1 = -0.5 * np.sum((x - self.class_means[1]) ** 2)
            logits.append(ll1 - ll0)
        return 1 / (1 + np.exp(-np.array(logits)))


# ── Adapter Registry ────────────────────────────────────────────────

class AdapterRegistry:
    def __init__(self, registry_dir: str):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir = self.registry_dir / "quarantine"
        self.quarantine_dir.mkdir(exist_ok=True)

        self.manifest_path = self.registry_dir / "manifest.json"
        self.metadata = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def _hash_file(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _hash_state(self, state_dict: Dict) -> str:
        data = b""
        for k in sorted(state_dict.keys()):
            data += k.encode() + state_dict[k].cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def install(self, name: str, lora_params: Dict[str, torch.Tensor], probe_logits_fn) -> Dict:
        """Install adapter with integrity gates."""
        adapter_dir = self.registry_dir / name
        adapter_dir.mkdir(exist_ok=True)

        # Save as safetensors
        save_file(lora_params, adapter_dir / "adapter.safetensors")

        # Gate 1: install integrity
        saved_hash = self._hash_file(adapter_dir / "adapter.safetensors")
        reloaded = load_file(adapter_dir / "adapter.safetensors")
        reloaded_hash = self._hash_state(reloaded)
        gate1 = saved_hash == reloaded_hash

        # Gate 2: functional equivalence
        probe_out1 = probe_logits_fn(lora_params)
        probe_out2 = probe_logits_fn(reloaded)
        gate2 = (probe_out1 - probe_out2).abs().max().item() < 1e-5

        # Gate 3: adapter hash consistency
        gate3 = saved_hash == reloaded_hash

        passed = gate1 and gate2 and gate3

        if passed:
            self.metadata[name] = {"hash": saved_hash, "installed": True}
            self.manifest_path.write_text(json.dumps(self.metadata, indent=2))

        return {"gate1": gate1, "gate2": gate2, "gate3": gate3, "all_passed": passed}

    def delete(self, name: str) -> bool:
        adapter_dir = self.registry_dir / name
        if not adapter_dir.exists():
            return False
        shutil.move(str(adapter_dir), str(self.quarantine_dir / f"{name}_deleted"))
        if name in self.metadata:
            self.metadata[name]["installed"] = False
            self.manifest_path.write_text(json.dumps(self.metadata, indent=2))
        return not (self.registry_dir / name).exists()

    def load(self, name: str) -> Dict:
        adapter_dir = self.registry_dir / name
        return load_file(adapter_dir / "adapter.safetensors")


# ── Dendrite RWKV Model ────────────────────────────────────────────

class DendriteRWKV(nn.Module):
    """RWKV backbone with injectable LoRA adapters."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        num_layers: int = 3,
        hidden_scale: int = 4,
        adapter_names: Optional[List[str]] = None,
        lora_rank: int = 8,
        lora_alpha: float = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.adapter_names = adapter_names or []
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Frozen backbone
        self.backbone = RWKVNano(vocab_size, dim, num_layers, hidden_scale)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Inject LoRA once (we'll swap weights per adapter)
        self.lora_map = inject_lora(self.backbone, lora_rank, lora_alpha)

        # Store adapter weights separately: name -> {param_name: tensor}
        self.adapter_weights = nn.ParameterDict()
        for name in self.adapter_names:
            for k, lora in self.lora_map.items():
                # Replace dots in key for ParameterDict compatibility
                safe_k = k.replace('.', '_')
                self.adapter_weights.register_parameter(
                    f"{name}__{safe_k}__A", nn.Parameter(lora.lora_A.data.clone())
                )
                self.adapter_weights.register_parameter(
                    f"{name}__{safe_k}__B", nn.Parameter(lora.lora_B.data.clone())
                )

        # Routing (fitted after training)
        self.address_heads = {n: AddressHead() for n in self.adapter_names}
        self.verifiers = {n: PPCAVerifier() for n in self.adapter_names}

    def activate_adapter(self, name: str):
        """Load adapter weights into LoRA layers."""
        for k, lora in self.lora_map.items():
            safe_k = k.replace('.', '_')
            A = self.adapter_weights[f"{name}__{safe_k}__A"]
            B = self.adapter_weights[f"{name}__{safe_k}__B"]
            lora.lora_A.data.copy_(A.data)
            lora.lora_B.data.copy_(B.data)

    def get_active_lora_params(self, name: str) -> Dict[str, torch.Tensor]:
        """Get LoRA params for an adapter (for saving)."""
        params = {}
        for k, lora in self.lora_map.items():
            safe_k = k.replace('.', '_')
            params[f"{k}.A"] = self.adapter_weights[f"{name}__{safe_k}__A"]
            params[f"{k}.B"] = self.adapter_weights[f"{name}__{safe_k}__B"]
        return params

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids)[0]

    def get_hidden(self, input_ids: torch.Tensor, tap_layer: int = 1) -> torch.Tensor:
        x = self.backbone.embed(input_ids)
        for i, block in enumerate(self.backbone.blocks):
            x, _ = block(x)
            if i == tap_layer:
                return x
        return x

    def pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([hidden.mean(dim=1), hidden[:, -1]], dim=1)

    def fit_address_head(self, name: str, X: np.ndarray, y: np.ndarray):
        self.address_heads[name].fit(X, y)

    def fit_verifier(self, name: str, X: np.ndarray, y: np.ndarray):
        self.verifiers[name].fit(X, y)

    def route(self, input_ids: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            hidden = self.get_hidden(input_ids)
            pooled = self.pool_hidden(hidden).cpu().numpy()

        scores = {}
        for name in self.adapter_names:
            addr = self.address_heads[name].predict_proba(pooled).mean()
            ver = self.verifiers[name].predict_proba(pooled).mean()
            scores[name] = float(addr * ver)
        return scores


# ── Training ────────────────────────────────────────────────────────

def train_adapter(
    model: DendriteRWKV,
    adapter_name: str,
    train_loader,
    val_loader,
    steps: int = 500,
    lr: float = 3e-4,
    device: str = 'cuda',
) -> Dict:
    model.activate_adapter(adapter_name)
    model.train()

    # Only adapter params are trainable
    adapter = model.lora_map
    params = []
    for lora in adapter.values():
        params.extend(lora.get_lora_params())

    optimizer = torch.optim.AdamW(params, lr=lr)

    best_acc = 0
    for step in range(steps):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits[:, -1], y)
            loss.backward()
            optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    pred = logits[:, -1].argmax(-1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
            acc = correct / total
            best_acc = max(best_acc, acc)
            print(f"  step {step}: val_acc={acc:.4f} (best={best_acc:.4f})")

    return {"val_acc": best_acc, "steps": steps}


# ── Quick Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    model = DendriteRWKV(vocab_size=64, dim=64, num_layers=2, adapter_names=['test'])
    print(f"Backbone params: {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"Adapter params: {sum(p.numel() for p in model.adapter_weights.parameters()):,}")
    x = torch.randint(1, 63, (2, 32))
    logits = model(x)
    print(f"Forward: {logits.shape}")
    model.activate_adapter('test')
    logits = model(x)
    print("OK")


def gen_sum_threshold(n_samples: int, threshold: int = 200) -> List[Tuple[List[int], int]]:
    """Sum of digits in sequence >= threshold?"""
    data = []
    for _ in range(n_samples):
        length = random.randint(8, 20)
        seq = [random.randint(1, 9) for _ in range(length)]
        total = sum(seq)
        label = 1 if total >= threshold else 0
        seq.append(label + 10)
        data.append((seq, label))
    return data


def gen_vowel_majority(n_samples: int) -> List[Tuple[List[int], int]]:
    """Vowels (a,e,i,o,u) > consonants?"""
    vowel_ids = {1, 5, 9, 15, 21}  # a,e,i,o,u in 1-26 mapping
    data = []
    for _ in range(n_samples):
        length = random.randint(8, 20)
        seq = [random.randint(1, 26) for _ in range(length)]
        vowels = sum(1 for c in seq if c in vowel_ids)
        label = 1 if vowels > len(seq) - vowels else 0
        seq.append(label + 10)
        data.append((seq, label))
    return data


def gen_endpoint_match(n_samples: int) -> List[Tuple[List[int], int]]:
    """First and last char match?"""
    data = []
    for _ in range(n_samples):
        length = random.randint(8, 20)
        seq = [random.randint(1, 26) for _ in range(length)]
        label = 1 if seq[0] == seq[-1] else 0
        seq.append(label + 10)
        data.append((seq, label))
    return data


def gen_count_trigger(n_samples: int) -> List[Tuple[List[int], int]]:
    """Count of 'x' (token 24) > 3?"""
    data = []
    for _ in range(n_samples):
        length = random.randint(8, 20)
        seq = [random.randint(1, 26) for _ in range(length)]
        count_x = sum(1 for c in seq if c == 24)
        label = 1 if count_x > 3 else 0
        seq.append(label + 10)
        data.append((seq, label))
    return data