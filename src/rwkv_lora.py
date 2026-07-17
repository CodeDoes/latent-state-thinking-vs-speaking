"""LoRA adapters for RWKV blocks — injected into the projection layers.

Targets: key, value, receptance, output, fc_key, fc_value, fc_receptance
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List


class LoRALinear(nn.Module):
    """LoRA wrapper around nn.Linear: W + A @ B * scale"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 0.0

        self.base = nn.Linear(in_features, out_features, bias=bias)
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            # Init
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            out = out + self.dropout(self.lora_B(self.lora_A(x))) * self.scale
        return out

    def merge(self):
        """Merge LoRA weights into base for faster inference"""
        if self.r > 0:
            self.base.weight.data += (self.lora_B.weight @ self.lora_A.weight) * self.scale
            if self.base.bias is not None and self.lora_B.bias is not None:
                self.base.bias.data += self.lora_B.bias * self.scale
            self.r = 0
            self.lora_A = None
            self.lora_B = None

    def state_dict(self, *args, **kwargs):
        """Override to save only LoRA params when frozen"""
        state = {}
        if self.r > 0:
            state['lora_A.weight'] = self.lora_A.weight.data
            state['lora_B.weight'] = self.lora_B.weight.data
            if self.base.bias is not None:
                state['base.bias'] = self.base.bias.data
        return state

    def load_state_dict(self, state, strict=True):
        if self.r > 0 and 'lora_A.weight' in state:
            self.lora_A.weight.data.copy_(state['lora_A.weight'])
            self.lora_B.weight.data.copy_(state['lora_B.weight'])
            if 'base.bias' in state and self.base.bias is not None:
                self.base.bias.data.copy_(state['base.bias'])


def inject_lora_into_rwkv_block(
    block: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Replace target Linear layers in an RWKVBlock with LoRALinear.

    Args:
        block: RWKVBlock module
        r: LoRA rank
        alpha: LoRA scaling alpha
        dropout: LoRA dropout
        target_modules: list of attribute names to replace. Defaults to all
            projection layers in time-mix and channel-mix.
    """
    if target_modules is None:
        target_modules = [
            'key', 'value', 'receptance', 'output',
            'fc_key', 'fc_value', 'fc_receptance'
        ]

    for name in target_modules:
        if hasattr(block, name):
            layer = getattr(block, name)
            if isinstance(layer, nn.Linear):
                lora_layer = LoRALinear(
                    layer.in_features, layer.out_features,
                    r=r, alpha=alpha, dropout=dropout, bias=layer.bias is not None
                )
                # Copy base weights
                lora_layer.base.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    lora_layer.base.bias.data.copy_(layer.bias.data)
                setattr(block, name, lora_layer)

    return block


def get_lora_params(model: nn.Module):
    """Yield only LoRA parameters (A and B matrices)"""
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            yield param


def get_base_params(model: nn.Module):
    """Yield backbone (non-LoRA) parameters"""
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            yield param


def freeze_base(model: nn.Module):
    """Freeze all non-LoRA parameters"""
    for p in get_base_params(model):
        p.requires_grad = False


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in get_lora_params(model))


def count_base_params(model: nn.Module) -> int:
    return sum(p.numel() for p in get_base_params(model))


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.rwkv_nano import RWKVBlock
    block = RWKVBlock(dim=128, hidden_scale=4)
    block = inject_lora_into_rwkv_block(block, r=8, alpha=16)
    print(f"Base params: {count_base_params(block):,}")
    print(f"LoRA params: {count_lora_params(block):,}")
    x = torch.randn(2, 16, 128)
    out, _ = block(x)
    print(f"Out: {out.shape}")
    # Test grad flow
    out.sum().backward()
    lora_grads = sum(1 for p in get_lora_params(block) if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"LoRA params with grad: {lora_grads}")