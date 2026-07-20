#!/usr/bin/env python3
"""Shared neural building blocks.

Rule for this file: a block lands here only after it was duplicated in two
or more threads *with identical semantics* (or is designed as the canonical
version of a duplicated block). One-off thread ideas stay in the thread.

[meta]
status: active
[/meta]
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing frozen nn.Linear.

    out = base(x) + (x @ Aᵀ @ Bᵀ) * (alpha / rank)

    Canonical version of the pattern copied across dendrite/memory threads:
    B starts at zero, so the adapter is initially an identity change and the
    frozen trunk is untouched. Base weights are frozen in-place.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    def trainable_parameters(self):
        return [self.lora_A, self.lora_B]
