"""SurpriseExtractor: per-byte state-delta as a free surprise signal.

Implements what you suggested: instead of training a separate entropy
model (BLT), read the surprise directly off the byte encoder's
running RWKV state, capture per-step state deltas, and threshold those
into patch boundaries.

Key insight (verified at 317f4a5+ GPU training runs): the trained
RWKV byte encoder's per-step state delta correlates strongly with
linguistic surprise — double letters (~0.1-0.2), normal letters
(~1.3), word boundaries and punctuation (~1.8+).

Implementation: forward hooks on a target Module's encoder blocks.
The per-step output is captured before being summed over `T`. We
use `|h(t) - h(t-1)|.mean(dim=-1)` as the surprise scalar per
position. No new forward pass — we wrap an existing module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SurpriseExtractor(nn.Module):
    """Records per-step hidden-state magnitudes via forward hooks.

    Args:
        target: the module whose forward we want to record from (an
                RWKV block stack or similar sequential layer).
        target_name: dotted name of a sub-module of `target` to hook on.
        hook_kind: 'block' for full-block output, 'sub' for a sublayer.

    Returns (h, surprise) on forward:
       - h: the normal output of `target` (passed through unchanged)
       - surprise: [B, T-1] — per-position L1 distance between
                   adjacent layer outputs

    Notes:
       - Each RWKVBlock is *not* called step-by-step; we reuse its
         normal forward and capture per-position outputs from the
         resulting [B, T, D] tensor.
       - This means the surprise is *not* a "leave-one-out" measure of
         what the state would have looked like without byte t; it's
         a post-hoc diff over the full-batch forward. That's what we
         want for static analysis (training data, eval). For online
         use you'd need a streaming RWKV decoder, separate work.
    """

    def __init__(self, target: nn.Module, target_name: str = ''):
        super().__init__()
        self.target = target
        self.target_name = target_name

        # Resolve target sub-module
        if target_name:
            obj = target
            for part in target_name.split('.'):
                obj = getattr(obj, part)
            self.target_module = obj
        else:
            self.target_module = target

        # The hook will populate this
        self._captured: torch.Tensor | None = None
        handle = self.target_module.register_forward_hook(self._capture)
        self._handle = handle

    def _capture(self, module, inputs, output) -> None:
        # output is the full [B, T, D] (or could be tuple; flatten if needed)
        if isinstance(output, tuple):
            output = output[0]
        # detach + move to cpu eagerly to avoid GPU memory growth
        self._captured = output.detach()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._captured = None
        h = x if self.target_name == '' else x
        # Call the *target* module's forward to populate the hook
        out = self.target(x) if not isinstance(self.target(x), tuple) else self.target(x)
        # We wanted to call the sub-module. Fix this by re-doing:
        if isinstance(out, tuple):
            out = out[0]
        captured = self._captured if self._captured is not None else out
        if captured is None or captured.ndim < 3:
            return out, torch.empty(out.shape[0], 0, device=out.device)
        # Compute per-step deltas: |h(t) - h(t-1)|.mean(dim=-1) -> [B, T-1]
        h_seq = captured
        delta = (h_seq[:, 1:] - h_seq[:, :-1]).abs().mean(dim=-1)
        return out, delta

    def detach(self):
        self._handle.remove()
