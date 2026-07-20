"""Bottleneck detection via activation analysis.

Given a trained model + easy examples + hard examples, computes five
mathematically-defined metrics per channel per layer. No manual threshold
tuning — just compute, sort, pick top-k.

Designed to work with any torch.nn.Module: LatentThink, RWKVNano, etc.

Usage:
    python src/analyze_bottlenecks.py --model_path exp001/model.pt \
        --metric saturation --easy-n 500 --hard-n 500

Each metric file produces one output CSV per (layer, channel) pair, sorted
by anomaly score descending. The caller decides what to do with the top-k.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Hook manager — attach to any module, record pre/post activation per step
# ──────────────────────────────────────────────────────────────────────────────

class _Activations:
    """Mutable container so the hook callback can append in-place."""
    def __init__(self):
        self.stack: list[torch.Tensor] = []

    def push(self, t: torch.Tensor) -> None:
        self.stack.append(t.detach().cpu())

    def dump(self) -> list[torch.Tensor]:
        s = self.stack
        self.stack = []
        return s


class HookManager:
    """Register forward hooks on a model and collect activations batch-by-batch."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # name -> _Activations
        self._buffers: dict[str, _Activations] = {}

    def register(
        self,
        target_modules: list[str],  # dot-names like "blocks.0", "speak_fc.1"
        *,
        post: bool = True,        # hook after fn (post-activation) or before
        include_last: bool = False,
    ) -> None:
        """Register hooks on named submodules.

        Args:
            target_modules: dotted names within ``model``.
            post: if True, hook is attached *after* the submodule's forward;
                  if False, attaches *before*.
            include_last: also hook the final layer's raw output (useful for
                          encoder-only models like RWKVNano).
        """
        for name in target_modules:
            buf = _Activations()
            self._buffers[name] = buf
            module = self.model
            parts = name.split(".")
            for p in parts[:-1]:
                module = getattr(module, p)
            leaf = getattr(module, parts[-1])

            def hook_fn(m: nn.Module, inp: tuple, out: torch.Tensor, b=buf):
                # Handle tuple outputs (e.g. RNN returns (output, hidden))
                if isinstance(out, tuple):
                    out = out[0]
                # Grab last timestep if we want recurrent state summaries
                # For now always keep full tensor [B, T, C] or [B, C]
                b.push(out)

            h = leaf.register_forward_hook(hook_fn)
            self._handles.append(h)

        if include_last and target_modules:
            # Also hook the overall model forward output
            buf = _Activations()
            self._buffers["_model_output"] = buf
            h = self.model.register_forward_hook(
                lambda m, i, o, b=buf: b.push(o)
            )
            self._handles.append(h)

    @torch.no_grad()
    def collect(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict[str, list[torch.Tensor]]:
        """Run every batch through ``dataloader`` and return accumulated tensors."""
        self.model.eval()
        for d in self._buffers.values():
            d.stack.clear()
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                self.model(batch)
            elif isinstance(batch, dict):
                kwargs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                          for k, v in batch.items()}
                self.model(**kwargs)
            elif isinstance(batch, (list, tuple)):
                # DataLoader often yields (tensor,), (ctx, q), etc.
                device = next(self.model.parameters()).device
                args = [
                    b.to(device) if hasattr(b, 'to') else torch.as_tensor(b).to(device)
                    for b in batch
                ]
                self.model(*args)
        result: dict[str, list[torch.Tensor]] = {}
        for name, buf in self._buffers.items():
            result[name] = buf.dump()
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Metric computations — each takes stacked tensors → ranked scores
# ──────────────────────────────────────────────────────────────────────────────

def _squeeze_to_2d(ts: list[torch.Tensor]) -> list[torch.Tensor]:
    """Flatten [B, T, C] → [B*T, C] or [B, C] → [B, C]. Keeps B*C dimension."""
    out = []
    for t in ts:
        if t.ndim == 3:
            out.append(t.reshape(-1, t.shape[-1]))
        elif t.ndim == 2:
            out.append(t)
        elif t.ndim == 4:
            out.append(t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]))
        else:
            raise ValueError(f"Unexpected tensor dim: {t.shape}")
    return out


def _channel_stats(
    ts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) per channel over all batches, sample-dimension flattened."""
    stacked = torch.cat(_squeeze_to_2d(ts), dim=0)  # [N, C]
    mean = stacked.mean(dim=0)  # [C]
    var = stacked.var(dim=0, correction=0)
    std = var.sqrt().clamp(min=1e-8)
    return mean, std


def metric_saturation_ratio(
    easy_ts: list[torch.Tensor],
    hard_ts: list[torch.Tensor],
    n_sigma: float = 4.0,
) -> torch.Tensor:
    """M1. Fraction of hard inputs where |activation| exceeds μ_easy + k·σ_easy.

    Returns [C] — higher = more saturated under stress.
    """
    e_stack = torch.cat(_squeeze_to_2d(easy_ts), dim=0)
    h_stack = torch.cat(_squeeze_to_2d(hard_ts), dim=0)
    mu, sigma = e_stack.mean(dim=0), e_stack.std(dim=0).clamp(min=1e-8)
    threshold = mu + n_sigma * sigma
    ratio = (h_stack.abs() > threshold).float().mean(dim=0)  # [C]
    return ratio


def metric_range_expansion(
    easy_ts: list[torch.Tensor],
    hard_ts: list[torch.Tensor],
) -> torch.Tensor:
    """M2. Ratio of max-activated range: hard / easy.

    For rectified activations use max; otherwise use abs-max.
    Returns [C] — values >> 1.0 indicate unusual activity under stress.
    """
    e_stacked = torch.cat(_squeeze_to_2d(easy_ts), dim=0)
    h_stacked = torch.cat(_squeeze_to_2d(hard_ts), dim=0)
    e_max = e_stacked.abs().max(dim=0).values.clamp(min=1e-8)  # [C]
    h_max = h_stacked.abs().max(dim=0).values  # [C]
    return h_max / e_max


def metric_gradient_norm_shift(
    easy_loader: torch.utils.data.DataLoader,
    hard_loader: torch.utils.data.DataLoader,
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    grad_accumulator: Callable[..., list[list[float]]],
) -> torch.Tensor:
    """M3. Shift in fraction of total gradient norm flowing through each channel.

    Requires backward pass. See docstring for grad_accumulator contract.

    grad_accumulator(model, input_batch, loss_fn) -> list[fraction_of_total_norm_per_channel]
    Must sum to ~1.0 across channels.

    Returns [C] of ΔG = G_hard - G_easy. Higher positive = suddenly important
    (or collapsed onto fewer channels).
    """
    e_frac = grad_accumulator.easy_mean(easy_loader)
    h_frac = grad_accumulator.hard_mean(hard_loader)
    return h_frac - e_frac


def metric_effective_dimensionality(
    easy_ts: list[torch.Tensor],
    hard_ts: list[torch.Tensor],
) -> torch.Tensor:
    """Per-channel spectral efficiency shift (adapted M4).

    Measures how much each channel's contribution to the principal components
    shifts between easy and hard inputs. Large normalized changes indicate
    bottlenecks — channels whose representational role destabilizes under load.

    For easy set, compute SVD → singular values s_j, left singular vectors u[:,j].
    Channel c's weighted loading: sum_j (frac_energy[j] * u[c,j]^2)
    where frac_energy[j] = s_j^2 / sum(s^2) weights PCs by their explained variance.

    Returns [C] — higher = channel's principal structure role changed most.
    """
    e_stacked = torch.cat(_squeeze_to_2d(easy_ts), dim=0)  # [N_e, C]
    h_stacked = torch.cat(_squeeze_to_2d(hard_ts), dim=0)  # [N_h, C]

    C = e_stacked.shape[1]
    k = min(16, C)  # number of principal components to consider

    for name, stacked in [('easy', e_stacked), ('hard', h_stacked)]:
        center = stacked - stacked.mean(dim=0, keepdim=True)
        _, s, vh = torch.linalg.svd(center, full_matrices=False)

        # Fraction of total energy captured by each PC
        energy = s[:k] ** 2
        total_energy = energy.sum()
        frac_energy = energy / (total_energy + 1e-8)  # [k]

        # Per-channel weighted loading: vh[j, :] is the right singular vector
        # for PC j — how each original channel contributes to that PC.
        # Weight by explained variance fraction → channels central to dominant
        # PCs get high loading; shifts under stress signal bottlenecks.
        # vh[:k, :] has shape [k, C]: row j = PC j's loading across all channels.
        loading = (frac_energy.unsqueeze(1) * vh[:k, :].abs() ** 2).sum(dim=0)  # [C]

        if name == 'easy':
            easy_loading = loading
        else:
            hard_loading = loading

    # Normalized absolute change: channels whose structural role shifted most
    diff = (hard_loading - easy_loading).abs()
    norm_diff = diff / (easy_loading.abs() + 1e-8)
    return norm_diff


def metric_predictability_loss(
    easy_ts: list[torch.Tensor],
    hard_ts: list[torch.Tensor],
    labels_easy: torch.Tensor,  # [N_easy, L_out] target vectors
    labels_hard: torch.Tensor,  # [N_hard, L_out]
) -> torch.Tensor:
    """M5. Drop in linear predictability of output from this layer's activations.

    Fit W such that W · activations ≈ labels on easy set, then measure R² on hard.
    Returns [C] — positive = was useful on easy but not readable on hard.
    """
    e_act = torch.cat(_squeeze_to_2d(easy_ts), dim=0)  # [N_e, C]
    h_act = torch.cat(_squeeze_to_2d(hard_ts), dim=0)  # [N_h, C]

    e_labels = labels_easy.view(-1, labels_easy.shape[-1]).float()  # [N_e, L]
    h_labels = labels_hard.view(-1, labels_hard.shape[-1]).float()

    c = e_act.shape[1]
    l = e_labels.shape[1]
    scores = torch.zeros(c)

    for j in range(c):
        # Linear regression: e_act[:, j] → e_labels
        x = e_act[:, j].unsqueeze(1)  # [N_e, 1]
        X = torch.cat([x, torch.ones_like(x)], dim=1)  # add bias
        # Closed-form least squares: β = (X'X)^{-1} X'y
        XtX = X.T @ X + 1e-6 * torch.eye(2)
        Xty = X.T @ e_labels  # [2, L]
        beta = torch.linalg.solve(XtX, Xty)  # [2, L]

        # Predict on hard
        xh = h_act[:, j].unsqueeze(1)
        Xh = torch.cat([xh, torch.ones_like(xh)], dim=1)
        pred = Xh @ beta  # [N_h, L]

        r2_easy = _r2_score(e_labels, X @ beta)
        r2_hard = _r2_score(h_labels, pred)
        scores[j] = r2_easy - r2_hard

    return scores


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
    return r2.mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator — runs metrics, returns scored rows
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_bottlenecks(
    hook_mgr: HookManager,
    easy_loader: torch.utils.data.DataLoader,
    hard_loader: torch.utils.data.DataLoader,
    metrics: list[str] | None = None,
    *,
    labels_easy: Optional[torch.Tensor] = None,
    labels_hard: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable] = None,
) -> list[dict[str, Any]]:
    """Collect activations and run selected metrics.

    Args:
        hook_mgr: already registered on desired modules.
        easy_loader: DataLoader yielding easy examples.
        hard_loader: DataLoader yielding hard examples.
        metrics: subset of ["saturation", "range_expansion",
                         "effective_dimensionality", "predictability_loss"].
                 Defaults to all forward-pass metrics.
        labels_easy/hard: required only for predictability_loss.
        loss_fn: used internally by some metrics (placeholder for extensibility).

    Returns:
        List of dicts, one per (layer_name, channel_index), each containing
        the anomaly score for each requested metric plus the original tensors.
    """
    metrics = metrics or ["saturation", "range_expansion", "effective_dimensionality", "predictability_loss"]

    # Collect activations
    easy_acts = hook_mgr.collect(easy_loader)
    hard_acts = hook_mgr.collect(hard_loader)

    results: list[dict[str, Any]] = []
    has_sigmoid = False

    for layer_name in easy_acts.keys():
        e_tensors = easy_acts[layer_name]
        h_tensors = hard_acts.get(layer_name)
        if h_tensors is None:
            continue
        if len(e_tensors) != len(h_tensors):
            continue

        c = e_tensors[0].shape[-1]  # channel count

        scores: dict[str, torch.Tensor] = {}

        # M1: Saturation ratio
        if "saturation" in metrics:
            scores["saturation"] = metric_saturation_ratio(e_tensors, h_tensors)

        # M2: Range expansion
        if "range_expansion" in metrics:
            scores["range_expansion"] = metric_range_expansion(e_tensors, h_tensors)

        # M4: Effective dimensionality
        if "effective_dimensionality" in metrics:
            scores["effective_dimensionality"] = metric_effective_dimensionality(
                e_tensors, h_tensors,
            )

        # M5: Predictability loss
        if "predictability_loss" in metrics and labels_easy is not None and labels_hard is not None:
            scores["predictability_loss"] = metric_predictability_loss(
                e_tensors, h_tensors, labels_easy, labels_hard,
            )

        # Compile per-channel results
        for ch_idx in range(c):
            row: dict[str, Any] = {"layer": layer_name, "channel": int(ch_idx)}
            for metric_name, score_tensor in scores.items():
                row[f"{metric_name}_score"] = float(score_tensor[ch_idx])
            # Add an aggregate rank-friendly summary: mean anomaly across all metrics
            vals = [v for k, v in row.items() if k.endswith("_score")]
            row["aggregate_score"] = float(torch.tensor(vals).nanmean())
            results.append(row)

    # Sort by aggregate_score descending
    results.sort(key=lambda r: r["aggregate_score"], reverse=True)
    return results


def format_csv(results: list[dict[str, Any]]) -> str:
    """Format results as CSV for downstream processing."""
    if not results:
        return ""
    fieldnames = list(results[0].keys())
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
    return out.getvalue()
