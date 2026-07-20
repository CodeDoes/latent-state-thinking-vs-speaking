"""CLI for progressive-expansion bottleneck analysis.

Runs a trained model through easy/hard samples, computes activation metrics,
and outputs a ranked list of bottleneck candidates (layer + channel).

Usage:
    # Load any experiment checkpoint, auto-detect architecture
    python src/analyze_cli.py --checkpoint experiments/exp_diag1/checkpoint.pt \
        --easy-n 500 --hard-n 500

    # Specify which metrics to compute
    python src/analyze_cli.py --checkpoint ... --metrics saturation range_expansion

    # Save CSV output
    python src/analyze_cli.py --checkpoint ... --output results.csv

Outputs:
    - STDOUT: top-N bottleneck channels as formatted table
    - File:   full CSV if --output specified
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

# Allow running as 'python src/analyze_cli.py' from repo root
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from threads.unsorted.bottleneck_analysis import (
    HookManager,
    analyze_bottlenecks,
    format_csv,
)


# ──────────────────────────────────────────────────────────────────────────────
# Architecture detection & model loading
# ──────────────────────────────────────────────────────────────────────────────

def detect_architecture(state_dict):
    """Detect model architecture from state_dict keys."""
    for k in state_dict.keys():
        if 'time_decay' in k and k.startswith('blocks'):
            return 'rwkv'
        elif k.startswith('enc.') or k == 'emb.weight':
            # Distinguish LatentThink vs BaselineAR by presence of speak_fc
            if any('speak_' in name for name in state_dict.keys()):
                return 'latent_think'
            else:
                return 'baseline_ar'
    raise ValueError(f"Unknown architecture: first key = {list(state_dict.keys())[0]}")


def load_model(path, device='cpu'):
    """Load trained model from any experiment checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state', ckpt)
    config = ckpt.get('config', {})
    arch = detect_architecture(sd)

    if arch == 'rwkv':
        return _load_rwkv(sd, device)
    elif arch == 'latent_think':
        return _load_latent_think(sd, device)
    elif arch == 'baseline_ar':
        return _load_baseline_ar(sd, device)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


@torch.no_grad()
def _load_rwkv(sd, device):
    """Instantiate RWKVNano from its state dict."""
    from domains.rwkv.rwkv_nano import RWKVNano
    
    vocab_size = sd['embed.weight'].shape[0]
    dim = sd['blocks.0.time_decay'].shape[0]
    
    # Count distinct block indices
    block_indices = set()
    for k in sd:
        if k.startswith('blocks.'):
            idx = int(k.split('.')[1])
            block_indices.add(idx)
    num_layers = max(block_indices) + 1 if block_indices else 1
    
    # Infer hidden_scale from fc_key shape if available
    hidden_scale = 4
    key_candidates = [k for k in sd if k.endswith('.fc_key.weight')]
    if key_candidates:
        hidden_dim = sd[key_candidates[0]].shape[0]
        hidden_scale = round(hidden_dim / dim)
    
    model = RWKVNano(
        vocab_size=vocab_size, dim=dim,
        num_layers=num_layers, hidden_scale=hidden_scale,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN: {len(missing)} missing keys (checkpoint predates model): {missing}")
    return model


def _load_latent_think(sd, device):
    """Instantiate LatentThink from its state dict."""
    from threads.memory_growth.models import LatentThink
    
    vocab_size = sd['emb.weight'].shape[0]
    d_hidden_enc = sd['enc.weight_hh_l0'].shape[0] // 6  # GRU = 6x hidden
    speak_0_shape = sd['speak_fc.0.weight'].shape
    d_emb = speak_0_shape[1] // 2  # cat([state, qv]) where both are d_emb/d_hidden_enc
    d_hidden_speak = speak_0_shape[0]
    
    model = LatentThink(
        vocab_size, d_emb=d_emb,
        d_hidden_enc=d_hidden_enc, d_hidden_speak=d_hidden_speak,
    )
    model.load_state_dict(sd, strict=False)
    return model


def _load_baseline_ar(sd, device):
    """Instantiate BaselineAR from its state dict."""
    from threads.memory_growth.models import BaselineAR
    
    vocab_size = sd['emb.weight'].shape[0]
    d_hidden = sd['gru.weight_hh_l0'].shape[0] // 3  # GRU = 3x hidden
    
    model = BaselineAR(vocab_size, d_emb=16, d_hidden=d_hidden)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Data pipelines
# ──────────────────────────────────────────────────────────────────────────────

class RwkvDataPipeline:
    """Generates easy/hard examples for RWKVNano using LogicNiiahGenerator.
    
    Easy = fewer variables/fewer transforms. Hard = more variables/more transforms.
    """
    def __init__(self, seed=42):
        self.seed = seed
        from threads.memory_growth.logic_niiah_generator import LogicNiiahGenerator
        self.generator = LogicNiiahGenerator(seed=seed)
        
        # Char-level vocab and tokenizer from train_rwkv
        self.CHARS = [
            '\n', ' ', '!', ',', '-', '.', ':', '=',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        ]
        SPECIAL = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        VOCAB = SPECIAL + self.CHARS
        self.char_to_id = {c: i for i, c in enumerate(VOCAB)}
        self.PAD_ID = self.char_to_id['<PAD>']
        self.UNK_ID = self.char_to_id['<UNK>']
        self.max_len = 256

    def encode(self, text):
        tokens = [self.char_to_id.get(c, self.UNK_ID) for c in text]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.PAD_ID] * (self.max_len - len(tokens))
        return tokens

    def build_dataset(self, n, difficulty_params):
        """Return list of (input_ids_tensor,) tuples."""
        from threads.memory_growth.logic_niiah_generator import LogicNiiahGenerator
        gen = LogicNiiahGenerator(seed=self.seed)
        examples = []
        for _ in range(n):
            ex = gen.generate(**difficulty_params)
            tokens = self.encode(ex['text'])
            examples.append((torch.tensor(tokens, dtype=torch.long),))
        return examples

    def build_dataloader(self, n, difficulty_params, batch_size=32):
        """Build DataLoader yielding (B, T) integer tensors."""
        examples = self.build_dataset(n, difficulty_params)
        dataset = torch.utils.data.TensorDataset(torch.stack([e[0] for e in examples]))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


class WorldModelPipeline:
    """Generates easy/hard examples for LatentThink/BaselineAR.
    
    Uses dataset.py which produces string tokens. Maps them to integer IDs
    via vocabulary. Easy = short chains (max_events=4). Hard = long chains (max_events=10).
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.vocab, self.cats = self._make_vocab()
        self._str_to_id = {s: i for i, s in enumerate(self.vocab)}
        self.n_queries = 4  # fixed number of queries per world for batching consistency

    @staticmethod
    def _make_vocab():
        from threads.memory_growth.dataset import build_vocab
        return build_vocab()

    def _tokenize_list(self, items):
        """Map list of strings to list of ints."""
        return [self._str_to_id[s] if s in self._str_to_id else 0 for s in items]

    def build_dataloader(self, n, difficulty_params, batch_size=16):
        """Returns DataLoader yielding dicts with 'context' and 'question' tensors."""
        from threads.memory_growth.dataset import generate_dataset
        
        worlds = generate_dataset(n=n, seed=self.seed, **difficulty_params)
        
        # Build per-query samples: one entry per (world, query_index)
        samples = []
        for world in worlds:
            ctx_ids = self._tokenize_list(world['context'])
            for q in world['queries'][:self.n_queries]:
                q_tokens = self._tokenize_list(q[0])
                answer_id = self._str_to_id.get(q[1], 0)
                samples.append((ctx_ids, q_tokens, answer_id))
        
        return torch.utils.data.DataLoader(samples, batch_size=batch_size, collate_fn=self._collate)

    @staticmethod
    def _collate(batch):
        ctxs, qs, ans = zip(*batch)
        max_ctx = max(len(c) for c in ctxs)
        max_q = max(len(q) for q in qs)
        
        ctx_pad = torch.zeros(len(ctxs), max_ctx, dtype=torch.long)
        q_pad = torch.zeros(len(ctxs), max_q, dtype=torch.long)
        for i, (c, q) in enumerate(zip(ctxs, qs)):
            ctx_pad[i, :len(c)] = torch.tensor(c)
            q_pad[i, :len(q)] = torch.tensor(q)
        
        return {'context': ctx_pad, 'question': q_pad, 'answer': torch.tensor(ans)}


# ──────────────────────────────────────────────────────────────────────────────
# Hook setup — automatic, architecture-agnostic
# ──────────────────────────────────────────────────────────────────────────────

def discover_hook_targets(model):
    """Auto-discover all leaf submodules worth hooking from actual model structure.
    
    Skips container modules (ModuleList, Sequential, ModuleDict) that route
    internally but never receive direct input/output. Only hooks modules that
    actually transform data: Linear, Embedding, LayerNorm, RNN variants,
    and custom blocks with their own forward method.
    """
    pass  # removed unused parametrize import.nn.utils.parametrize as parametrize
    
    CONTAINERS = (nn.ModuleList, nn.Sequential, nn.ModuleDict, nn.ParameterList)
    
    targets = []
    skipped_small = []
    
    for name, mod in model.named_modules():
        # Skip the root module itself
        if name == '':
            continue
        
        # Skip container modules — they never receive direct I/O
        if isinstance(mod, CONTAINERS):
            continue
        
        keep = False
        if isinstance(mod, nn.Linear):
            keep = True
        elif isinstance(mod, nn.Embedding):
            keep = True
        elif isinstance(mod, nn.LayerNorm):
            keep = True
        elif isinstance(mod, (nn.GRU, nn.LSTM, nn.RNN)):
            keep = True
        elif hasattr(mod, 'forward'):
            # Custom block with its own computation (e.g., RWKVBlock)
            # Must have parameters to avoid hooking useless wrappers
            if sum(1 for _ in mod.parameters()) > 0:
                keep = True
        
        if not keep:
            continue
        
        # Skip trivially small modules
        if isinstance(mod, nn.Linear) and mod.out_features < 4:
            skipped_small.append(name)
            continue
        if isinstance(mod, nn.Embedding) and mod.num_embeddings < 5:
            skipped_small.append(name)
            continue
        
        targets.append(name)
    
    # Prune children of already-selected parents to avoid redundant hooks
    # e.g., if we hook blocks.0, don't also hook blocks.0.ln1 (unless useful)
    final = []
    for t in targets:
        dominated = False
        for other in targets:
            if t != other and t.startswith(other + '.'):
                dominated = True
                break
        if not dominated:
            final.append(t)
    
    if skipped_small:
        print(f"  Skipped small modules: {skipped_small[:5]}{'...' if len(skipped_small) > 5 else ''}")
    
    return final


# ──────────────────────────────────────────────────────────────────────────────
# Metrics registry
# ──────────────────────────────────────────────────────────────────────────────

VALID_METRICS = ['saturation', 'range_expansion', 'effective_dimensionality', 'predictability_loss']


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Bottleneck detection for progressive expansion")
    ap.add_argument('--checkpoint', required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument('--task', choices=['rwkv', 'world'], default=None,
                    help="Task type (auto-detected from checkpoint if omitted)")
    ap.add_argument('--easy-n', type=int, default=500, help="Number of easy examples")
    ap.add_argument('--hard-n', type=int, default=500, help="Number of hard examples")
    ap.add_argument('--metrics', nargs='+', choices=VALID_METRICS, default=None,
                    help="Metrics to compute (default: all forward-only)")
    ap.add_argument('--top-k', type=int, default=20, help="Top-N bottlenecks to display")
    ap.add_argument('--device', default='cpu', help="Device (cpu / cuda:0)")
    ap.add_argument('--output', default=None, help="Save full CSV to this path")
    ap.add_argument('--batch-size', type=int, default=32, help="Batch size for inference")

    args = ap.parse_args()

    device = torch.device(args.device)

    # ── Step 1: Load model & detect architecture ──
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get('model_state', ckpt)
    config = ckpt.get('config', {})
    
    try:
        arch = detect_architecture(sd)
    except ValueError as e:
        if args.task:
            arch = args.task
            print(f"WARN: Auto-detection failed ({e}), using --task={arch}")
        else:
            raise

    print(f"Architecture: {arch}")
    if config:
        print(f"Config: {json.dumps(config, indent=2)}")

    # Load model weights into correct architecture class
    if arch == 'rwkv':
        model = _load_rwkv(sd, device)
    elif arch == 'latent_think':
        model = _load_latent_think(sd, device)
    elif arch == 'baseline_ar':
        model = _load_baseline_ar(sd, device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Step 2: Build easy/hard data loaders ──
    if arch == 'rwkv':
        # LogicNiiahGenerator: fewer vars = easier, more vars = harder
        easy_pipeline = RwkvDataPipeline(seed=42)
        hard_pipeline = RwkvDataPipeline(seed=9999)
        
        easy_loader = hard_pipeline.build_dataloader(
            args.easy_n,
            {'num_vars': max(1, config.get('num_vars', 3) - 1),
             'min_transforms': max(1, config.get('min_transforms', 2) - 1),
             'max_transforms': max(1, config.get('max_transforms', 5) - 2),
             'noise_min': max(0, config.get('noise_min', 1) - 1),
             'noise_max': max(0, config.get('noise_max', 3) - 2),},
            batch_size=args.batch_size,
        )
        hard_loader = hard_pipeline.build_dataloader(
            args.hard_n,
            {'num_vars': config.get('num_vars', 3) + 1,
             'min_transforms': min(10, config.get('min_transforms', 2) + 1),
             'max_transforms': config.get('max_transforms', 5) + 2,
             'noise_min': config.get('noise_min', 1),
             'noise_max': config.get('noise_max', 3) + 2,},
            batch_size=args.batch_size,
        )
        
    elif arch in ('latent_think', 'baseline_ar'):
        pipeline = WorldModelPipeline(seed=42)
        easy_loader = pipeline.build_dataloader(
            args.easy_n, {'max_events': 4}, batch_size=16)
        hard_loader = pipeline.build_dataloader(
            args.hard_n, {'max_events': 10}, batch_size=16)
    else:
        raise NotImplementedError(f"Data pipeline for {arch} not implemented")

    # ── Step 3: Set up hooks on appropriate modules ──
    hook_targets = discover_hook_targets(model)
    print(f"Hook targets ({len(hook_targets)} modules): {hook_targets}")
    
    hook_mgr = HookManager(model)
    hook_mgr.register(hook_targets, post=True)

    # ── Step 4: Collect activations & compute metrics ──
    metrics_to_run = args.metrics or ['saturation', 'range_expansion', 'effective_dimensionality']
    
    labels_easy = labels_hard = None
    if 'predictability_loss' in metrics_to_run and arch in ('latent_think', 'baseline_ar'):
        # Extract answer labels for world-model
        from threads.memory_growth.dataset import build_vocab
        vocab_cats = build_vocab()[1]
        answer_map = {s: i for i, s in enumerate(vocab_cats['rel'])}
        
        easy_worlds = generate_dataset(n=args.easy_n, seed=42, max_events=4)
        easy_answers = torch.tensor([
            answer_map[q[1]] 
            for w in easy_worlds 
            for q in w['queries'][:4]
        ])
        
        hard_worlds = generate_dataset(n=args.hard_n, seed=9999, max_events=10)
        hard_answers = torch.tensor([
            answer_map[q[1]]
            for w in hard_worlds
            for q in w['queries'][:4]
        ])
        
        labels_easy = easy_answers.to(device)
        labels_hard = hard_answers.to(device)
    
    # Collect easy activations first (for calibration)
    print(f"\nCollecting easy activations ({args.easy_n} examples)...")
    easy_acts = hook_mgr.collect(easy_loader)
    print(f"  Captured {len(easy_acts)} layer groups: {list(easy_acts.keys())}")
    
    # Collect hard activations
    print(f"Collecting hard activations ({args.hard_n} examples)...")
    hard_acts = hook_mgr.collect(hard_loader)
    print(f"  Captured {len(hard_acts)} layer groups")

    # Compute metrics manually (don't need the full orchestrator since we already have acts)
    from threads.unsorted.bottleneck_analysis import (
        metric_saturation_ratio,
        metric_range_expansion,
        metric_effective_dimensionality,
        metric_predictability_loss,
    )
    
    results = []
    matching_layers = set(easy_acts.keys()) & set(hard_acts.keys())
    
    for layer_name in sorted(matching_layers):
        e_tensors = easy_acts[layer_name]
        h_tensors = hard_acts[layer_name]
        if len(e_tensors) != len(h_tensors) or len(e_tensors) == 0:
            continue
            
        c = e_tensors[0].shape[-1]  # channel dimension
        row_scores = {}
        
        if 'saturation' in metrics_to_run:
            score = metric_saturation_ratio(e_tensors, h_tensors)
            row_scores['saturation_score'] = score.tolist()
        
        if 'range_expansion' in metrics_to_run:
            score = metric_range_expansion(e_tensors, h_tensors)
            row_scores['range_expansion_score'] = score.tolist()
        
        if 'effective_dimensionality' in metrics_to_run:
            score = metric_effective_dimensionality(e_tensors, h_tensors)
            # EDD is layer-level (scalar); broadcast across all channels
            if score.ndim == 0:
                score = torch.full((c,), float(score))
            row_scores['effective_dimensionality_score'] = score.tolist()
        
        if 'predictability_loss' in metrics_to_run and labels_easy is not None:
            score = metric_predictability_loss(e_tensors, h_tensors, labels_easy, labels_hard)
            row_scores['predictability_loss_score'] = score.tolist()
        
        # Compile per-channel rows
        for ch_idx in range(c):
            row = {'layer': layer_name, 'channel': ch_idx}
            for metric_name, score_list in row_scores.items():
                row[metric_name] = float(score_list[ch_idx])
            
            # Aggregate score = mean of all metric scores for ranking
            vals = [v for k, v in row.items() if k.endswith('_score')]
            if vals:
                import numpy as np
                row['aggregate_score'] = float(np.nanmean(vals))
            else:
                row['aggregate_score'] = 0.0
            
            results.append(row)

    # Sort by aggregate score descending
    results.sort(key=lambda r: r['aggregate_score'], reverse=True)

    # ── Step 5: Output results ──
    top = min(args.top_k, len(results))
    print(f"\n{'='*80}")
    print(f"BOTTLENECK ANALYSIS RESULTS — {arch.upper()} model")
    print(f"Top {top} candidate bottleneck channels ({len(results)} total scored):")
    print(f"{'='*80}")
    
    col_names = ['Layer', 'Ch', 'Aggregate', 'Saturation', 'Range Exp', 'Eff Dim']
    col_keys = ['layer', 'channel', 'aggregate_score', 'saturation_score', 'range_expansion_score', 'effective_dimensionality_score']
    
    header = f"{'Layer':<25} {'Ch':>4} {'Agg':>10} {'Sat':>10} {'Range':>10} {'EffDim':>10}"
    print(header)
    print('-' * 75)
    
    for row in results[:top]:
        vals = []
        for key in col_keys[2:]:
            vals.append(row.get(key, float('nan')))
        
        print(
            f"{row['layer']:<25} "
            f"{row['channel']:>4} "
            f"{vals[0]:>10.4f} "
            f"{vals[1]:>10.4f} "
            f"{vals[2]:>10.4f} "
            f"{vals[3] if len(vals) > 3 else '':>10}"
        )

    # Save full CSV
    if args.output:
        Path(args.output).write_text(format_csv(results))
        print(f"\nFull CSV saved to: {args.output} ({len(results)} rows)")
    
    print(f"\nDone. {len(results)} channel-layer pairs scored across {len(matching_layers)} layers.")
    return results


if __name__ == '__main__':
    main()
