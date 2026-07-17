# delta-mem.infer.md

> **Source**: `delta-mem.md` (verbatim).  
> **Date**: 2025-07-17

---

## What this theory is doing

A **stateful-attention side-channel experiment**. Two adapters live inside standard Causal Self-Attention:

1. **DeltaRuleStateMemory** — associative online state with keep-erase-write timing. Low-rank memory vectors per head.
2. **RWKVStateMemory** — uses WKV recurrence inside attention to inject RWKV-style state.

Both project state-readout back to **delta-q, delta-k, delta-v, delta-o** (low-rank corrections to standard attention Q/K/V/O).

## Why "adapters inside transformer" instead of "swap the block"?

The .md explicitly cites the motivation: **swapping** RWKV/recurrent for Transformer blocks causes *distribution shift* and downstream degradation (e.g., MMLU). **Adapters** are a smaller intervention: keep the transformer backbone, add stateful side-channels that introduce *some* recurrence without losing the transformer's downstream skill profile.

This is the bridge concept between:
- Large-scale Transformer weights (already trained)
- Stateful "thinking" (the goal)

## Connection to the project's thinking-vs-speaking framework

| Concept | Maps to |
|---|---|
| Prefill / ingestion | **Amortized thinking** (builds memory state) |
| Delta-Rule $S_t$ update | **Speaking-side state** (per-step rule application) |
| Read-before-write | **Pen-before-ink** (state sees prior token before current) |
| Delta to Q/K/V/O | **Latent influence on speak** |

The state is not a separate decoder — it's a *modifier of attention*.

## What's proven vs. open

| Claim | Status |
|---|---|
| Both adapters train stably | **proven** (loss 4.28→3.24, 4.42→3.23 — 10 steps) |
| Gradient flows back to all memory params | **proven** (full gradient check passed) |
| Adapter contribution is *qualitatively* different from baseline | **open** — need ablation: attention WITH vs WITHOUT adapter on same steps |
| Adapter makes downstream tasks (MMLU-style) *better*, not just *lossier* | **open** — task-agnostic not yet measured |
| Adapter survives *long-context* regimes without O(n²) cost | **open** — runtime scaling unmeasured |

## Latent assumptions the .md leaves implicit

- The "adapters inside attention" approach is a **library**, not a single adapter. Different adapter types should be pluggable.
- "Keep-erase-write timing" implies read-before-write, but the write-erase order is implementation-specific.
- RWKVStateMemory is not the same as **the** RWKV — it's a *tiny* WKV recurrence grafted into attention. This naming overlap is a trap.
- Low-rank corrections (delta-q etc.) work because attention is approximate; if attention was exact this wouldn't decompose cleanly.

## Status

Implementation done. Partial training proven. **No measured claim** that the stateful adapter *helps* over baseline. Need:

1. Run baseline `transformer_attention` for the same steps, same params — confirm delta-mem > baseline on loss.
2. Distill: once trained, can the state be transferred to a second model?
3. Latency / memory profile on a long context — does the adapter keep O(n)?

Recommend **ablation #1 first** (cheapest, most informative). Until then this is "code that trains, may or may not earn its keep."
