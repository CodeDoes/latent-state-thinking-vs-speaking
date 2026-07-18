# delta-mem

A clean integration of **Delta-Rule State Memory** and **RWKV-State Memory** as associative memory adapters inside Transformer Attention blocks.

## Philosophy & Background

This experiment is based on the idea from [xiaol/HRM-RWKV-Text](https://github.com/xiaol/HRM-RWKV-Text), which introduces online stateful adapters (e.g. delta-rule associative memory) inside causal attention layers. Rather than fully swapping Transformer blocks with Recurrent blocks (which has been shown to cause massive distribution shifts and degrade performance on downstream reasoning benchmarks like MMLU), this approach adds small trainable recurrent side-channels.

In the context of our **thinking vs speaking** framework:
- Amortized thinking is the core. The model builds a memory state of the sequence context during prefill / ingestion.
- The **Delta-Rule Memory Adapter** operates as a stateful associative memory where:
  - It projects sequence features to low-rank memory queries, keys, and values.
  - It updates an online associative state tensor $S_t$ with a keep-erase-write sequence using the delta-rule.
  - It reads the current context from the associative state before writing the current token (read-before-write timing).
  - The readout is projected to delta-$q$, delta-$k$, delta-$v$, and delta-$o$, which are added to the attention projections.

## Implementation Details

We implemented two primary adapters inside `src/delta_mem.py`:
1. **DeltaRuleStateMemory**:
   - Computes low-rank memory vectors and updates a fast online associative state $S \in \mathbb{R}^{\text{rank} \times \mathbb{R}^{\text{rank}}}$ per head.
   - Leverages a Numerically stable PyTorch scanning sequence.
   - Injects low-rank delta corrections to Attention Q/K/V/O.
2. **RWKVStateMemory**:
   - Uses an RWKV-4/7 mixing and WKV recurrence mechanism in PyTorch to compute state reads.
   - Employs a read-before-write shift so the current step's read depends exclusively on the previous token state.

Both mechanisms are integrated directly into our custom `CausalSelfAttentionWithMemory` module in `src/transformer_with_memory.py`.

## Proof & Validation Status

We ran controlled, matched-parameter validation experiments on the `logic_niiah` task:
- Baseline standard transformer trains and establishes the task learning trend.
- `delta_rule` trains stably, with loss reducing from `4.28` to `3.24` in 10 steps.
- `rwkv_state` trains stably, with loss reducing from `4.42` to `3.23` in 10 steps.
- Full gradient checking confirms that gradients successfully flow from the next-token prediction loss all the way back to all memory parameters, including gates, keys, values, and projection parameters.
