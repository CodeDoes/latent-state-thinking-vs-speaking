# BLT Architecture Progression

Four-step progression testing byte+patch dual-stream architectures with different combinations of transformer vs RWKV blocks.

## Goal

Test which component (byte-level encoder/decoder vs patch-level mixer) benefits more from RWKV recurrence vs transformer attention at small scale.

## Experiments

All trained on 99KB TinyStories slice, 300 steps, batch=8, max_len=128, CPU.

### Step 1: Pure BLT (all transformer)

**Architecture:**
- Byte encoder: CausalTransformerBlock × 2
- Patch mixer: CausalTransformerBlock × 1
- Byte decoder: LayerNorm + Linear

**Results:**
- Params: 150,434
- Final loss: 0.94 (avg 1.12)
- Training speed: ~32K bytes/s
- Generation: Repetitive ("she she she")

**Commit:** `bbdf1ad`

### Step 2: RWKV byte encoder/decoder + transformer patch mixer

**Architecture:**
- Byte encoder: RWKVBlock × 2
- Patch mixer: CausalTransformerBlock × 1
- Byte decoder: LayerNorm + Linear

**Results:**
- Params: 158,370
- Final loss: 0.55 (avg 0.69)
- Training speed: ~36K bytes/s
- Generation: Better word-level structure

**Commit:** `be9de16`

**Observation:** RWKV byte encoder/decoder significantly outperforms transformer (0.55 vs 0.94). RWKV's recurrence is more efficient at byte-level modeling at this scale.

### Step 3: Transformer byte encoder/decoder + RWKV patch mixer

**Architecture:**
- Byte encoder: CausalTransformerBlock × 2
- Patch mixer: RWKVBlock × 1
- Byte decoder: LayerNorm + Linear

**Results:**
- Params: 151,394
- Final loss: 0.83 (avg 0.97)
- Training speed: ~43K bytes/s
- Generation: Mixed quality

**Commit:** `08c643c`

**Observation:** RWKV patch mixer (0.83) outperforms transformer patch mixer (step 1: 0.94), but RWKV byte encoder/decoder (step 2: 0.55) still wins. RWKV recurrence is more effective at byte-level modeling than patch-level.

### Step 4: Pure RNN BLT (all RWKV)

**Architecture:**
- Byte encoder: RWKVBlock × 2
- Patch mixer: RWKVBlock × 1
- Byte decoder: LayerNorm + Linear

**Results:**
- Params: 159,330
- Final loss: 1.73
- Training speed: ~32K bytes/s
- Generation: Highly repetitive ("was was was")

**Experiment:** `blt_rwkv_001` (not committed — used existing `blt_rwkv.py`)

**Observation:** All-RWKV performs worst (1.73). The hybrid (RWKV byte + transformer patch) is best (0.55).

## Summary

| Step | Byte enc/dec | Patch mixer | Final loss | Params |
|------|--------------|-------------|------------|--------|
| 1    | Transformer  | Transformer | 0.94       | 150K   |
| 2    | RWKV         | Transformer | **0.55**   | 158K   |
| 3    | Transformer  | RWKV        | 0.83       | 151K   |
| 4    | RWKV         | RWKV        | 1.73       | 159K   |

## Key findings

1. **RWKV byte encoder/decoder wins**: Step 2 (RWKV byte + transformer patch) achieves 0.55, far better than all-transformer (0.94) or all-RWKV (1.73).

2. **RWKV patch mixer helps modestly**: Step 3 (transformer byte + RWKV patch) is 0.83, better than all-transformer (0.94) but not as good as RWKV byte.

3. **All-RWKV fails**: Step 4 (all RWKV) is the worst at 1.73. The transformer patch mixer is important.

4. **Hybrid is best**: RWKV for byte-level (fine-grained temporal structure) + transformer for patch-level (coarse-grained attention over patches) is the winning combination.

## Interpretation

RWKV's linear recurrence is well-suited for byte-level modeling: each byte depends on the previous byte in a sequential, temporal way. The recurrence accumulates state efficiently without the quadratic cost of attention.

Transformer attention is better for patch-level processing: patches are coarser, and attention can selectively focus on relevant patches without being constrained by sequential order.

The all-RWKV failure suggests that the patch-level stream needs the global attention mechanism to integrate information across patches. RWKV's sequential processing is too local for this.

## Next steps

- Longer training (1000+ steps) to see if the ranking holds
- Larger scale (dim=128, more layers)
- Dynamic patch boundaries (entropy-based patcher) instead of fixed windows
- Test on harder tasks (not just TinyStories next-token prediction)
