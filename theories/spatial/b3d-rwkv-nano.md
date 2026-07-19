# b3d-rwkv-nano (Triplet-Block Diffusion RWKV, small)

Paper: https://arxiv.org/abs/2605.25969 / HF https://huggingface.co/leonardklin/B3D-RWKV

B3D-RWKV 7.2B is diffusion RWKV that gets O(L) inference + parallel bidirectional diffusion via **triplet-block layout**:

Each logical block of size B (e.g., 32 tokens) appears three times left-to-right as physical blocks:
  b1(i) = masked copy (e.g., 50% masks)
  b2(i) = identical masked copy (loss computed here)
  b3(i) = clean ground-truth copy (refreshes RWKV recurrent state for block i+1)

Because backbone reads left→right, hidden state arriving at any masked position of b2 has already seen every *unmasked* token of b1. So b2 gets **pseudo-bidirectional** access while staying strictly causal. No backbone change.

Inference per-block iterative denoising:
  while masked positions remain in block:
    run RWKV over [c_context + b1 + b2]
    commit every position where top-1 prob > τ (e.g., 0.9)
    -> unmask those positions in b1 and b2 for next iteration
  once block fully committed, append clean b3 to context, start next logical block.

Hypothesis (nano): triplet-block layout works at **228K–1M params** on byte-level or char-level tasks, enabling parallel decode (1.6× speedup observed at 7.2B) at small scale without loss vs causal AR baseline.

Why smaller matters: paper at 7.2B proves scale, but does not prove mechanism is sufficient at minimal scale. If mechanism works at nano, it is architectural, not emergent. If fails at nano, speedup is scale artifact.

Minimal test (single variable = layout, matched params/compute):
- Same RWKVNano dim=128, 3 layers, vocab 258 bytes (or CHAR vocab 70 from train_rwkv.py)
- Arm A: causal AR baseline (standard next-token CE, sequential decoding)
- Arm B: triplet-block diffusion (B=8 bytes logical, mask rate 0.3, τ=0.9 at inference, same total tokens seen during training because triplet expands length 3x per block – to match compute, reduce steps so total bytes processed equal)
- Train both 2k steps on byte_ts_001 or logic_niiah Generator
- Metric: (1) recon/CE loss at same total bytes seen, (2) throughput: avg steps to decode 128-byte block (AR needs 128 forward passes, diffusion needs ~iter*3 forward? Paper reports 1.6×). At nano, win if B loss ≤ A +0.1 AND B throughput ≥1.2× A (fewer iterative steps than sequential).

Single variable: training layout (triplet diffusion vs causal AR). Params locked.

This would be first nano-scale replication of B3D-RWKV architectural trick, proving mechanism not scale-dependent.

---

**Research links:** [`research/diffusion_rnn_hybrids.md`](../research/diffusion_rnn_hybrids.md) — B3D-RWKV paper, DREAMSTATE, Recurrent Autoregressive Diffusion, RDM. See also [`research/rwkv_overview.md`](../research/rwkv_overview.md) for the RWKV foundation.
