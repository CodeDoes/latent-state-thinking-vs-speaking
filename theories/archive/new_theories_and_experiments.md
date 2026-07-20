# New Theories & Ready-to-Go Experiments
**Date:** 2026-07-17 | **Repo:** CodeDoes/latent-state-thinking-vs-speaking | **Base commit:** B5 `adaptive_loop_001` + bug fixes `rwkv_state_passing_001` + `shared_state_unrolled_feedback_001`

You asked for **more theories + more experiments ready to go**. Here's a batch of **5 single-variable ablations** derived directly from `byte-state-byte.status.md` open follow-ups #2, #5, #6 and from `reports/user_theory_mapping.md`.

Each theory follows `ultimate.md` frame: smallest viable system, one hypothesis, matched params, observable metrics, fast CPU runnable.

---

## Summary Table

| # | Theory | Single Variable | Hypothesis | Train Script | Smoke Test | Full Exp IDs |
|---|--------|----------------|------------|--------------|------------|--------------|
| I | **injection-frequency** | core→decoder fusion: front-only vs per-layer | Per-layer fusion > front fusion, lower recon loss by ≥0.05 | `src/train_injection_freq.py` | ✅ 50 steps loss 5.75→4.40 front, 5.66→4.59 per-layer | `inj_freq_front_001` vs `inj_freq_perlayer_001` |
| D | **dynamic-patch-vs-fixed** | patch boundary: fixed stride 4 vs surprise threshold | Dynamic maintains loss at better compression rho | `src/train_dynamic_patch.py` | ✅ fixed 5.51→4.33, dynamic 5.60→4.36 (rho 4 vs 2) | `dyn_patch_fixed_001` vs `dyn_patch_dynamic_07_001` |
| T | **token-vs-byte-head** | decoder head vocab: 258 bytes vs 1024 tokens (sparse) | Byte dense supervision > token sparse at 228K | `src/train_token_byte.py` | ✅ byte 5.79→4.35, token 7.08→6.53 | `tok_byte_bytehead_001` vs `tok_byte_tokenhead_001` |
| S | **rwkv-state-carry** | RWKV initial state: zero vs stateful carry vs learned | Stateful carry improves long-horizon (noise_max=10) accuracy +0.1 | `src/train_rwkv_carry.py` | ✅ zero mode loss 4.30→3.96 | `rwkv_carry_zero_001` vs `rwkv_carry_stateful_001` vs `learned_001` |
| E | **adaptive-exit-entropy** | entropy weight: 0.0 vs 0.01 vs 0.1 | 0.01 prevents loop collapse, enables enc 1→3 adaptation | `src/train_adaptive_entropy.py` | ✅ 0.01 loss 5.82→4.54, lam_var 0.013→0.004 | `adapt_ent_0.0_001`, `0.01_001`, `0.1_001` or `--sweep` |

All at **~66K params for dim=32 smoke**, **~228K params for dim=64 full** (B5 scale). 2k steps full, 200 steps smoke, CPURunnable.

---

## Theory Details (files)

### I) injection-frequency
- **Files:** `theories/injection-frequency.md` + `.infer.md` + `.status.md`
- **Claim I1:** per-layer fusion of core state into decoder improves recon loss over front-only fusion at matched params.
- **Mechanism:** front fusion adds `enc+core` once. Per-layer re-injects `core_broadcast` before each decoder layer with gated `sigmoid(W·[h,core])`. If global state overwritten by later layers, per-layer forces survival.
- **Experiment:** `src/injection_freq_model.py` = `ByteEncoder` + `LoopedRWKV7Core` + `PerLayerFusionDecoder`. Two modes: `front`, `per_layer`, `per_layer_nogate`.
- **Run:**
  ```bash
  python -m src.train_injection_freq --fusion_mode front --exp_id inj_freq_front_001 --steps 2000 --dim 64
  python -m src.train_injection_freq --fusion_mode per_layer --exp_id inj_freq_perlayer_001 --steps 2000 --dim 64
  ```
- **Metrics:** recon loss, gate mean/var, enc/core/dec loops, rho, samples. Win if per_layer loss < front -0.05 and gate mean not zero (model uses injection).

### D) dynamic-patch-vs-fixed
- **Files:** `theories/dynamic-patch-vs-fixed.md` + `.infer.md` + `.status.md`
- **Claim D1:** surprise-based dynamic patching maintains loss at matched compression rho~4.
- **Mechanism:** `surprise_per_step(h)=|h(t)-h(t-1)| mean`. Boundary if >threshold. Measured earlier: normal letters 1.0-1.5, double-letters 0.1-0.2, punctuation 1.8+ → natural word boundaries get short patches.
- **Experiment:** `src/train_dynamic_patch.py` wraps existing `AdaptiveLoopModel` with `dynamic_patch` flag. Fixed `patch_size=4` vs dynamic `threshold 0.5/0.7/0.9`.
- **Run:**
  ```bash
  python -m src.train_dynamic_patch --patch_mode fixed --exp_id dyn_patch_fixed_001 --steps 2000
  python -m src.train_dynamic_patch --patch_mode dynamic --threshold 0.7 --exp_id dyn_patch_dynamic_07_001 --steps 2000
  python -m src.train_dynamic_patch --patch_mode dynamic --threshold 0.5 --exp_id dyn_patch_dynamic_05_001 --steps 2000
  ```
- **Metrics:** loss, rho, mean_patch_len histogram, std (instability detector), loops.

### T) token-vs-byte-head
- **Files:** `theories/token-vs-byte-head.md` + `.infer.md` + `.status.md`
- **Claim T1:** byte head (258) beats token head (1k) at matched 228K because byte gives dense gradient every position.
- **Mechanism:** token = hashed 4-byte n-gram bucket (simulates BPE sparsity). Byte model CE every byte; token model CE every 4 bytes → 4x sparser signal. This is the one-variable delta from B5 to user's proposed loop in `reports/user_theory_mapping.md`.
- **Experiment:** `src/token_byte_head_model.py` = base `AdaptiveLoopModel` with head swapped to `Linear(dim,1024)`.
- **Run:**
  ```bash
  python -m src.train_token_byte --vocab_mode byte --exp_id tok_byte_bytehead_001 --steps 2000
  python -m src.train_token_byte --vocab_mode token --bytes_per_token 4 --exp_id tok_byte_tokenhead_001 --steps 2000
  ```
- **Metrics:** recon loss (token CE vs byte CE trend), detokenized samples, rho.

### S) rwkv-state-carry
- **Files:** `theories/rwkv-state-carry.md` + `.infer.md` + `.status.md`
- **Claim S1:** stateful carry across segments improves long-horizon WHERE accuracy over zero init. Validates `rwkv_state_passing_001` fix usefulness.
- **Mechanism:** RWKVBlock now correctly saves `num, den, xx, xx2` with proper decay. Carrying them detached across batches (TBPTT) preserves >10k token memory. Zero init resets to zeros each batch.
- **Experiment:** `src/train_rwkv_carry.py` = `RWKVNano` on `LogicNiiahGenerator` with `noise_max=10` (needs memory 200 tokens). Modes: `zero`, `stateful`, `learned` (learnable init params).
- **Run:**
  ```bash
  python -m src.train_rwkv_carry --mode zero --exp_id rwkv_carry_zero_001 --steps 2000 --max_len 256 --noise_max 10
  python -m src.train_rwkv_carry --mode stateful --exp_id rwkv_carry_stateful_001 --steps 2000 --max_len 256 --noise_max 10
  python -m src.train_rwkv_carry --mode learned --exp_id rwkv_carry_learned_001 --steps 2000 --max_len 256 --noise_max 10
  ```
- **Metrics:** exact accuracy, digit_accuracy, mean decay w_ = exp(-exp(time_decay)) (long-memory channels).

### E) adaptive-exit-entropy
- **Files:** `theories/adaptive-exit-entropy.md` + `.infer.md` + `.status.md`
- **Claim E1:** entropy weight 0.01 prevents collapse to 1 loop, enables encoder 1→3 adaptation observed in B5.
- **Mechanism:** `AdaptiveExitGate` outputs λ_r, π_r = λ_r prod(1-λ). Loss = Σ π_r L_r - entropy_weight*H. No entropy → λ_1→1 early (never explores deeper loops). Too high entropy → never converges.
- **Experiment:** `src/train_adaptive_entropy.py` = same `AdaptiveLoopModel` with sweep `0.0,0.001,0.01,0.05,0.1`.
- **Run:**
  ```bash
  python -m src.train_adaptive_entropy --entropy_weight 0.01 --exp_id adapt_ent_0.01_001 --steps 2000
  python -m src.train_adaptive_entropy --sweep --exp_id_prefix adapt_ent --steps 2000  # runs all 5 weights, writes sweep.json
  ```
- **Metrics:** loss, enc_loops distribution, dec_loops, lambda mean/var per depth, correlation surprise vs loops.

---

## How to Run Everything

### Smoke (2 mins per exp, CPU)
```bash
# install torch cpu if not present
pip install torch --index-url https://download.pytorch.org/whl/cpu

# one-off smoke tests (50 steps each, already verified)
python -m src.train_injection_freq --fusion_mode front --exp_id inj_freq_front_smoke_test --steps 50 --dim 32
python -m src.train_dynamic_patch --patch_mode fixed --exp_id dyn_patch_fixed_smoke_test --steps 50 --dim 32

# OR run full batch via runner
python -m src.run_new_batch --smoke
# writes experiments/new_batch_summary.json
```

### Full (matches theory spec, 228K params, 2k steps, ~10 mins each on CPU, ~2 mins on GPU)
```bash
python -m src.run_new_batch --full
```

### Individual full runs
```bash
# I
python -m src.train_injection_freq --fusion_mode front --exp_id inj_freq_front_001 --steps 2000 --dim 64
python -m src.train_injection_freq --fusion_mode per_layer --exp_id inj_freq_perlayer_001 --steps 2000 --dim 64

# D
python -m src.train_dynamic_patch --patch_mode fixed --exp_id dyn_patch_fixed_001 --steps 2000
python -m src.train_dynamic_patch --patch_mode dynamic --threshold 0.7 --exp_id dyn_patch_dynamic_07_001 --steps 2000

# T
python -m src.train_token_byte --vocab_mode byte --exp_id tok_byte_bytehead_001 --steps 2000
python -m src.train_token_byte --vocab_mode token --exp_id tok_byte_tokenhead_001 --steps 2000

# S
python -m src.train_rwkv_carry --mode zero --exp_id rwkv_carry_zero_001 --steps 2000 --max_len 256 --noise_max 10
python -m src.train_rwkv_carry --mode stateful --exp_id rwkv_carry_stateful_001 --steps 2000 --max_len 256 --noise_max 10

# E
python -m src.train_adaptive_entropy --sweep --steps 2000
```

All experiments log to `experiments/<exp_id>/config.json`, `metrics.jsonl`, `metrics.json`, `sample.txt` per AGENTS.md convention.

---

## What This Unlocks Next

These 5 are cheap-to-expensive ordered, all prove one thing. After they run:

1. **If per-layer wins (I1):** Scale to 1M params, test attention-style injection (Q=decoder, KV=core)
2. **If dynamic wins (D1):** Implement learned entropy model (small RWKV scorer) instead of parameter-free surprise delta, and warmup 200 steps fixed→dynamic switch (D3)
3. **If byte beats token (T1):** Confirms dense supervision needed at nano scale; try hybrid: byte training, token inference distillation
4. **If carry wins (S1):** Apply stateful carry to `AdaptiveLoopModel` core (currently reset each batch) – enables true byte-loop closure
5. **If entropy 0.01 is not optimal (E1):** Use per-layer learned entropy weight, or Gumbel-softmax sampling vs expected loss

Each follow-up is already listed in respective `.status.md` under "Open follow-ups".

---

## File Map

New files added in this batch:
```
theories/injection-frequency.md / .infer.md / .status.md
theories/dynamic-patch-vs-fixed.md / .infer.md / .status.md
theories/token-vs-byte-head.md / .infer.md / .status.md
theories/rwkv-state-carry.md / .infer.md / .status.md
theories/adaptive-exit-entropy.md / .infer.md / .status.md
src/injection_freq_model.py
src/train_injection_freq.py
src/train_dynamic_patch.py
src/token_byte_head_model.py
src/train_token_byte.py
src/train_rwkv_carry.py
src/train_adaptive_entropy.py
src/run_new_batch.py
theories/archive/new_theories_and_experiments.md
```

Updated:
```
theories/status.md – lists new threads as active ready-to-run
```

---

## Proof Ledger (unchanged, but new slots reserved)

Current proven in `theories/proofs.md`:
- exp001, encoder_state_ablation_001 (B1), rnn_patch_002 (B2 neg), shared_state_unrolled_010 (B3), shared_state_unrolled_shared_010 (B4 neg), adaptive_loop_001 (B5), shared_state_unrolled_feedback_001 (fixes B3/B4), rwkv_state_passing_001

New batch will add when runs complete:
- inj_freq_front_001 / perlayer_001 → I1, I2, I3
- dyn_patch_fixed_001 / dynamic_07_001 → D1, D2, D3
- tok_byte_bytehead_001 / tokenhead_001 → T1, T2, T3
- rwkv_carry_* → S1, S2, S3
- adapt_ent_* → E1, E2, E3

---

## Smoke Verification (already run 2026-07-17)

All smoke tests passed in this environment (torch 2.13.0+cpu):
- inj_freq_front_smoke_test: 50 steps 5.75→4.40 loss, params 66K
- inj_freq_perlayer_smoke_test: 5.66→4.59, params 70K (per-layer gate adds ~4K)
- dyn_patch_fixed_smoke_test: 5.51→4.33 rho 4.0
- dyn_patch_dynamic_smoke_test: 5.60→4.36 rho ~2 (need threshold tuning for rho match)
- tok_byte_byte_smoke_test: 5.79→4.35
- tok_byte_token_smoke_test: 7.08→6.53 (sparser, higher loss as hypothesized)
- adapt_ent_smoke_test: 5.82→4.54 lam_var 0.013→0.004
- rwkv_carry_zero_smoke_test: loss 4.30→3.96 (accuracy 0 at 100 steps expected)

These are in `experiments/*_smoke_test/` on workspace.

---

## TL;DR for Agent Instructions

If you're an agent (Jules, Claude, etc) picking up this repo:

1. Read `theories/status.md` → new threads section → pick one `.status.md`
2. Take the first **open** claim (e.g., I1)
3. Run its train script at full scale (2000 steps, dim 64)
4. Log results to `experiments/<exp_id>/` + update `theories/proofs.md` with one line if proven
5. Move to next claim.

No multi-variable changes. One experiment per commit. Git hash recorded in config.

