> **Archived 2026-07-20:** source dump; distilled into [`../spatial/b3d-rwkv-nano.md`](../spatial/b3d-rwkv-nano.md) and [`../spatial/diffusion-grid-terminal.md`](../spatial/diffusion-grid-terminal.md).

# B3D-RWKV Nano + Diffusion Grid Terminal Theories
**Date:** 2026-07-17 | **User request:** https://huggingface.co/leonardklin/B3D-RWKV but smaller + grid diffusion terminal with certainty typing

---

## 1) B3D-RWKV Nano — Triplet-Block Diffusion at Small Scale

**Source:** Paper Triplet-Block Diffusion RWKV (2605.25969) – 7.2B param model, HF `leonardklin/B3D-RWKV`
**Paper core idea (from arXiv HTML):**
> Each logical generation block of size B appears three times consecutively: masked copy b1, identical masked copy b2 on which denoising loss computed, clean ground-truth copy b3 that refreshes recurrent state before next block. Because backbone reads left-to-right, hidden state arriving at any masked position of b2 has already absorbed every unmasked token of b1, so b2 gains pseudo-bidirectional access on strictly causal model.

Inference: per-block iterative denoising – commit every position where top-1 prob > τ, loop until fully committed, append clean block to context, next block.

**Nano hypothesis:** Mechanism works at 128-dim / 3 layers / 228K params, not just 7.2B. If true, 1.6× throughput speedup is architectural, not emergent.

**Files:**
- `theories/b3d-rwkv-nano.md` – theory (verbatim)
- `theories/b3d-rwkv-nano.infer.md` – interpretation (why b3 needed, compute matching trick, tau sweep, relation to state-carry)
- `theories/b3d-rwkv-nano.status.md` – proof chain BD1-BD4
- `src/b3d_rwkv_model.py` – `B3DRWKVModel` wraps `RWKVNano` + `build_triplet_batch()` + `diffusion_decode_block()`
- `src/train_b3d_rwkv.py` – train modes `triplet`, `ar`, `no_b3`

**BD claims:**
- **BD1**: triplet trains at 228K without loss vs AR at matched physical tokens (CE ≤ AR+0.1)
- **BD2**: parallel decode – avg iters per block < block_size (e.g., 8) at τ=0.9 → speedup
- **BD3**: clean copy b3 load-bearing – no-b3 variant loss +0.2 or compounding errors
- **BD4**: pseudo-bidirectional via b1→b2 carry – b2 accuracy > b1 +0.2 on synthetic future-dependent task

**Run:**
```bash
# AR baseline
python -m src.train_b3d_rwkv --mode ar --exp_id b3d_ar_001 --steps 2000 --dim 128 --max_len 64 --block_size 8 --vocab_mode char

# Triplet diffusion (same params, same physical tokens)
python -m src.train_b3d_rwkv --mode triplet --exp_id b3d_triplet_001 --steps 2000 --dim 128 --max_len 64 --block_size 8 --mask_ratio 0.3 --vocab_mode char

# Ablate b3 clean refresh
python -m src.train_b3d_rwkv --mode no_b3 --exp_id b3d_no_b3_001 --steps 2000 --dim 128

# Smoke (already verified 50 steps loss 4.31→4.26)
python -m src.train_b3d_rwkv --mode triplet --exp_id b3d_triplet_smoke_test --steps 50 --dim 32 --max_len 32 --block_size 8 --log_every 10
```

**Throughput eval included:** after training, code automatically sweeps τ=0.8/0.9/0.95 and logs `tau_sweep.json` with iters per block and commit history. Paper reports 1.6× avg; nano should show ≥1.2× if mechanism holds.

**Why smaller matters:** At 7.2B, dim=4096 state can remember many unmasked tokens from b1. At nano dim=128, state tiny – can it still store 8 tokens? Tests `rwkv-state-carry` theory. If fails, need smaller B or larger dim – still informative negative.

---

## 2) Diffusion Grid Terminal — Screen → Screen-RWKV → Screen with Certainty Trigger

**User theory verbatim:** "RWKV with diffusion and takes a grid of bytes as input and outputs a grid of bytes stochastically, triggers typing when full certainty. Theory is would improve terminal use. would allow for vision approximation. would allow recurrent understanding of the world. screen -> screen-rwkv -> screen -> trigger on full certainty of each byte or repeat diffusion step. Other theory is that it can learn to use the output as reasoning traces and also that it can learn a form of tool calling and also that it will have temporal awareness."

**Architecture (nano):**
- Input grid `G_t` H×W bytes (16×32=512 bytes for nano, 24×80=1920 real). Flatten row-major, keep `\n` or add learned row+col embeddings (implemented `row_embed` + `col_embed` in `GridRWKVModel`).
- Model: `B3DRWKVModel` substrate but applied to whole grid as one logical block (or per-row blocks). Training: mask 30% cells, build triplet [masked b1, masked b2 lossable, clean b3], CE loss on b2 masked positions.
- Inference diffusion: start with partially known screen (prompt `$ ` known, rest masked). Run RWKV over [b1,b2], get logits per cell, commit cells where p(top1)>τ (certainty). Unmask committed, repeat until fully committed or max iters. Committed cells = typing triggered. Next screen observed from terminal env → state carry.
- Recurrent state `S_t` carries across screens: `screen_t + S_t -> screen_{t+1} + S_{t+1}` – temporal awareness.

**Why this improves terminal:**
- Current LLM terminal: token-by-token generation, no vision of own rendering, errors persist.
- Grid diffusion: sees full screen (including own previous typing, borders, colors via ANSI bytes), self-corrects low-certainty cells in next diffusion iter (parallel self-correction, not sequential).
- Vision approx: screen rendering *is* byte grid – box-drawing `+-|`, ANSI colors, file list columns. Learning to predict those correctly proves 2D spatial understanding without vision encoder.
- Typing trigger at full certainty prevents spamming low-conf commands – safety.

**Three sub-theories included:**

### 2a) Reasoning traces as grid scratchpad
Trace rows 10-13 reserved. Model can write intermediate results there (e.g., running total). Rows persist across diffusion steps (grid memory) vs linear CoT discarded. Test: task needs remembering intermediate (sum list). With trace vs trace erased each step. Win if with-trace +0.15 accuracy.

### 2b) Tool calling as certainty-triggered typing
Screen shows `$` prompt, file list. Tool call = typing `cat file.txt\n` committed atomically when all chars in command row >τ. Tool output appears as next screen. State carry learns tool loop as policy. Test: synthetic chain – screen1 file list, screen2 after cat shows SECRET, screen3 asks SECRET. Needs tool use. Measure error rate vs greedy (τ=0 greedy commits partial commands).

### 2c) Temporal awareness
`S_t` carries across screens, so model can answer "what was screen_{t-2}?" or detect file change. Probe decoder to reconstruct previous screen from current state. Zero-init baseline can't.

**Files:**
- `theories/diffusion-grid-terminal.md` – full theory
- `theories/diffusion-grid-terminal.infer.md` – why byte grid not tokens, 2D pos need, certainty=adaptive exit, vision test, relation to B3/ feedback fix
- `theories/diffusion-grid-terminal.status.md` – GT1-GT6 claims
- `src/diffusion_grid_model.py` – `GridRWKVModel` with 2D row/col embed, `diffusion_decode_grid()`, synthetic generators: `generate_grid_random`, `generate_grid_box` (vision box test), `generate_terminal_chain` (3-step tool chain), `generate_grid_with_trace`
- `src/train_diffusion_grid.py` – modes `recon`, `chain`, `trace`, `certainty`

**GT claims:**
- **GT1** grid diffusion reconstructs masked 16×16 vs row-by-row AR, fewer forward passes
- **GT2** stateful carry across 3-step terminal chain enables memory (SECRET task)
- **GT3** trace rows load-bearing for multi-step calc
- **GT4** certainty τ=0.9 reduces tool-calling errors vs greedy
- **GT5** temporal probe reconstructs screen_{t-2} from state
- **GT6** vision box: model learns 2D box border alignment (needs spatial)

**Run (smoke verified):**
```bash
# GT1 random grid recon
python -m src.train_diffusion_grid --mode recon --grid_type random --exp_id grid_recon_random_001 --steps 2000 --H 16 --W 32 --dim 64

# GT6 vision box (2D structure)
python -m src.train_diffusion_grid --mode recon --grid_type box --exp_id grid_vision_box_001 --steps 2000 --H 16 --W 32 --dim 64

# GT2 stateful chain (SECRET memory)
python -m src.train_diffusion_grid --mode chain --exp_id grid_state_chain_001 --steps 2000 --H 16 --W 32 --dim 64 --stateful

# GT3 trace
python -m src.train_diffusion_grid --mode trace --exp_id grid_trace_enabled_001 --steps 2000 --H 16 --W 32 --trace_enabled
python -m src.train_diffusion_grid --mode trace --exp_id grid_trace_disabled_001 --steps 2000 --H 16 --W 32  # no flag = erased

# GT4 certainty sweep (tool calling)
python -m src.train_diffusion_grid --mode certainty --exp_id grid_certainty_09_001 --tau 0.9 --steps 1000 --H 16 --W 32

# Smoke (50 steps, verified)
python -m src.train_diffusion_grid --mode recon --grid_type random --exp_id grid_recon_smoke_test --steps 50 --H 8 --W 8 --dim 32 --log_every 10
# loss 5.55→5.53, diffusion decode iter 10 acc 0.67 (random baseline because untrained, will improve with training)
```

**Synthetic data (infinite, no overfit, per rwkv.md):**
- Random printable ASCII grid
- Box border `+--+ / |  |` – 2D locality test
- Terminal chain: ls → cat file1.txt → "What was SECRET?" – tool + temporal
- Trace grid: NUMS in rows 0-2, trace rows 10-13 scratchpad

**Relation to existing fixes:**
- Depends on `shared_state_unrolled_feedback_001` (decoder feedback = previous screen embedding), without it grid decoder would stall like B3
- Depends on `rwkv-state-carry` (carry num/den across screens) for temporal
- Uses B3D triplet as training layout for grid – b3d-rwkv-nano is substrate

**Why this matters for terminal use:**
- Current terminal LLM types char-by-char, can't see its own rendering errors, no temporal memory
- Grid diffusion types only when certain, sees full screen (vision approx), self-corrects low-certainty cells via re-diffusion, carries state across screens (world understanding)
- Could be scaled to real 24×80 terminal (1920 bytes) at 1M params, with ANSI colors → true vision approx without CNN

---

## Combined Batch Status

You now have **7 theories, 15 theory files, 11 train scripts**:

**Batch A (B5 follow-ups, already in theories/archive/new_theories_and_experiments.md):**
- injection-frequency, dynamic-patch, token-vs-byte, rwkv-state-carry, adaptive-exit-entropy

**Batch B (this doc):**
- b3d-rwkv-nano, diffusion-grid-terminal

All CPU runnable, 2k steps matches paper's small-scale proof style.

**Smoke verification 2026-07-17 (this env torch 2.13 cpu):**
- b3d_triplet_smoke_test 50 steps loss 4.31→4.26 params 45K ✅
- grid_recon_smoke_test 50 steps loss 5.55→5.53 params 44K ✅
- (previous batch A also verified – see NEW_THEORIES...md)

**Next steps (cheap→expensive):**
1. Run BD1 full 2k steps at dim=128 char vocab – prove triplet trains vs AR
2. Run BD2 tau sweep – measure iter per block speedup, compare to paper 1.6×
3. Run GT6 box vision – if model reconstructs box border, 2D spatial proven
4. Run GT2 chain stateful vs zero – if +0.2 acc, temporal awareness proven (enables tool calling)
5. Scale grid to 24×80 at 1M params, capture real terminal `ls/cat` logs as dataset

All experiments log to `experiments/<exp_id>/` per AGENTS.md.

## TL;DR Implementation Map

```
screen (HxW bytes) --mask 30%--> [b1=masked, b2=masked lossable, b3=clean] --concat--> physical 3*H*W
physical --embed + row_embed + col_embed--> RWKV nano (with state carry S_t) --> logits
loss = CE on b2 masked positions vs clean
inference: b1 masked -> RWKV -> commit where prob>τ -> unmask -> repeat -> trigger typing
next screen observed -> S_{t+1} carries
```

This is B3D triplet + 2D pos + certainty gate + state carry = terminal agent substrate.
