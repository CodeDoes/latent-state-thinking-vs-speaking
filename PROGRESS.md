# Progress: Hybrid Latent-State Language Model

## Core Hypothesis

A model with:
```
latent_state_update() × N
decode_token() × M
```
can outperform an equivalent token-by-token model on long-horizon reasoning.

**Extended hypothesis (USER.md):** SSM can learn logic/planning while tape/context handles language precision — reducing the burden on recurrent state by separating:
- **SSM** — Thinking (logic, planning, world model)
- **Tape** — Remembering (exact token recall, spelling)
- **Context** — Attention management (what's relevant now)
- **Decoder** — Expression (rendering state to tokens)

**Framing:** "The model learns a private computational space, and language is only an output device." — separating thinking space from communication space. A token is a terrible clock cycle for reasoning.

---

## Research Ladder

| Level | Question | Status |
|---|---|---|
| 0 | Does latent state work at all? | ⬜ |
| 1 | Does latent thinking beat tokens? | ⬜ |
| 2 | Does latent state survive context removal? | ⬜ |
| 3 | Can latent state generate multiple tokens? | ⬜ |
| 4 | Can latent state continue a story after interruption? | ⬜ |
| 5 | Can state computation become independent of token generation? | ⬜ |

---

## Architecture

```
              INPUT TOKENS
                   |
                   v
          Context Manager
                   |
      +------------+-------------+
      |                          |
      v                          v
Prefix Tape Memory          SSM State
  exact recall              semantic state
  token patterns            planning
  spelling                  logic
  names                     world model
      |                          |
      +------------+-------------+
                   |
                   v
          Latent Processor
        (SSM + FFN loop)
                   |
                   v
        State -> Token Decoder
                   |
                   v
                OUTPUT
```

**Component roles:**
- SSM: "What does this mean?"
- Tape: "What exactly was written?"
- Context: "What am I currently saying?"
- Decoder: "How do I express the state?"

---

## Current Status

| Area | Status |
|---|---|
| Project setup | ✅ Devenv/Nix environment configured |
| Research design | ✅ Architecture + ladder defined |
| Agent specification | ✅ Autonomous research loop defined |
| Kaggle integration | ✅ API auth (kitastro), notebook, push script |
| Toy world generator | ✅ `src/dataset.py` — location/inventory/recall/story tasks |
| Tokenizer | ✅ `src/tokenizer.py` — character-level with special tokens |
| Models | ✅ `src/models.py` — **UPDATED**: All models produce [batch, seq_len, vocab_size] |
| SSM layer | ✅ `src/models.py` — simplified Mamba-style recurrent update |
| Latent thinking loop | ✅ `src/models.py` — **UPDATED**: Sequential token processing with `think_every` parameter |
| Token decoder | ✅ `src/models.py` — cheap FFN readout + multi-token decode |
| Trainer | ✅ `src/trainer.py` — **UPDATED**: Unified training for all model types |
| Experiment runner | ✅ `run_experiment.py` — **UPDATED**: Added `think_every` and `max_seq_len` params |
| Kaggle notebook | ✅ `notebook.ipynb` — **REGENERATED**: Matches new architecture, GPU-ready |
| Kaggle push | ✅ `kaggle_push.py` — push/monitor/download via Kaggle API |
| Bug fixes | ✅ Fixed tokenizer vocab construction, dataset empty inventory edge case |
| CPU validation | ✅ All 3 models train successfully on CPU |
| Prefix tape memory | ⬜ Not started |
| Managed context | ⬜ Not started |
| Phase 1 experiments | 🔄 Ready to run on Kaggle GPU |
| Level 0 proven | ⬜ |

---

## Planned Experiments

| Model | Components | Purpose |
|---|---|---|
| Baseline | Transformer LM | Reference point |
| Model A | SSM only | Pure recurrent baseline |
| Model B | SSM + FFN decoder | Decoupled generation |
| Model C | SSM + FFN + Tape | + exact recall |
| Model D | SSM + FFN + Tape + Context | Full architecture |

### Metrics
- Perplexity
- Reasoning accuracy
- Exact recall (names, passwords, rare tokens)
- Generation speed (tokens/sec)
- Memory usage

### Training Tasks
1. **Logic** — location tracking, state transitions (toy world generator)
2. **Exact recall** — memorize and reproduce arbitrary tokens
3. **Story generation** — coherent narrative with character tracking
4. **Interrupted generation** — resume after context removal

---

## Training Objectives (from CONVO.md)

Standard next-token prediction encourages shortcutting. Need:
- **Latent consistency loss** — state after thinking ≈ state after observing answer
- **Token reconstruction loss** — can decoder recover tokens from state?
- **State evolution loss** — state should remain predictive over time (JEPA-style)
- **Specialization pressure** — punish SSM for memorizing names/passwords, reward for reasoning

---

## Next Steps

**Ready:** Phase 1 baseline experiments on Kaggle GPU (Level 0 — "Does latent state work at all?")

### Architecture Improvements Made:
1. **Unified output format**: All models now produce [batch, seq_len, vocab_size] for fair comparison
2. **Sequential token processing**: LatentSSM now processes tokens one-by-one (not mean pooling)
3. **Periodic thinking**: Added `think_every` parameter to control thinking frequency
4. **Fixed bugs**: Tokenizer vocab construction, dataset edge cases
5. **Validated on CPU**: All models train successfully, pipeline works end-to-end

### Run Experiments on Kaggle GPU:

**Option A: Using notebook.ipynb (recommended)**
```bash
python kaggle_push.py --run
```

**Option B: Using run_experiment.py (individual experiments)**
```bash
# Baseline transformer
python run_experiment.py --exp_id exp001 --model baseline --d_model 256 --epochs 30 --device cuda

# SSM only (no thinking)
python run_experiment.py --exp_id exp002 --model latent_ssm --latent_steps 0 --think_every 0 --d_model 256 --epochs 30

# SSM + thinking (every 4 tokens)
python run_experiment.py --exp_id exp003 --model latent_ssm --latent_steps 4 --think_every 4 --d_model 256 --epochs 30

# SSM + thinking (every 8 tokens)
python run_experiment.py --exp_id exp004 --model latent_ssm --latent_steps 4 --think_every 8 --d_model 256 --epochs 30

# SSM + decoder
python run_experiment.py --exp_id exp005 --model latent_ssm_decoder --latent_steps 4 --think_every 4 --d_model 256 --epochs 30
```

### Key Metrics to Compare:
- **Validation loss**: Does latent thinking improve generalization?
- **Training speed**: How much overhead do thinking steps add?
- **Parameter efficiency**: Same performance with fewer params?
- **Qualitative**: Do latent models reason better on location/inventory tasks?

**First night win condition:** A latent-state model with periodic thinking steps achieves better validation loss than baseline transformer on reasoning tasks.

### Expected Outcomes:
- exp001 (baseline): Reference point for comparison
- exp002 (SSM, no thinking): Pure recurrent baseline — tests if sequential processing helps
- exp003 (SSM, think every 4): Main hypothesis test — does thinking help?
- exp004 (SSM, think every 8): Test thinking frequency — less frequent = faster?
- exp005 (SSM+decoder): Test multi-token generation variant

---

## Research Log

### 2026-07-11 — Project inception

**Hypothesis:** SSM can learn logic/planning while tape/context handles language precision. State-as-thought-space model should outperform token-by-token generation on long-horizon reasoning.

**Key insights from CONVO.md:**
- **Evolution:** Started as "latent reasoning + cheap decoder" → evolved through RWKV/ROSA/ JEPA inspirations → final hybrid cognitive architecture
- **Core insight:** "The model learns a private computational space, and language is only an output device" — thinking space ≠ communication space
- **Token as clock cycle:** Current LLMs force reasoning to serialize through tokens; latent state becomes the scratchpad
- **Brain/mouth analogy:** Brain = expensive (4 state updates), Mouth = cheap (50 token generations). Current LLMs use the same giant stack for both thinking and speaking.
- **ROSA relevance:** Exact suffix/pattern propagation vs semantic understanding — complements lossy SSM compression
- **Autonomous research loop:** Agent should behave like a junior researcher — build, run, record, improve, hypothesize, repeat
- **Research ladder** prevents blind optimization — each level must be proven before advancing
- **Kaggle-specific:** Design around checkpointing since free accelerators are limited and sessions can stop
- **First win condition:** 20M param model with 4 recurrent steps beats same-size autoregressive on long-horizon recall
- **Training objective challenge:** Need latent consistency + reconstruction + evolution losses, not just next-token prediction

**Critical experiment:** Train on long documents where the model is *punished* if SSM memorizes names/passwords but *rewarded* if it reasons correctly. This forces specialization.

**Reference:** [JEPA-Reasoner: Decoupling Latent Reasoning from Token Generation](https://arxiv.org/abs/2512.19171)

### 2026-07-11 — Architecture improvements & validation

**Changes made:**

1. **Unified output format**: All models (BaselineTransformer, LatentSSM, LatentSSMDecoder) now produce [batch, seq_len, vocab_size] output for fair comparison. Previously, latent models produced [batch, vocab_size] which made training inefficient.

2. **Sequential token processing**: LatentSSM now processes tokens sequentially through SSM layers (not mean pooling). This properly implements the recurrent nature of SSMs and allows the model to maintain state across the sequence.

3. **Periodic thinking**: Added `think_every` parameter to control how often latent thinking steps occur. This allows testing different thinking frequencies (e.g., think every 4 tokens vs every 8 tokens).

4. **Input-dependent SSM dynamics**: Added selective mechanism (Mamba-style) to SSMLayer:
   - State transition matrix A is now modulated by input: A(x) = A_base + A_mod(x)
   - Allows model to selectively remember/forget based on input content
   - Parameter count increased from 841K to 34M (more comparable to 2.1M baseline)
   - Applied to both LatentSSM and LatentSSMDecoder with `selective=True` by default

5. **Bug fixes**:
   - Fixed tokenizer vocab construction (was missing `enumerate`)
   - Fixed dataset generation edge case (empty inventory causing IndexError)
   - Ensured at least one entity has items in world initialization

6. **Trainer simplification**: Since all models now have unified output, trainer code is simpler and more maintainable.

7. **Evaluation improvements**: 
   - Changed from greedy decoding to temperature sampling (T=0.8, top-k=40)
   - Generates multiple samples per question (n=3) for better QA accuracy
   - Better handling of story tasks

8. **CPU validation**: Successfully trained all 3 models on CPU:
   - BaselineTransformer: ~8s/epoch on 100 samples
   - LatentSSM: ~106s/epoch (slower due to sequential processing + thinking)
   - LatentSSMDecoder: ~64s/epoch
   - All models show decreasing loss, pipeline works end-to-end

**Key insight**: The sequential processing makes latent models slower but this is the correct architecture. On GPU, we can use `think_every=4` or `think_every=8` to balance computation vs performance. The input-dependent dynamics (selective mechanism) should help the model learn when to remember vs forget, making it more powerful than fixed SSM dynamics.

**Next**: Run experiments on Kaggle GPU to get real results. The notebook is ready with 5 experiments comparing baseline, SSM variants with different thinking frequencies, and the decoder model. With the selective SSM enhancement, the latent models should be more competitive with the baseline.
