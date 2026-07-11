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
| Kaggle planning | ✅ Checkpointing + session constraints noted |
| Conversation history | ✅ Documented in CONVO.md (~2000 lines) |
| SSM implementation | ⬜ Not started |
| Prefix tape memory | ⬜ Not started |
| Managed context | ⬜ Not started |
| Latent thinking loop | ⬜ Not started |
| Token decoder | ⬜ Not started |
| Training tasks | ⬜ Not started |
| Experiments | ⬜ Not started |

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

See PLAN.md for detailed phased plan. Immediate next step: **Phase 1 — Baseline** (implement tiny transformer LM + recurrent latent model + toy world generator).

**First night win condition:** A 20M parameter latent-state model with 4 recurrent thinking steps achieves X% better long-horizon recall than a same-size autoregressive model.

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
