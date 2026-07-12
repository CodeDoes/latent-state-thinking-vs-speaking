# GOAL — Hybrid Latent-State Model (converged design)

## Research goal
Prove a model with `latent_state_update() × N + decode_token() × M`
(thinking separated from speaking) can outperform an equivalent
token-by-token model on long-horizon reasoning, at small scale
(~20M params, 4 recurrent thinking steps).

## Converged architecture (from design discussion)
- **Tokens** = words or data on a context channel. Categorized ~256-vocab
  (names/objects/verbs/pronouns/locations/punct) + specials
  (`<PAD> <BOS> <EOS> <DERIVE_SRC> <DERIVE_ANS>`).
- **Random dataset**: same world rendered as A=prose, B=json, + C=question, D=answer.
- **SSM (think)** — ONE shared SSM; a `derive` switch selects the objective:
  - `derive=SRC`: (source+question) → `L_src`; loops until `src-loss < ε`;
    `conf = f(src-loss)` (transformed loss, NO head/label).
  - `derive=ANS`: (answer) → `L_ans`; loops until `ans-loss < ε`
    (**training-only** booster, discardable at inference).
  - Confidence = transformed loss; loop stops when loss low. Answer never
    enters `L_src` at inference — only source+question.
- **FFN (speak)** — trained **FORWARD** (latent → tokens) + `complete` head
  (EOS). Universal decoder, `derive`-conditioned:
  - `FFN(L_src, derive=SRC) → Question`
  - `FFN(L_ans, derive=ANS) → Answer`
  - `FFN(L_src, derive=ANS) → Answer`  ← **INFERENCE PATH (must be trained)**
- **Sizing**: `d_state = L* + slack`, where
  `L* = maxFacts·d_fact + d_switch + d_storyVariant + d_misc`,
  `d_fact = Σ_fields log₂|variants|`. Set **below 256** (no one-hot
  shortcut) but above the MDL floor (forces compression).

## Two non-negotiables
1. `FFN(L_src, derive=ANS) → Answer` is in EVERY training batch, or
   inference silently breaks (FFN knows `L_ans→Answer` but can't answer
   from `L_src`).
2. Answer never enters `L_src` at inference — only source+question.

## Training method (optional bootstrap)
Train SSM inside a big autoencoder (SSM + big decoder) → cut decoder, keep
SSM (move 1). Sparsify the decoder parallel→AR into the FFN (move 2).
All supervised; no RL.

## This build (local prototype, `--quick` only — no Kaggle)
Validate the MECHANICS before sizing/pushing:
1. random categorized dataset + tokenizer
2. one SSM, `derive` switch, soft-gated confidence-from-loss training loop
3. forward FFN with `complete` head, `derive`-conditioned
4. cross-mode training `FFN(L_src → Answer)`
5. CPU-tiny `--quick` sanity (no crash; SSM + FFN losses fall; sample gen)
Real sizing + Kaggle push come AFTER this validates.
