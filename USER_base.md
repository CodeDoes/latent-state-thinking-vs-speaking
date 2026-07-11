Ah, then you don't want a "write me a proof of concept" prompt. You want an **autonomous research loop specification**.

The agent should behave more like a junior researcher:

1. Build baseline.
2. Run experiments.
3. Record results.
4. Keep improvements.
5. Generate hypotheses.
6. Repeat until compute budget expires.

This is very similar to recent "AI researcher" style workflows where agents modify code, run experiments, and keep only validated improvements. ([DataCamp][1]) Experiment tracking is also considered critical because failed experiments contain useful information about what does not work. ([Google for Developers][2])

---

I would give the agent something like this:

```markdown
# Project: Latent State Reasoning Model

## Goal

Develop and evaluate a neural architecture where:

- latent_state is the primary reasoning memory
- token generation is a lightweight decoder
- internal computation can happen without emitting tokens

Hypothesis:

A model with:
    latent_state_update() × N
    decode_token() × M

can outperform an equivalent token-by-token model on long horizon reasoning.

---

# Rules

You are an autonomous ML researcher.

Continue working until compute budget is exhausted.

For every experiment:

1. Create a unique experiment ID.
2. Record:
   - hypothesis
   - code changes
   - hyperparameters
   - training loss
   - evaluation scores
   - generation samples
   - conclusions

Never overwrite previous results.

Maintain:

experiments/
    exp001/
    exp002/

results.json
best_model.pt
research_log.md

---

# Phase 1: Baseline

Implement:

A) Tiny transformer LM

B) Recurrent latent model

Architecture:

input_tokens
    |
embedding
    |
latent_state

repeat:
    latent_state =
        SSM(latent_state, input)

decode:

latent_state -> token logits


---

# Phase 2: Synthetic reasoning tasks

Create datasets:

1. Memory task

Example:

John entered kitchen.
John picked apple.
John left.

Question:
Where is apple?

---

2. Story continuation

Input:

Write a story about John.

Internal state should represent:

entity:
    John

location:
    kitchen

current_action:
    eating

narrative_goal:
    introduce mystery


---

# Phase 3: Experiments

Test:

latent_steps:

1
2
4
8
16


Compare:

same parameter count

against:

normal autoregressive baseline


Measure:

- accuracy
- perplexity
- long context recall
- tokens generated
- inference speed


---

# Phase 4: Improvements

Try:

- gated SSM updates
- FFN inside latent loop
- separate memory/context vectors
- latent consistency loss
- state reconstruction loss
- state -> token decoder compression


---

# Research questions

Answer:

1.
Does more latent computation improve reasoning?

2.
Can token generation be cheaper than state computation?

3.
Can latent state preserve information after context removal?

4.
Can the model resume generation from latent_state alone?


---

# Final output

Before stopping:

Create:

README.md

containing:

- architecture diagram
- best model
- failed experiments
- graphs
- conclusions
- next research directions

Upload notebook/model/results.
```

---

The biggest thing I would change from your earlier idea:

**Don't let the agent optimize blindly. Give it a "research ladder."**

Otherwise it will spend 10 hours tweaking learning rates.

The ladder:

```
Level 0:
Does latent state work at all?

Level 1:
Does latent thinking beat tokens?

Level 2:
Does latent state survive context removal?

Level 3:
Can latent state generate multiple tokens?

Level 4:
Can latent state continue a story after interruption?

Level 5:
Can state computation become independent of token generation?
```

---

For Kaggle specifically, design around checkpointing because free accelerators are limited and sessions can stop. Kaggle notebooks have accelerator quotas and session constraints, so saving models/results frequently matters. ([Packt][3])

The agent should therefore do:

```python
every experiment:

save checkpoint
save metrics
save sample outputs
commit result
```

---

The interesting part is that you don't actually need the overnight agent to discover the final architecture.

A successful first night would be:

> "A 20M parameter latent-state model with 4 recurrent thinking steps achieves X% better long-horizon recall than a same-size autoregressive model."

That is enough to justify the next iteration.

The overnight agent is not building the brain. It is trying to prove whether the **state-as-thought-space** hypothesis has any signal.

[1]: https://www.datacamp.com/tutorial/guide-to-autoresearch?utm_source=chatgpt.com "A Guide to Andrej Karpathy’s AutoResearch: Automating ML with AI Agents | DataCamp"
[2]: https://developers.google.com/machine-learning/managing-ml-projects/experiments?utm_source=chatgpt.com "Experiments  |  Machine Learning  |  Google for Developers"
[3]: https://subscription.packtpub.com/book/data/9781801817479/2/ch02lvl1sec04/introducing-kaggle?utm_source=chatgpt.com "Introducing Kaggle and Other Data Science Competitions | The Kaggle Book"
