# Plan: Hybrid Latent-State Language Model

## Agent Behavior Rules

You are an autonomous ML researcher. Continue working until compute budget is exhausted.

For every experiment:
1. Create a unique experiment ID
2. Record: hypothesis, code changes, hyperparameters, training loss, evaluation scores, generation samples, conclusions
3. Never overwrite previous results
4. Save checkpoint, metrics, and sample outputs after each experiment

Maintain:
```
experiments/
    exp001/
        config.json
        metrics.json
        samples.txt
        model.pt
    exp002/
    ...

results.json
best_model.pt
research_log.md
```

---

## Research Ladder

Each level must be proven before advancing. Prevents blind hyperparameter tweaking.

### Level 0: Does latent state work at all?
- Implement minimal SSM backbone
- Verify latent state can encode and decode input
- Train on simple copy/reconstruction task
- **Success criterion:** Loss decreases, model can reconstruct input from latent state

### Level 1: Does latent thinking beat tokens?
- Compare latent state model against equivalent autoregressive transformer
- Test on reasoning tasks requiring multi-step computation
- Vary latent_steps: 1, 2, 4, 8, 16
- **Success criterion:** Latent model achieves better accuracy at same parameter count

### Level 2: Does latent state survive context removal?
- Train model to preserve information in latent state
- Remove input context after encoding
- Test if model can still answer questions from state alone
- **Success criterion:** Model retains key information without access to input tokens

### Level 3: Can latent state generate multiple tokens?
- Implement state → token decoder that generates multiple tokens per state update
- Target ratio: 4 expensive state updates → 50 cheap token generations
- **Success criterion:** Generated text is coherent across multiple decode steps

### Level 4: Can latent state continue a story after interruption?
- Generate partial story, pause, remove context
- Resume from latent_state + tape + managed context
- Measure coherence of continued generation
- **Success criterion:** Story continuation is semantically consistent

### Level 5: Can state computation become independent of token generation?
- Full architecture with SSM + Tape + Context + Decoder
- State evolves without emitting tokens
- Decoder renders state to tokens on demand
- **Success criterion:** State computation and token generation are functionally decoupled

---

## Phase 1: Baseline

### Goal
Implement and verify two baseline models:

**A) Tiny Transformer LM**
- Standard autoregressive transformer
- Reference point for comparison

**B) Recurrent Latent Model**
```
input_tokens
    |
  embedding
    |
  latent_state (256-1024 dim)

repeat:
    latent_state = SSM(latent_state, input)

decode:
    latent_state -> token logits
```

**Two operations:**
```python
state = think(state, input_embedding)
token = speak(state)
```

### Deliverables
- [ ] SSM layer implementation
- [ ] Transformer baseline implementation
- [ ] Embedding layer
- [ ] Token decoder (linear projection)
- [ ] Training loop
- [ ] Basic evaluation on synthetic tasks

---

## Phase 2: Synthetic Reasoning Tasks

### Goal
Create datasets for evaluating latent reasoning.

### Toy World Generator
```python
world = {
    "people": {
        "John": {
            "location": "kitchen",
            "inventory": ["apple"]
        }
    }
}
```

Generates narratives and questions from a simulated world state.

### Task 1: Memory / Logic
```
John entered kitchen.
John picked apple.
John left.

Question: Where is apple?
Answer: kitchen
```
SSM should learn location tracking and state transitions.

### Task 2: Exact Recall
```
The password is: Zx91Kq77

[100 tokens later]

Question: What was the password?
Answer: Zx91Kq77
```
Tape should solve this; SSM should not need to memorize exact tokens.

### Task 3: Story Continuation
```
Write a story about John.

Expected latent representation:
  entity: John
  location: kitchen
  current_action: eating
  narrative_goal: introduce mystery
```

### Task 4: Interrupted Generation
- Generate partial story
- Pause and remove context
- Resume from latent_state + tape
- Measure coherence

### Deliverables
- [ ] Dataset generation scripts
- [ ] Evaluation metrics
- [ ] Sample outputs

---

## Phase 3: Experiments

### Model Variants

| Model | Components | Purpose |
|---|---|---|
| Baseline | Transformer LM | Reference |
| Model A | SSM only | Pure recurrent baseline |
| Model B | SSM + FFN decoder | Decoupled generation |
| Model C | SSM + FFN + Tape | + exact recall |
| Model D | SSM + FFN + Tape + Context | Full architecture |

### Variables to Test
- `latent_steps`: 1, 2, 4, 8, 16
- Parameter count: keep equal across variants
- Compare against normal autoregressive baseline

### Metrics
- [ ] Accuracy on reasoning tasks
- [ ] Perplexity
- [ ] Long context recall
- [ ] Tokens generated per second
- [ ] Inference speed
- [ ] Memory usage

### Deliverables
- [ ] Experiment runner
- [ ] Results comparison
- [ ] Ablation studies

---

## Phase 4: Improvements

### Ideas to Test
- [ ] Gated SSM updates
- [ ] FFN inside latent loop
- [ ] Separate memory/context vectors
- [ ] Latent consistency loss
- [ ] State reconstruction loss
- [ ] State → token decoder compression

### Deliverables
- [ ] Improved architecture
- [ ] Final comparison against baseline
- [ ] Documented findings

---

## Training Objectives

Standard next-token prediction encourages shortcutting. Implement:

1. **Latent consistency loss** — state after thinking ≈ state after observing answer
2. **Token reconstruction loss** — can decoder recover tokens from state?
3. **State evolution loss** — state should remain predictive over time (JEPA-style)
4. **Specialization pressure** — punish SSM for memorizing names/passwords, reward for reasoning

---

## Evaluation Metrics

1. **Latent persistence** — Can it answer after the input is gone?
   - Read story → forget tokens → question → answer

2. **Latent rollout** — Does state → think → state improve accuracy?

3. **Decoder speed** — Compare:
   - Normal: 100 tokens = 100 expensive model passes
   - Ours: 4 expensive state updates + 100 cheap decoder calls

---

## Final Output

Before stopping, create README.md containing:
- Architecture diagram
- Best model checkpoint
- Failed experiments and learnings
- Performance graphs
- Conclusions
- Next research directions

Upload notebook/model/results to Kaggle.

---

## Kaggle Considerations

- Design around checkpointing — free accelerators are limited, sessions can stop
- Save models, metrics, and samples frequently
- Commit results after each experiment
- Target: 20M parameter model fits within Kaggle constraints

**First night win condition:** A 20M parameter latent-state model with 4 recurrent thinking steps achieves X% better long-horizon recall than a same-size autoregressive model.

---

## Research Questions

1. Does more latent computation improve reasoning?
2. Can token generation be cheaper than state computation?
3. Can latent state preserve information after context removal?
4. Can the model resume generation from latent_state alone?
