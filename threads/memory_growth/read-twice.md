# read-twice

A cheaper alternative to progressive expansion when capacity is pressed: instead of inserting new layers, run the existing recurrence for additional forward passes. The state accumulates further. No new weights. No new training.

## What the user said

*"i think RNN's can do 'read twice' instead."*

## Hypothesis

A model given 2 recurrent passes learns harder tasks faster at matched compute than a model that does 1 pass + a new layer.

- Pro: no new parameters, no risk of disturbing existing capabilities, no new training.
- Con: extra compute cost.

## Mechanism sketch

At inference time, instead of `output = model(input)`, run:
```
state = state_0
output_1, state_1 = model(input, state_0)
output_2, state_2 = model(input, state_1)  # same input, second pass
final = output_2  # or f(output_1, output_2)
```

The second pass sees its own previous output through the state but not through new weights. Pure recurrence deepens the same function.

## Relationship to other work in the project

| Work | Relationship |
|---|---|
| `progressive-expansion.md` | Sister theory. Progressive adds layers; read-twice adds passes. Same inverse goal (surgical surgery on capacity). |
| `adaptive-exit-entropy.md` | Reads as well: instead of exiting *early* on confidence, run *longer* on insufficient capacity. Symmetric to current "exit early when confident" logic. |
| `dendrite_growth.md` | Pass-depth ammortizes over a *single trunk*; dendrite-growth has many trunks + shared base. Different lever. |
| `exp001` (proven) | Already showed amortised thinking wins when the same state can be queried cheaply. Read-twice extends to the case where one pass is not enough. |

## Open questions

1. Monotonicity. Does more passes always help, or does it start unlearning the easy task after too many? Need a curve.
2. Per-example gating. Should every input get the same passes, or should the model learn when to stop?
3. GRU vs RWKV. Does it work for both recurrence types?
4. Cost framing. At matched *FLOPs*, pass-depth vs new-layer creates different wallclock profiles. The right comparison is "FLOPs-equivalent."

## Minimal test

1. Train a small GRU on task A (single pass) until converged.
2. Run on task B (harder — same domain, but bigger) with 1, 2, 3, 4 passes. Plot accuracy vs pass-depth.
3. Same compute budget. Train a model with an extra linear layer inserted at the bottleneck (per progressive-expansion) on task B. Compare accuracy.
4. Single question: which learns faster at matched compute?

Status: not yet run. Implementation could land in a smoke test in ~30 minutes once a bottleneck-aware layer-insertion variant exists to compare against.
