# Dendrite Growth

A second attempt at the dendrite idea, after the LoRA version didn't quite match what was wanted. The intent here is an RWKV core that stays exactly as it is while new modules attach to it — like adding limbs to a tree trunk.

## What the user actually said

*"its an RWKV that grows"* — *"no i wanted extensions to the network. imagine in the ideal. growing from a core and adding parts as you go. but keeping RWKV as is."*

Plus the additional context from later sessions:

- Dendrites can carry their own state specifically for the dendrite's purpose.
- The state can drive whether the dendrite routes or not.
- A dendrite can be stochastically picked by inserting NO-OPS and observing which one steers toward the target state.
- The trunk's intelligence is assumed maxed-out across all relevant sectors — adding more knowledge would cause entanglement in a single model.
- A larger model is assumed to solve the entanglement (control).
- The dendrite allows expansion without retraining unrelated parts.
- The gating logic can be optional; it could be a pre-trained gate that watches activations on the target weights.
- "Gate neurons" are singular neurons in a region that other neurons might pass through, switching the whole model into a different mode.
- It is possible to track activations for specific topics across an RWKV model.

## What this implies mechanically

- The trunk (RWKV layers 0..L-1) is frozen.
- Each branch is its own module with its own weights and possibly its own state buffer.
- Routing is **optional** for the simplest case — a branch just processes trunk output. Routing is mandatory if many branches would otherwise interfere.
- State-based routing: the trunk's WKV state can be tapped to decide which branch runs.
- No-OPS routing: at inference, drop in NO-OP branches and keep the one that moves the state in the desired direction. This works because branches are *stochastic in their effect on state*.
- Activation tracking is a research tool: run a probe through trunk, find a region where some neurons spike on target-topic inputs — that's where the gating neuron lives.

## Predicted structure

```
Trunk (frozen RWKV) ──► hidden h_l ──┬──► branch-1 trainable ──► logits
                                      ├──► branch-2 trainable
                                      ├──► NO-OP branch-A      (probe only)
                                      └──► NO-OP branch-B      (probe only)
```

On inference:
1. Probe by running all branches including NO-OPs.
2. Pick the one whose output moves trunk-state closest to the desired region.
3. Take that branch's output as the next-step prediction.

On training:
1. Trunk frozen forever.
2. Each branch trained in isolation on its task.
3. Routing (when present) trained on probe-derived gates.

## What's the smallest proof

A trivial case: trunk-RWKV with one branch trained to do a single task the trunk was not trained to do. The branch is a cross-attention block + head. Show:
- Trunk logits on probe unchanged (within ε) after branch training.
- Branch outputs the right answer on its task.
- Adding a NO-OP branch does not change trunk behavior.
- Adding a second functional branch does not change the first branch's accuracy.

That's the G1 claim from the previously-deleted infer file, packaged here as the only hypothesis worth proving.

## Not this

Not LoRA. A LoRA adds rank-r matrices to existing projections — that mutates trunk weights in disguised form. Not state-only — branch state is auxiliary; trunk WKV state stays untouched. Not pure MoE on Transformer — the trunk is RWKV.

## Theory status

- **G1**: Frozen trunk + branch, trunk unchanged, branch task learned — not proven, implementation pending.
- **G1a**: Trunk logits within ε after branch train — open.
- **G1b**: Two branches independent — open.
- **G1c**: Activation tracking finds gating neurons — open.
- **G1d**: NO-OP probing picks the right branch without learned router — open.

---

**Research links:** [`research/modular_dendritic_networks.md`](../research/modular_dendritic_networks.md) — LoRA, PathNet, MoE, Progressive Networks, Net2Net, dendritic computation. See also [`research/memory_associative.md`](../research/memory_associative.md) for complementary learning systems (hippocampus ↔ neocortex) as a framework for trunk + branches.
