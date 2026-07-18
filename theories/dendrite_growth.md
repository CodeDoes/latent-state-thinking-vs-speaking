# Dendrite Growth — Verbatim

> **Source**: Verbatim from user across sessions (the "growing RWKV" subset, not the LoRA one).  
> **Date range**: 2026-07-15 to 2026-07-17  
> **You said**:  
  
"its an RWKV that grows"

"no i wanted extensions to the network. imagine in the ideal. growing from a core and adding parts as you go. but keeping RWKV as is."

"the word of god said to use state instead for @reports/dendritron.md"

"also the word of angle said this reminded them of this paper https://arxiv.org/html/2606.14243v1"


=== ADDITIONS (remove after reading and updating docs)

when adding dendrite expansion. you can add dedicated state specifically for this. 

or you can use the state to determine whether to route to the dendrite or not. 

Theory: the dendrite can be stochastically determined. you can randomly add NO-OPS and see which dendrites is more likely to cause the desired state.

Assumption: The model's intelligence capacity is maxed out in all relavent sectors and adding more knowledge would cause entanglement. 

Assumption: This can be solved by a larger model.

Assumption: Dendrite allows expansion without requiring retraining of unrelated parts, and at most requires gating logic to be trained (optionally could also be part of the dendrite and could be trained beforehand from observing activations on the target weights)

Unrelated Assumption: Gate neurons are singular neurons in a region that other neurons might pass through that cause a different mode of behaviour for the greater model.

Assumption: you can track activations for certain topics accross a RWKV model.
