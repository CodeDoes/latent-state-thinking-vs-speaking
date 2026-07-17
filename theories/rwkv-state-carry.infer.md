# rwkv-state-carry.infer

Interpretation of rwkv-state-carry.md.

What .md leaves implicit:

**Connection to B1 (encoder-state matters)**: encoder_state_ablation proved static encoder+patch state beats mutable. That was about *whether* state content matters, not about *how* state persists across time. This theory is temporal extension: does persisting RWKV's internal num/den across segments beat resetting?

**Why TBPTT style detaches**: If we carry state with full grad, gradient flows across 100s of steps → vanishing/exploding + memory O(T^2). Detaching preserves memory content but stops gradient, isolating memory usefulness from optimization difficulty. Full BPTT would conflate.

**What fix unlocked**: Pre-fix bug had num/den decay wrong (exp blending inverted) and xx state saved at wrong scale (post-ln vs pre-ln). That meant even if you tried to carry, it was garbage. Now carry is correct, so theory can be tested. Without fix, this ablation would have been meaningless – would have "proven" carry doesn't help due to bug.

**Long vs short context task**: logic_niiah_generator already has noise_min/max controlling distance between needle and question. For this theory, we need max distance 200+ tokens to need memory. Existing generator defaults noise 1-3 short. So need to configure noise_max=10, repeat transforms.

**Learned init**: Alternative to carry is to learn initial state vector (like RNN initial hidden). That could encode "task prior" without needing carry. Arm C tests that. If learned init beats zero but not carry, then carry still adds value beyond prior.

**Metrics beyond accuracy**: Log per-channel decay values w_ = exp(-exp(time_decay)). Distribution tells if model learned long-memory channels (w_ close to 1). Before fix, distribution collapsed. After fix, should see spread. Carry arm should push more channels to long-memory (higher w_).

**Failure mode**: Carry might hurt if batch ordering random (generator RNG). Must ensure stream continuity: chunk should be contiguous slice of long synthetic story, not random resampling each batch. Otherwise carrying state from unrelated story is noise. So B needs sequential batching, not random.
