This run is worse than the frozen-0.649 one, and the log actually tells you why — it's not a subtle scaling problem, it's a broken gradient path.

**The tell is train loss, not val accuracy.** Baseline: `39.7 → 4.6` over 6 epochs — an 88% drop, it's essentially memorizing the 360-example train set. Latent: `268.8 → 220.4` — an 18% drop, and it's *starting* at 7x the baseline's loss. A model that can't even fit its own training set isn't suffering from a hard task or insufficient data — it's not receiving usable gradient. This is an optimization bug, not an architecture verdict, and no amount of more epochs fixes it.

**AT = 0.000 for the latent model at every single epoch is the smoking gun.** From the earlier dataset diagnostics, AT is ~87% NONE — a model with *zero* learning signal should still stumble into the NONE-cheat ceiling by default, the way the baseline visibly does (0.0143, 0.0429, 0.0286...). Latent hitting exactly 0.0 six times in a row means it isn't defaulting to NONE, it isn't doing anything coherent on AT — something in that output path isn't getting trained at all.

**The sample outputs confirm mode collapse.** Look at the `L=` predictions across different queries: `L13, L13, L21, L13, L21...` and `O24, O60, O0, O24, O0...` — a small rotating set of tokens regardless of what was asked. That's a decoder that's stopped conditioning on the query/state and is just emitting whatever's cheapest under whatever gradient it does get.

**Root cause, most likely:** you added three new loss terms in this pass — deep supervision across all `loop_max=6` unrolled steps, the trajectory-based confidence loss, and the autoencoder loss (`ae_w=0.1`) — on top of task CE. You're only logging the *sum*. If deep-sup is summed rather than averaged across 6 loop steps × K=2 think steps, that term alone can be 10-15x the scale of a single task-CE loss, which would explain both the inflated total (220-268, vs baseline's normal-looking 4-40) and why the actual answer-token gradient is getting drowned out — the optimizer is mostly chasing loop-confidence/AE loss and barely touching the part that produces correct tokens.

What I'd do before changing anything else:

1. **Log each loss component separately** (task_ce, deep_sup_sum, conf_loss, ae_loss) per epoch. Right now you have one scalar hiding four different signals — you can't diagnose blind.
2. **Turn off deep-sup and AE loss, task CE only, unrolled loop still runs forward but no loss on intermediate steps.** Confirm the latent model can at least memorize the 360 train examples like the baseline does. If it can't even do that with just task CE, the bug is upstream of anything you added this round (state fold, loop mechanics, decode wiring) — not in the new training scheme.
3. Once (2) fits, **add back one term at a time** (AE, then deep-sup, then conf) and watch whether train loss stalls again — that tells you which term is the culprit rather than guessing.
4. If deep-sup is the problem, **average over loop steps, don't sum**, and consider only supervising a subsample of steps rather than all 6 — matches the compute-cost concern from before too.

Want to paste the loss computation code for this training loop? I can point at the exact spot if it's a sum-vs-mean issue.
