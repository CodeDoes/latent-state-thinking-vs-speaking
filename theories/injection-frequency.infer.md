# injection-frequency.infer

Interpretation of injection-frequency.md.

What the .md leaves implicit:

**Why front fusion could be enough**: If encoder_out already contains local byte context, adding global once might be sufficient because subsequent RNN layers preserve information via hidden state. The RWKV core state is already in the same dim as encoder; residual connections should carry it. So the alternative hypothesis is "per-layer is bloat".

**What matched-params means here**: Per-layer fusion needs extra parameters (gate W). To keep total params equal, we must shrink something else (e.g., reduce core layers hidden_scale from 4→3, or cut dec_layers dim by 2). The .md suggests subtracting dim elsewhere. The cleanest: keep all dims identical, but the *front* variant gets an extra Linear to match gate param count, unused functionally but counted. That isolates compute graph shape vs capacity.

**Why this is the #2 open follow-up from byte-state-byte.status.md**: decoder-ablations.md explicitly notes "What we need to test is interleaving patch context through every block vs fusing once at the front." This theory is that test, made concrete with adaptive_loop as substrate rather than shared_state.

**Failure modes not in .md**: 
- Per-layer could cause double-counting of global signal, leading to mode collapse (B4-like eee...). Need to gate.
- Front fusion's success in B5 (loss 0.47) already suggests decoder uses 1 loop; per-layer might cause decoder to start using more loops (measurable via exit stats).
- If both variants hit identical loss, claim negatively proven, but we still learn that decoder depth doesn't need global re-injection (useful compression).

**What to log**: per-layer gate mean activation, per-layer exit_lambdas, final loss, samples, compression ratio rho. Gate mean near 0 = model learned to ignore per-layer injection (self-negation cue).
