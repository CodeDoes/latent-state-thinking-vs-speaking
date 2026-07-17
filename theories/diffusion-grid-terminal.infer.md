# diffusion-grid-terminal.infer.md

Interpretation of diffusion-grid-terminal.md.

What the .md leaves implicit:

**Why grid of bytes, not tokens**: Terminal screen is already byte grid (char per cell + maybe attributes). Representing as tokens (BPE) destroys 2D locality – `ls` output column alignment lost. Byte grid preserves spatial structure: neighbor bytes are spatially adjacent cells, not just sequence neighbors. RWKV's time_decay per channel can learn row vs column decay differently (long along row, short across newline).

**2D positional encoding needed?**: Flat sequence loses row/col. Need to add `row_embed + col_embed` to byte embed, or use `raster order + newline token = 10`. Simplest for nano: 16×32 grid, flatten row-major, keep `\n` as delimiter, so model sees row boundaries. At larger, add learned row/col embeddings.

**Certainty trigger as adaptive compute**: This is same as `adaptive-exit-entropy.md` gate: λ = certainty. Trigger typing when λ>τ is same as exiting diffusion loop. So this theory reuses adaptive exit mechanism but for *output space* (grid cells) not *depth*. Could share gate implementation.

**Vision approximation claim**: The .md says "would allow for vision approximation". Reasoning: terminal screen rendering (colors, borders) is visible as ANSI escape bytes in grid. If model learns to predict those bytes, it implicitly learns visual layout without CNN. This is byte-level vision – same as `byte-state-byte` but now 2D. Need to test: can model predict box-drawing characters that depend on 2D structure? If yes, vision approx proven.

**Recurrent understanding of world**: Screen_t -> model -> screen_{t+1} -> env -> screen_{t+2} loop is world model. RWKV state is world state. This is `shared_state_unrolled_feedback_001` fix applied to env loop: previous decoder feedback is previous screen embedding, not static. So this theory depends on that fix – without feedback, decoder would stall exactly like B3.

**Reasoning traces as grid scratchpad**: Traditional CoT is linear token stream that gets discarded. Grid trace is persistent: reasoning written to e.g., rows 12-15 stays visible for future steps. This is "NIAH but for reasoning". Testable via ablation: erase trace rows each step vs keep. If keep wins, traces load-bearing.

**Tool calling as certainty-triggered typing**: In terminal env, tool call is string `cat file\n` typed into prompt. In diffusion grid, tool call is committed when model certain about entire command string (all cells in command row >τ). This prevents partial tool calls (typing `c` then hallucinating). Certainty trigger ensures atomic tool call.

**Temporal awareness testable via state carry**: `rwkv-state-carry.md` already proves state carry matters for long-horizon. Grid terminal adds *screen* carry: screen_{t-1} visible in state? Actually RWKV state should encode previous screens. Test: ask model to detect change between t and t-1. Needs state.

**Why diffusion not AR for grid**: AR would type grid cell-by-cell left-to-right, cannot self-correct earlier typo after seeing later cells. Diffusion allows parallel self-correction: if model types `ls` then sees file list contains `main.py` but typed `main.px`, later diffusion iteration can correct `x`→`y` because whole grid context visible. This is critical for terminal where full screen context matters.

**What negative means**: If grid diffusion needs more steps than AR row-by-row for same accuracy, diffusion not beneficial at nano – suggests 2D structure needs larger capacity or different positional encoding. Still informative.

**Cheap to expensive ordering**: G1 (grid recon) cheap, G2 (stateful chain) medium, G3 (trace) medium, G4 (full terminal env loop with tool calling) expensive. Should prove in order.

**Connection to B3D**: B3D triplet layout is exactly how to train grid diffusion: masked screen b1, lossable b2, clean b3. So b3d-rwkv-nano is substrate for diffusion-grid. Implement B3D first, then extend to 2D grid by adding row/col embeddings.

**Dataset for nano**: Synthetic terminal screens: generate random `ls` like output (file list), random prompt, random file content, mask some cells, task reconstruct. Easy to generate procedurally – infinite data, no overfitting, per `rwkv.md` frame.

**Safety**: Typing triggered only at full certainty prevents model from spamming terminal with low-conf commands – safety via certainty threshold.

