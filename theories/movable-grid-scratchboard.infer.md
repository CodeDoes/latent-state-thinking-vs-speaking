# movable-grid-scratchboard.infer.md

Interpretation of movable-grid-scratchboard.md.

What .md leaves implicit:

**Why 2D vs 1D scratchpad matters**: Linear CoT scratchpad (token stream) forces sequential dependency – reasoning step 2 must attend to step 1 via causal attention, O(L). 2D grid scratchpad allows *parallel* reasoning threads side-by-side (e.g., cell (0,0) = sum, (0,1) = product). Read can be patch 3×3, so step can read multiple previous threads at once. This is closer to human whiteboard – you place partial results around and later look at region.

**Movement as attention control**: In standard RWKV, attention is fixed decay per channel. Movable viewport is *learnable* attention position – you decide where to read next based on current text. This is like adaptive computation but in space, not depth. It turns passive reading (consume all bytes) into active foraging (decide which grid region informs next byte). At nano, this should improve sample efficiency on tasks where relevant info is spatially clustered.

**Text→position binding is parsing**: "A at 2,3" requires parsing numbers from byte stream. Byte-level RWKV can do this but needs to associate symbol 'A' with coordinate (2,3). This is similar to WHERE queries in exp001 (WHERE is A?). In exp001, WHERE accuracy was hard (0.015 baseline vs 0.318 latent). 2D version makes WHERE explicit as grid coordinate, not just list index.

**Movement differentiability challenge**: If Δpos is discrete up/down (argmax), not differentiable. Options:
- Soft: pos is continuous (x,y float), read via bilinear interpolation of 4 neighboring cells – differentiable
- Hard + REINFORCE: Δpos sampled from policy head, reward = QA accuracy at end, train via policy gradient (like RL) – more complex but allows discrete actions
- Supervised: give auxiliary loss pos should be near described coordinates – makes movement easy but not emergent
For first proof, use supervised auxiliary (teacher forcing for position) to isolate grid memory from movement learning. Then ablate to emergent (no aux) – two-stage proof.

**Scratchboard reuse circularity**: Output as scratchboard means model's output is also its future input (read). This is recurrent. If training uses teacher forcing for write (force correct symbol at position), model never learns to recover from its own errors. Need scheduled sampling: during training, sometimes read from model's own written grid (not ground truth). This is same issue as shared_state_unrolled_feedback_001 decoder feedback – we fixed it by feeding previous target embedding. Same fix applies here: for grid write, feed back previous predicted cell.

**Why Place&Query is minimal but captures all three**: 
- 8×8 grid = 64 cells, each can hold symbol – small enough for 70K params to memorize
- 10 placements = 10× move + write operations – needs movement
- Query random placement – needs to read from correct position (association)
- Grid after placements is scratchboard reused for query
If model solves this, all three primitives proven.

**Relation to diffusion-grid-terminal certainty trigger**: In MGRWKV, write gate g is certainty – only write when certain. This is same as τ threshold in diffusion terminal. So scratchboard write can be gated by certainty: don't overwrite grid with low-conf write. Prevents corrupting scratchboard.

**What final result could be (even if not fully defined)**:
- One path: terminal agent that builds mental map of file system from `ls` byte blocks, each file at 2D position (maybe directory tree as grid), moves viewport to navigate, writes edits to scratchboard grid that becomes new file content.
- Another: visual QA agent that reads alt-text block, maps it to position on rendered page image (represented as byte grid), moves to that position, answers.
- Both share same mechanism, only dataset differs.

**Cheap→expensive ordering for exploration (per ultimate.md)**:
1. M0 static vs M2 movable (no move ablation) – proves move needed
2. M1 no grid vs M2 grid – proves scratchboard needed
3. Full M2 vs text without positions – proves association parsing needed
4. 8×8 → 16×16 → 24×80 scale
5. Replace hard crop with soft bilinear read
6. Replace supervised movement with emergent REINFORCE

**Observable metrics needed beyond QA accuracy**:
- Grid reconstruction accuracy (how many cells correct)
- Position trajectory plot (does pos visit described positions? Visual ascii)
- Write gate g histogram (how often it writes)
- Read attention heatmap (which grid cells read most)

**Why this is trainable even if conceptually hard**: Because we can generate infinite synthetic data with ground truth world + ground truth pos trajectory. We have supervision for everything initially (teacher forcing), then gradually remove supervision (scheduled sampling) to let emerge. This is same curriculum as rnn_patch_002 phase-1→phase-2 but now with grid.

**Risk**: If we give too much supervision (force pos to be correct), movement not learned, just copied. Need to measure if movement head actually learns to predict Δ from h, not just memorize.

