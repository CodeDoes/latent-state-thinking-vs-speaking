# screen-viewport-zoom-pan.infer.md

Interpretation of screen-viewport-zoom-pan.md.

What .md leaves implicit:

**Context budget math**: Traditional LM: world 64×64=4096 tokens, context window 512 → must truncate, loses info, or fills. Viewport model: viewport 8×8=64 tokens per step, RWKV state carries history (say 128 dim). Per-step context = 64+state = small constant. Even if world 1000×1000=1M tokens, per-step still 64. Total tokens seen across 10 steps =640, but never simultaneously. This keeps context from filling.

**Scroll vs pan**: Scroll is 1D movement (y only) for text. Pan is 2D (x,y) for image/grid. Implementation same: pos (x,y) delta. For terminal history, H_world = number of lines (e.g., 1000), W_world = 80, viewport H_view=10, W_view=80, movement only in y (scroll up/down). For image, both x,y.

**Zoom vs quantization – subtle difference**:
- Zoom changes *coverage*: viewport covers larger world area. Zoom=4 means 8×8 viewport shows 32×32 world region downsampled to 8×8. Good for overview.
- Quantization changes *detail per cell*: quant=4 means each viewport cell is average of 4×4 world cells (mean pool). Even at same coverage, quant controls blur.
Both affect token count? No – viewport size fixed, so token count fixed regardless of zoom/quant. But information density changes: high zoom + high quant = very coarse overview (low detail, large coverage). Low zoom + low quant = fine detail, small coverage.

This two-dimensional control is exactly what image understanding needs: you described "control the zoom level and panning and also the quantization". Zoom=field of view, quant=resolution.

**Why this enables vision approximation**: Image 512×512 rendered as bytes (e.g., ASCII art or RGB bytes). At quant=8, each viewport cell averages 8×8 pixels → 8×8 viewport shows 64×64 region at 8× downsample. Model can spot coarse blob (e.g., box). Then it pans to blob center, zooms to 1, quant to 1, now viewport shows 8×8 region at full res – can see fine detail like symbol inside box. This is foveated vision, no CNN needed.

**Training signal for zoom/pan/quant**: How to learn? Options:
- Supervised: ground truth target pos and required zoom to see target. Give aux loss that predicted pos should be near target, zoom should be low when close. Easy for first proof.
- RL: reward when agent finds hidden symbol (QA correct). Movement actions sampled, REINFORCE. Harder but emergent.
For MG1→VZ1, we can start supervised (teacher force pos to move towards target) to isolate viewport benefit, then remove supervision for emergent.

**Read implementation detail**: `crop(W, center=pos, size=H_view*z)` then `downsample(factor=q)` via mean pool. Need to handle boundaries – clip crop to world bounds.

**Write for scratchboard**: World can be writable – agent edits file (terminal) or draws on whiteboard. Write same as MGRWKV: `W[y,x] = (1-g)*W + g*w`. For massive content case, world is mostly static (file system), but scratchboard could be separate grid that is always writable (whiteboard). We can have two grids: world (read-only, large) and scratchboard (writable, small) that persists. Simplest for first proof: world writable (agent can mark visited).

**Connection to diffusion-grid-terminal certainty trigger**: Write gate g can be gated by certainty τ – only write to world/scratchboard when prob>τ. Prevents overwriting world with low-conf.

**What final result even if whiteboard not developed**: Even without whiteboard, viewport with zoom/pan/quant alone solves "context filling with massive content". Whiteboard can be added later as extra writable grid that is not tied to world coordinates, but to reasoning traces.

**Cheap→expensive ordering**:
1. VZ1 movable vs fixed viewport on 32×32 with hidden symbol – proves scroll/pan keeps context small
2. VZ2 zoom control vs fixed zoom – proves zoom needed for detail
3. VZ3 quant control vs fixed quant – proves quant needed
4. VZ4 combined zoom+pan+quant on 64×64 box+symbol – proves image-like search
5. VZ5 scroll up for massive text 100-line file – proves original scroll idea for terminal

**Observable metrics**:
- QA accuracy (found symbol?)
- Steps to find (efficiency, fewer steps better)
- Trajectory: pos, zoom, quant over time – should show overview → pan → zoom in pattern if learned
- Context tokens per step (fixed 64) vs world size (4096) – shows budget

**Risk**: If zoom/quant not learned, model may stay at fixed zoom and fail. Need to ensure action space explored – use entropy regularization like adaptive-exit-entropy.

**Relation to B3D**: Triplet diffusion can train viewport content filling: b1 masked viewport at low res, b2 lossable at high res, b3 clean high res refresh. So B3D is training method for viewport model.

