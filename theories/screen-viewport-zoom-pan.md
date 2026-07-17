# screen-viewport-zoom-pan (scroll/zoom/pan/quant to keep context small)

**User refined idea:** "I see it as a screen and my original idea was some way to keep the model's context from getting filled with massive content. Instead of seeing everything it can scroll up. I also thought it would totally be able to understand an image like this if it was able to control the zoom level and panning and also the quantization."

This theory takes the movable-grid-scratchboard and makes it **explicitly about context budget**.

## Problem: massive content fills context

Traditional LM terminal use: cat huge file (10k lines) → all lines go into context → context window filled, no room for reasoning. Or image rendered as 1920 bytes (24×80) but high-res image 1024×1024 = 1M bytes – impossible.

Instead, you want agent to **control its own viewport** like a human:

- **Scroll up/down**: move viewport in y to see previous content without keeping all history in context. Context only contains current viewport (e.g., 10×80=800 bytes), not entire history (10k lines).
- **Panning**: move viewport in x,y over large 2D world (e.g., 64×64 grid world, or image). Like moving camera.
- **Zoom**: change coverage. Zoom out = see large area at low resolution (overview). Zoom in = see small area at high resolution (detail). Keeps token count fixed (viewport size fixed, but world coverage via zoom variable).
- **Quantization**: change detail level per cell. High quantization = coarse (e.g., 4×4 world cells averaged into 1 viewport cell, or byte → token BPE coarse). Low quantization = fine (byte-level). Also keeps token count fixed but controls information density.

Combined, agent can keep context small (e.g., 8×8 viewport = 64 tokens) even though world is massive (64×64=4096 tokens or image 1024×1024). It decides *what* to see next via actions, rather than seeing everything.

This is **foveated vision + scroll for text**, implemented as controllable read from external grid memory.

## Architecture: Viewport-Zoom-Pan-Quant RWKV

World grid `W` [H_world, W_world, D] – large, e.g., 64×64, could be file content, terminal history, or image rendered as byte embeddings.

Viewport `V` [H_view, W_view, D] – small, fixed token budget, e.g., 8×8=64.

State:
- pos (x,y) – viewport center in world coordinates
- zoom z ∈ {1,2,4,8} – scale factor: viewport covers `H_view*z × W_view*z` area in world, downsampled to `H_view×W_view`
- quant q ∈ {1,2,4} – pooling factor: average `q×q` world cells into 1 viewport cell (further downsample)

Read:
```
world_patch = crop(W, center=pos, size=(H_view*z, W_view*z))  # large patch
viewport = downsample(world_patch, factor=q)  # mean pool q×q → H_view×W_view
read_vec = flatten(viewport) + row/col pos embeds
```

Write (scratchboard):
- World can be writable too (agent edits file, or draws on whiteboard). Write is to world at pos with gate `g`, same as movable-grid-scratchboard.

Actions (output heads from RWKV hidden h):
- Δx, Δy ∈ [-1,0,1] – move
- Δz ∈ [-1,0,1] – zoom in/out (change z)
- Δq ∈ [-1,0,1] – quant up/down
- write vector w, gate g

All actions keep context fixed size (viewport), even though world massive.

Training: supervised or RL – given task "find symbol A hidden at (25,20) in 64×64 world", agent starts at (0,0) zoom=8 (overview sees whole world at low res, A invisible because quantized away). It must learn to: zoom out overview → detect coarse blob → pan to (25,20) → zoom in (z=1) → quant fine (q=1) → see A → answer.

## Minimal proofs (single variable, 70K–120K params, CPU 2k steps)

### VZ1: Movable viewport beats fixed viewport on large world

World 32×32 with one hidden symbol at random pos. Viewport 8×8 fixed at (0,0) vs movable (can move). Fixed cannot see symbol unless symbol happens to be in initial viewport (prob 64/1024=6%). Movable should learn to search and find with >80% accuracy within N=10 moves.

Proves scroll/pan keeps context small but still can find massive content.

### VZ2: Zoom control beats fixed zoom on detail hidden at low res

World 32×32 where symbol only visible at high res (e.g., symbol is 1×1 cell, but when zoom=4 and quant=4, that cell averaged with 15 others → invisible). At zoom=1 quant=1, visible. Fixed zoom=4 fails, controllable zoom should learn to zoom in after overview.

Proves zoom needed for image understanding with controllable level.

### VZ3: Quantization control beats fixed quantization

Similar: world has coarse structure visible at high quant (e.g., box border visible when quant=4 as thick line) but fine detail (small symbol inside box) only visible at quant=1. Agent needs to switch quant: overview q=4 to find box, then q=1 to find symbol inside.

Proves quantization control for vision approx.

### VZ4: Combined zoom+pan+quant solves image-like search

World 64×64 with box + symbol inside box (like visual QA). Viewport 8×8, starts zoomed out. Must: pan to find box (coarse), zoom in, quant fine, find symbol. Measures steps to find vs baseline sees whole world (1024 tokens) at once.

Proves context budget: viewport model uses 64 tokens per step × 10 steps = 640 total, but per-step context only 64, never fills massive 4096.

### VZ5: Scroll up for massive text (terminal history)

World is 100-line file (100×80=8000 bytes) as 100×80 grid. Viewport 10×80 (10 lines). Task: grep says "TODO at line 75", agent starts at line 0, must scroll down to line 75 to see TODO. Fixed viewport fails, movable (scroll) succeeds. Proves original idea "instead of seeing everything it can scroll up".

Each is single variable, matched params, synthetic infinite data.

## Why this works for massive content

Per-step context = viewport size (e.g., 64 tokens), constant. World size can be arbitrarily large (e.g., 10k lines, 1M pixel image) but never fully loaded. Agent controls what to load next via actions. This is like human reading: you don't keep entire book in working memory, you move eyes, turn pages, zoom.

RWKV state `S_t` carries history of what has been seen (temporal awareness from diffusion-grid-terminal), so even though current viewport small, model remembers coarse overview from previous zoomed-out view.

Quantization is key for images: you said "understand an image like this if it was able to control the zoom level and panning and also the quantization". Zoom+quant gives multi-scale vision: low quant coarse = low-res overview, high quant fine = high-res detail, but token count stays fixed.

## Connection to previous theories

- `movable-grid-scratchboard.md`: this is extension – adds zoom and quant to movement. MG1 proved movable+scratchboard works at 8×8. Now world larger, viewport small, movement is scroll, zoom/quant are new controllable.
- `diffusion-grid-terminal.md`: screen is viewport, certainty trigger τ is write gate, but now viewport can zoom/pan, not just commit.
- `b3d-rwkv-nano.md`: triplet diffusion can train viewport filling in parallel – B3D block = viewport content at different zoom levels.
- `rwkv-state-carry.md`: state carry needed for temporal awareness across scrolls – remember what was seen before.

## What final result could look like (even if whiteboard not developed)

Terminal agent that:
1. Sees current 10-line viewport (10×80=800 bytes) + RWKV state (memory of previous viewports)
2. Outputs action: scroll down 10, zoom out to overview, quant coarse to see file structure, pan to file list
3. World (file system) is large grid, but context never exceeds 800 bytes
4. When finds TODO, zooms in, quant fine, edits file (writes to world grid = scratchboard)
5. Triggers typing only when certain (from diffusion-grid-terminal)

For image: same but world is image rendered as byte grid (e.g., 512×512). Viewport 32×32, zoom controls downsample factor, quant controls color depth. Agent can pan to find object, zoom in to read text in image.

Whiteboard can be added later as separate grid that is always writable regardless of viewport – persistent reasoning traces.

## Next step as you asked to work together

We have MG1 proven. Now simplest next proof for your refined idea is **VZ1: movable viewport beats fixed on 32×32 world with hidden symbol**. This directly proves "keep context from getting filled with massive content – instead scroll up".

I can implement `src/viewport_zoom_pan_model.py` and `src/train_viewport_zoom_pan.py` now, ready to go, with modes fixed vs movable, zoom control.

Want me to build that as next experiment?
