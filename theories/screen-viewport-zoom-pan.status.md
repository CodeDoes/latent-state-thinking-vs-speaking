# screen-viewport-zoom-pan.status

Proof chain for viewport that keeps context small via scroll/zoom/pan/quant.

## Claims

- **VZ1** — *movable viewport beats fixed viewport on 32×32 world with hidden symbol, keeping per-step context 64 vs world 1024.* World 32×32 random with one hidden symbol A at random pos, viewport 8×8 fixed at (0,0) vs movable that can move Δx,Δy. Both ~120K params, 2k steps. Proven if movable QA acc >0.8 and fixed <0.2 (fixed only sees 6% of world). Status: **open**. Exp: `viewport_fixed_001` vs `viewport_movable_001`.

- **VZ2** — *zoom control beats fixed zoom on detail hidden at low res.* World 32×32 where symbol is 1×1 cell invisible when zoom=4 quant=4 (averaged away), visible at zoom=1 quant=1. Fixed zoom=4 fails, controllable zoom (can change z) succeeds. Proven if controllable acc > fixed +0.5. Status: **open**. Exp: `viewport_zoom_fixed_001` vs `viewport_zoom_control_001`.

- **VZ3** — *quantization control beats fixed quant on coarse-to-fine search.* World 32×32 with box border visible at quant=4 (thick) but symbol inside only at quant=1. Need to switch quant: overview q=4 to find box, then q=1 to find symbol. Fixed q=4 fails to see symbol, fixed q=1 fails to find box quickly. Controllable quant wins. Status: **open**. Exp: `viewport_quant_fixed_001` vs `viewport_quant_control_001`.

- **VZ4** — *combined zoom+pan+quant solves 64×64 image-like search with context budget 64 vs 4096.* World 64×64 with box + symbol inside. Viewport 8×8. Movable+zoom+quant controllable vs fixed. Proven if controllable finds symbol in <10 steps with acc >0.7, fixed fails. Also measure total tokens seen: controllable 64*steps vs fixed sees whole world 4096 at once – proves context kept small. Status: **open**. Exp: `viewport_combined_001`.

- **VZ5** — *scroll up for massive text keeps context small.* World 100×80 grid (100 lines file), viewport 10×80 (10 lines). Task grep says "TODO at line 75", agent starts at line 0, must scroll down to line 75. Fixed viewport (no scroll) fails, movable scroll succeeds. Proven if scroll acc >0.8 vs fixed <0.1. Status: **open**. Exp: `viewport_scroll_fixed_001` vs `viewport_scroll_movable_001`. This directly proves your original idea "keep context from getting filled, instead scroll up".

- **VZ6** — *zoom/pan/quant enables image understanding via controllable foveation.* Synthetic image 64×64 ASCII art with object, viewport 8×8, controllable zoom/pan/quant vs fixed. Measure if controllable can answer "what is inside box?" that requires zoom in. Proven if controllable acc > fixed +0.3. Status: **open**. Exp: `viewport_image_understanding_001`.

## Open follow-ups (cheap→expensive)

1. Add separate writable whiteboard grid (scratchboard) that persists across viewport moves, not tied to world coordinates – for reasoning traces
2. Combine with certainty trigger from diffusion-grid-terminal: only write/move when prob>τ
3. Real terminal capture: 1000-line bash history as world, viewport 20 lines, task find error
4. Real image: render MNIST or CIFAR as byte grid 32×32, test zoom/pan search
5. Scale world to 256×256 (65k tokens) with viewport 16×16 (256 tokens) – proves massive content handling
6. RL training for movement (REINFORCE) vs supervised teacher forcing – emergent vs forced

## Relation to other theories

- `movable-grid-scratchboard.md`: MG1 proved movable+scratchboard works at 8×8. This extends to larger world with small viewport + zoom/quant.
- `diffusion-grid-terminal.md`: GT1-GT6 are screen as grid, certainty trigger. This adds zoom/pan/quant controls to that screen.
- `b3d-rwkv-nano.md`: BD1 triplet can train viewport filling at multiple zoom levels in parallel
- `rwkv-state-carry.md`: state carry needed to remember overview while zoomed in (temporal awareness)
- `byte-state-byte.md`: encoder-state is viewport content, patch model is world?

## Current stage

User clarified: screen as viewport to avoid massive content filling context, plus zoom/pan/quant for image understanding. Whiteboard not yet developed. MG1 already proven (movable grid 1.0 acc). Next is VZ1 (movable viewport on 32×32).
