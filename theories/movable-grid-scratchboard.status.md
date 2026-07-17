# movable-grid-scratchboard.status

Proof chain for movable grid memory + scratchboard + text→2D association.

## Claims

- **MG1** — *movable grid with scratchboard beats static grid and no-grid baselines on Place&Query 8×8 task.* Task: 10 random placements "A 2 3;" etc in 8×8 world, query random placement. Models: M0 static pos (0,0) no move, M1 no grid (RWKV state only), M2 full movable + scratchboard, all ~70K params, 2k steps. Proven if M2 QA accuracy >0.8 and M0/M1 <0.3. Status: **open**. Exp: `movable_grid_static_001`, `movable_grid_nogrid_001`, `movable_grid_full_001`.

- **MG2** — *text→position association parsing is load-bearing.* Same M2 but text without positions ("A B C" only) vs with positions. Proven if with-pos accuracy > without-pos +0.3. Status: **open**. Exp: `movable_grid_nopos_001`.

- **MG3** — *movement trajectory correlates with described positions (emergent navigation).* Measure pos trajectory vs ground truth described positions. If supervised movement aux loss removed, does pos still visit near described positions? Proven if distance between pos and described pos <2 cells avg without aux loss. Status: **open**. Exp: `movable_grid_emergent_move_001`.

- **MG4** — *scratchboard reuse: write then later read same cell works.* Task: write "A" at (2,3), later overwrite with "B" at same pos, query should return "B" (most recent). Tests if grid correctly overwrites and reads latest. Proven if overwrite accuracy >0.9. Status: **open**. Exp: `movable_grid_overwrite_001`.

- **MG5** — *2D grid beats 1D scratchpad (linear CoT) on parallel reasoning.* Task: two independent sums in different grid regions (e.g., sum in left half, product in right half). 2D grid can hold both side-by-side, 1D linear scratchpad must interleave. Proven if 2D accuracy >1D +0.15 at matched params. Status: **open**. Exp: `movable_grid_2d_vs_1d_001`.

- **MG6** — *certainty-gated write prevents corruption (from diffusion-grid-terminal).* Add write gate threshold τ=0.9 – only write when prob>τ. Compare to always-write. On noisy placements (10% wrong placements), gated should preserve correct cells better. Proven if gated grid accuracy > ungated +0.1 under noise. Status: **open**. Exp: `movable_grid_certainty_001`.

- **MG7** — *scales to real terminal 24×80 with file editor viewport.* File 100 lines, viewport 10×80, agent must move viewport to TODOs described in grep output block. Proven if can fix ≥1 TODO via movable + scratchboard at 1M params. Status: **open, expensive**.

## Open follow-ups (cheap→expensive)

1. Soft bilinear read/write vs hard crop – differentiability tradeoff
2. REINFORCE for discrete movement vs supervised auxiliary
3. Combine with B3D triplet diffusion: parallel grid filling instead of sequential byte-by-byte
4. Add row/col 2D pos embeddings (from diffusion_grid_model) to improve spatial
5. Real terminal capture dataset – ls/cat sessions rendered as grid
6. Scale to code editing: file as grid, function as block of bytes, move to associate

## Relation to other theories

- `b3d-rwkv-nano.md`: BD1-3 needed for parallel grid diffusion training
- `diffusion-grid-terminal.md`: GT1-GT6 are prerequisite (certainty trigger, vision box)
- `rwkv-state-carry.md`: S1 needed for movement to be history-dependent across placements
- `injection-frequency.md`: per-layer grid re-injection may help read
- `byte-state-byte.md`: B1 encoder-state is same as grid state but 1D

## Current exploration stage

Raw idea → reframed into 3 primitives → minimal Place&Query task defined → M0/M1/M2 ablations designed → ready to implement src/movable_grid_model.py

We don't have final result fully defined, exploring via proofs MG1-MG7 in order.
