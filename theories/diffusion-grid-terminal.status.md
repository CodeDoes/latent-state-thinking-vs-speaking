# diffusion-grid-terminal.status

Proof chain for diffusion grid terminal, reasoning traces, tool calling, temporal awareness.

## Claims

- **GT1** — *grid diffusion reconstructs masked 16×16 byte screen at matched params vs row-by-row AR, with fewer forward passes.* Same RWKV dim=64, mask 0.3, 2k steps. Diffusion commit τ=0.9 parallel vs AR sequential. Proven if diffusion loss ≤ AR +0.1 and avg iterations < 16 (row count) i.e., parallel win. Status: **open**. Exp: `grid_diffusion_ar_001` vs `grid_diffusion_diff_001`.

- **GT2** — *stateful carry across screens enables 3-step terminal chain memory.* Synthetic env: screen1 shows task file name, screen2 shows file content after `cat`, screen3 question needs file name. Zero-init vs stateful carry. Proven if stateful accuracy > zero +0.2. Status: **open**. Exp: `grid_state_zero_001` vs `grid_state_carry_001`.

- **GT3** — *reasoning trace rows are load-bearing.* Grid 16×16 with rows 10-13 reserved as trace scratchpad (model can write intermediate). Task requires multi-step calc (e.g., sum list, keep running total in trace). Compare trace-enabled vs trace-erased-each-step ablation. Proven if with-trace accuracy > without +0.15. Status: **open**. Exp: `grid_trace_enabled_001` vs `grid_trace_disabled_001`.

- **GT4** — *certainty trigger τ=0.9 reduces tool-calling errors vs greedy.* Terminal task: model must type `cat file.txt\n`. Diffusion with τ=0.9 commits only when full command certain; greedy commits first token always. Measure incorrect command rate. Proven if high-τ error rate < greedy -0.2. Status: **open**. Exp: `grid_certainty_09_001` vs `grid_certainty_greedy_001`.

- **GT5** — *temporal awareness: model can reconstruct screen_{t-2} from state.* After 5 screen transitions with state carry, probe decoder to generate previous screen from current state. Zero-init baseline can't. Proven if stateful reconstruction accuracy > zero +0.15. Status: **open**. Exp: `grid_temporal_probe_001`.

- **GT6** — *vision approximation via byte grid: model learns 2D box-drawing.* Task: screen contains box border made of `+-|` chars that must align rows. Mask interior of box. If model reconstructs border correctly (needs 2D structure) vs baseline that scrambles, proves 2D spatial learning. Status: **open**. Exp: `grid_vision_box_001`.

## Open follow-ups (cheap→expensive)

1. Add ANSI color bytes to grid – test if model learns color semantics
2. Real terminal dataset: capture actual `~/` ls/cat sessions, train diffusion to predict next screen
3. Combine with injection-frequency per-layer fusion (global screen context re-injected each layer)
4. Scale to 24×80 real terminal (1920 bytes) at 1M params
5. Full RL loop: model types, real bash executes, next screen observed, reward on task completion – tool calling emergence

## Relation to other theories

- `b3d-rwkv-nano.md` – BD1-4 are prerequisite (triplet-block diffusion substrate)
- `rwkv-state-carry.md` – S1 is prerequisite for GT2/GT5 (state carry needed for temporal)
- `injection-frequency.md` – per-layer fusion may help grid global context
- `byte-state-byte.md` – byte→state→byte is same as screen→state→screen at higher dimension
