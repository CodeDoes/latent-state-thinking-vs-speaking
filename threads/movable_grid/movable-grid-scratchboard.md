# movable-grid-scratchboard

**User raw idea:** "agent was able to read a block of text or bytes and associate it with a 2d grid position. my idea was that the agent could move the grid around. my other idea was that the agent could use its output as an scratchboard. this is not the easiest conceptual thing to understand or think about training. i don't even have a complete idea of the final result in my head."

This file takes that raw idea and turns it into **realistic, small, provable steps**.

## Reframing the idea in three primitives

Your idea contains three primitives that are usually separate in literature, but you want them together:

1. **Text/Byte → Position Association**: Read a byte stream (e.g., "A at 2,3") and bind semantics to a 2D coordinate (2,3). Not just next-token, but *where* in an external map it lives.

2. **Movable Grid / Viewport**: The agent has a position `(x,y)` on a larger grid/world. It can output movement `Δx,Δy`. Its read is from current position (or neighborhood). This turns reading from passive (read all) into active (decide where to look next). This is "move the grid around".

3. **Output-as-Scratchboard (Writeable Memory)**: The grid is not fixed – it's the agent's own output that persists. Agent writes vectors to grid cells `G[y,x] = (1-g)*G + g*w`. Later steps read from same grid. This is external working memory, but 2D, not linear CoT. It's a scratchboard.

Combined: **Movable Grid Memory RWKV (MGRWKV)**: RWKV core reads byte, reads from grid at pos, writes to grid at pos, moves pos.

```
byte_t + read(G, pos_t) -> RWKV -> [write_vector w_t, write_gate g_t, move Δpos, answer_logits]
G_{t+1}[pos_t] = (1-g_t)*G_t[pos_t] + g_t*w_t
pos_{t+1} = clip(pos_t + Δpos)
```

`G` after processing is scratchboard. For QA, query position is given, read from G.

## Why this is different from byte-state-byte and diffusion-grid-terminal

- `byte-state-byte`: state is vector, not 2D grid. No explicit move.
- `diffusion-grid-terminal`: grid is screen buffer, diffusion fills in parallel, but movement is only commit threshold – no explicit navigable position.
- `movable-grid-scratchboard`: grid is *memory*, position is *controllable*, association is *explicit* text→position binding. It is closer to NTM/DNC but 2D and RWKV-based, and scratchboard is readable/writable 2D.

## Realistic applications (where final result could land)

We don't need final result fully defined – we can explore via small tasks that share same primitives, each useful on its own:

### A) Terminal file editor (closest to your terminal theory)

File is 24×80 grid (lines × cols). Agent reads byte block (e.g., `cat file.txt` output as text). It must associate file content with grid positions (line numbers). It can move viewport up/down (scroll) via Δpos. It uses output as scratchboard – writes edited version to same grid. Trigger typing when certain (from diffusion-grid-terminal).

Real task: "File has 100 lines, viewport is 10×80, agent sees `grep TODO` results as text block, must move to each TODO line and write fix". This proves text→position + movable + scratchboard in real terminal.

### B) Visual document QA (vision approximation without vision encoder)

PDF/page rendered as byte grid (char per cell). Text block is question "Where is the total amount?". Agent must associate question with 2D position (e.g., bottom-right table). Moves grid to that region, reads, writes answer to scratchboard cell (0,0). This is vision approx because grid *is* rendered image as bytes.

### C) Code / world-map navigation

World 8×8 with symbols A-Z. Text stream says "Place A at 2,3; Place B at 5,1;... What at 2,3?". Agent must parse text→position and build grid in scratchboard, then move to queried position to answer. This is minimal synthetic that isolates primitive 1+2+3 without terminal complexity.

### D) Computation scratchboard (reasoning traces as grid)

Text: "A=5 B=7 C=A+B". Agent reads, must compute intermediate 12, write to scratchboard position (0,0), later query reads from there. Grid provides parallel scratchpad: different computations can be in different cells side-by-side, not linear chain. This tests output-as-scratchboard reuse.

All A-D share same architecture, only dataset generator changes.

## Minimal first proof (single-variable, matched params, CPU 2k steps)

**Task C – Place & Query – is the smallest that captures all three primitives at once.**

Generate random world 8×8, symbols A-Z.

Input text (byte stream): `"A 2 3; B 5 1; C 0 7; ..."` – 10 placements. Each placement is `"{symbol} {x} {y};"` as bytes.

Model: MGRWKV
- H=8,W=8,D=32, dim=64, 2 layers, 70K params
- pos starts (0,0)
- For each byte/block token, RWKV reads byte + read(G,pos) (3×3 patch around pos), outputs write w,g, move Δ (tanh scaled to [-1,1] rounded)
- After processing all placements, grid G should contain symbols at described positions
- Query: "Q 2 3?" – we give query position (2,3) as additional input, model reads G[2,3] and outputs symbol logits

Losses:
- Grid reconstruction loss: CE between G cells and ground truth world (if we make G discrete via quantization) – or MSE if G stores embeddings
- QA loss: CE answer correct
- Movement supervision (optional): we can give auxiliary loss that pos trajectory should visit described positions – but better to let emerge without supervision, then analyze trajectory

Baseline ablations (prove one thing at a time):
- **M0 static grid no move**: pos fixed at (0,0), read only from (0,0), write to (0,0) – cannot solve, should fail. Proves move needed.
- **M1 no scratchboard**: no grid, only RWKV state – like exp001 (think-once). Should fail on 8×8 with many items because state alone cannot store all positions exactly.
- **M2 movable + scratchboard (full)**: should succeed, QA accuracy >0.8 vs M0/M1 ~0.1
- **M3 text→position parsing ablated**: input text without positions (just symbols list) – should fail, proves association parsing is load-bearing.

This ordering is cheap→expensive per `ultimate_thesis.md`.

If M2 wins at 70K params, we have proven text→2D association + movement + scratchboard works at nano – not emergent.

## How to train (realistic, observable, resumable)

Dataset generator infinite (like logic_niiah_generator):
- Random world 8×8, random placements, serialize to text
- Infinite variation → no overfitting
- Generator logs trajectory: ground truth pos sequence that optimal agent would follow (visit each described position)
- Training observable: log grid accuracy, QA accuracy, pos trajectory plots (save as ascii art), steps/s

Training loop:
```
for step in 1..2000:
  world, text = generator.generate()
  pos = (0,0), G = zeros
  for byte_chunk in text_chunks:
    read = crop(G, pos, 3x3)
    h, state = rwkv(byte_chunk + read, state)
    w,g,Δ = heads(h)
    G[pos] = (1-g)*G[pos] + g*w
    pos = clip(pos+Δ)
  # after text, query
  q_pos = random described pos
  read_q = crop(G, q_pos)
  ans_logits = qa_head(read_q)
  loss = CE(ans, truth) + λ*MSE(G, world) (optional)
  backprop
```

Note: read/write via hard crop is not differentiable for Δ (movement). To make differentiable, use soft bilinear read/write (like NTM) or straight-through with REINFORCE for Δ. For nano first proof, we can use hard movement but train movement via auxiliary loss (MSE between pos and described pos) – simpler.

For full differentiable, use soft: `read = Σ_{y,x} G[y,x] * kernel(pos,y,x)` where kernel is e.g., Gaussian centered at pos or 3×3 bilinear.

Scratchboard differentiability: write is differentiable (g*w) – OK.

## Connection to previous batch

- `b3d-rwkv-nano`: triplet layout can be used to train grid filling in parallel (fill whole 8×8 grid at once via diffusion, not sequential)
- `diffusion-grid-terminal`: screen is grid, movement is scroll, scratchboard is screen buffer – MGRWKV is generalization where grid is memory, not just screen
- `rwkv-state-carry`: state carry across placements = temporal awareness, needed for movement to be history-dependent
- `injection-frequency`: per-layer re-injection of grid read (not just front) may help – grid context needed each layer

## What we still don't know (open, not blocking first proof)

- What should final grid size be for real terminal? 24×80=1920 cells × D=32 = large memory but still manageable
- Should movement be discrete (up/down/left/right) or continuous Δ? Discrete easier to interpret, continuous differentiable
- Should scratchboard be discrete symbols or continuous vectors? Discrete for QA, continuous for reasoning traces – we can quantize via VQ or use both (logits head for discrete, vector for continuous)
- How to trigger typing? Certainty threshold from diffusion-grid-terminal still applies: only write to grid when g>τ and prob>τ
- How to scale to real files? Need real terminal capture dataset (you have `experiments/byte_ts_001/text.txt` – could render as grid)

## Proposal for you

Let's **build M0/M1/M2 as three small models in src/movable_grid_model.py** and run Place&Query 8×8 task at dim=32 smoke (70K params, 500 steps). If M2 wins, we have proof that your intuition works.

Then we iterate:
- If you like terminal direction (A), next exp is file editor viewport 10×80
- If you like vision direction (B), next exp is box-drawing vision but now with movable viewport (not fixed)
- If you like computation direction (D), next exp is A+B=C with trace rows

We can keep theory file open-ended – final result not needed, we explore via proofs.

Your raw idea is essentially **2D NTM with RWKV and byte text grounding** – small proof first, then scale to terminal.

Want to start with Place&Query 8×8 movable grid experiment? I can implement it now as `src/train_movable_grid.py` ready to go.
