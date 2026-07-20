# diffusion-grid-terminal

**RWKV with diffusion takes a grid of bytes as input and outputs a grid of bytes stochastically, triggers typing when full certainty.**

```
screen (H×W bytes) -> screen-rwkv (RWKV diffusion) -> screen' (H×W bytes)
-> trigger typing for cells where p(top1) > τ, else repeat diffusion step
```

This is the terminal-use improvement theory: instead of token-by-token LM generating characters for a terminal, model operates on **full screen buffer** as state.

## Why terminal?

Current LLM terminal use: LM generates token stream, terminal renders, no vision. If LM makes mistake mid-screen, it doesn't see its own rendering error until next tool call. Screen-as-grid closes loop: model sees entire screen including previous outputs, cursor, colors, etc.

- **Vision approximation**: terminal screen *is* a rendered image but represented as byte grid (e.g., 24×80=1920 bytes, plus ANSI codes). Grid RWKV learns visual patterns (borders, prompts, error highlights) without vision encoder.
- **Recurrent world understanding**: RWKV state carries across screens: `screen_t + state_t -> screen_{t+1} + state_{t+1}`. So model has temporal awareness of world (file edits, command outputs). This is recurrent understanding, not just next-token.
- **Stochastic diffusion typing**: instead of greedy typing next token, model diffuses entire screen in parallel, commits only cells where certainty = 1.0 (or >τ). This allows **self-correction**: low-certainty cells get re-diffused with full screen context. Triggers typing when full certainty – mimics human typing only when sure.

## Architecture (nano minimal)

Input: grid `G_t` [H,W] bytes, flattened to sequence [H*W] but with 2D positional encoding (row+col). Also carry RWKV state `S_t` from previous screen.

Model: RWKV nano (or AdaptiveLoop) as diffusion denoiser:

1. Mask random cells in G_t (e.g., 30%) -> `G_masked`
2. Triplet-block layout (from b3d-rwkv-nano): training sample = [G_masked (b1), G_masked (b2 lossable), G_clean (b3)] – same trick gives pseudo-bidirectional within screen.
3. At inference:
   - Start with current screen + masked unknown cells (cells to be typed)
   - Diffusion iter: run RWKV → logits per cell → commit cells where max prob > τ
   - Unmask committed cells, repeat until all committed or max iters
   - Now full certainty screen -> trigger actual key presses for committed cells (typing)
   - Next screen observed from terminal → next step with carried state S_{t+1}

Grid size for nano: 16×32=512 bytes (small enough for dim=64, 2 layers, 228K params). Terminal 24×80=1920 for larger.

## Three sub-claims in this theory

### 1. Reasoning traces
Model can learn to use screen grid as **scratchpad reasoning traces** – not just final output, but intermediate reasoning written to off-screen buffer or commented area. Because screen is recurrent, previous reasoning traces remain visible to next diffusion step. This is CoT but as **grid memory**, not token stream.

Test: task where model must store intermediate results in specific grid rows (e.g., compute path, leave breadcrumbs). If model learns to use those rows, trace is load-bearing.

### 2. Tool calling
Screen contains tool outputs (e.g., `ls`, `cat`). Model can learn to trigger tool via committed typing (e.g., typing `cat file\n` when certain). Tool output appears as next screen `G_{t+1}`. Since RWKV state carries, model learns **tool calling loop** as stateful policy, not prompt pattern.

Test: synthetic terminal env where screen shows prompt `$` and task file. Model must output `cat task.txt` → next screen shows task → model outputs solution. Reward if temporal chain works.

### 3. Temporal awareness
Because state `S_t` carries across screens, model has time sense: `screen_t != screen_{t-1}` but state remembers. Can answer "what was previous screen?" or "how many steps ago did file change?". This is missing from stateless AR models.

Test: after 5 screen transitions, ask model to reconstruct screen_{t-2}. If state carry enabled, accuracy > zero-init.

Single variable for each: diffusion-grid vs token-only baseline, stateful vs zero-init, trace-enabled vs trace-disabled grid region.

## Minimal experiments (single variable, matched params)

- **G1 grid diffusion vs token AR**: same RWKV dim=64, train to reconstruct masked 16×16 grid. Diffusion commits in parallel, AR predicts row-by-row. Win if diffusion needs fewer forward passes for same accuracy.
- **G2 stateful vs zero**: 3-step terminal chain, state carry across steps vs reset. Ask final screen contains info that requires memory of step 1. Win if stateful accuracy +0.2.
- **G3 trace load-bearing**: grid with dedicated trace rows vs without. Task requires remembering intermediate. Win if with-trace accuracy > without +0.15.
- **G4 certainty trigger**: stochastic commit τ=0.9 vs greedy always-commit. Measure if high-τ reduces errors by allowing re-diffusion.

This theory would improve terminal use because model types only when certain, sees its own screen rendering, and has temporal memory – approximating vision without vision encoder, recurrent understanding.
