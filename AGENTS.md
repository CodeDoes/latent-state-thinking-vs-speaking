# AGENTS.md

Clean local-experimentation scaffold for researching architectures that
separate **thinking** (latent-state computation) from **speaking**
(token generation).

## Method (philosophy)
See **`theories/ultimate.md`** (your words) and **`theories/ultimate.infer.md`**
(my interpretation). The rules that govern everything in this repo:

- Produce something **novel** by running **small** experiments.
- Machine learning can be done with a **smaller system**.
- Prove **one thing at a time** — single-variable ablations, matched params,
  matched compute, attributable cause.
- **Never** wait for emergent properties. Every claimed capability must come
  with a named mechanism and a controlled disablement that breaks it.

If an experiment doesn't isolate one property, it's not an experiment here.

## Status
See **`theories/status.md`** for live project state (last checkpoint,
blocking issue, next prove-one-thing experiment).

## Layout
```
src/             # your models, datasets, training loops (Python package)
theories/        # design rationales, experimental proposals, research notes
experiments/     # results (gitignored); one dir per run, e.g. experiments/exp001/
devenv.nix       # Python + torch environment (the only dependency manager)
.gitignore       # ignores experiments/, *.pt, __pycache__, etc.
```

## Theories
Design rationales and experimental proposals live under `theories/`.

- **`theories/<topic>.md`** — your own write-up: the claim, the reasoning, the
  architecture sketch, whatever you want to capture. When you send a message
  with a new idea or refinement, I will update this file with your *verbatim*
  words so the `.md` always reflects exactly what you said.
- **`theories/<topic>.infer.md`** — my *external interpretation* of the same
  theory. This file fills in latent assumptions, makes implicit connections
  explicit, and highlights what your `.md` leaves out (the intuitive leaps,
  the unstated constraints, the sensitivities, the things you *know* but
  didn't write down). It never repeats content already in your `.md`.

The split mirrors the core research question: separating **what you think**
from **what you say** — here applied to how we think about the architecture
itself.

**Read order** if you're new: `ultimate.md` → `rwkv.md` → `status.md`.
Other `theories/<topic>.md` files are incremental proposals, each one
proving one specific thing under the `ultimate` frame.

**Topical theories (one property per file):**
- `generation-loss.md` — train on logits of the model's own (greedy) generation, gradient only on wrong-token positions.

## Local workflow
1. Develop in `src/` — keep model, dataset, and training loop as separate modules.
2. Training must be **observable** (log loss, accuracy, samples per second to stdout + file).
3. Training must be **resumable** — save checkpoints that can be loaded and continued.
4. The training loop must support **constant improvement**: you can stop, tweak the data generator, and resume without starting over.
5. Every experiment records the **git commit hash** of the code that produced it, so results are traceable.
6. Trained models must be **easy to clear** — deleting `experiments/<exp_id>/` removes everything; no state leaks across experiments.
7. Commit progress frequently; never overwrite prior results.

## Environment
Managed by devenv (`devenv up`, or `direnv allow`). The interpreter lives at
`.devenv/state/venv/bin/python`. All dependencies are declared in `devenv.nix`
— no `requirements.txt`, no pip installs outside of it.
