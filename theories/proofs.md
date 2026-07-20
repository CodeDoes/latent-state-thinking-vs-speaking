# Claim Ledger

<!--
LEDGER FORMAT (enforced by AGENTS.md, Loop step 8):
Every entry — supported OR refuted — records:
  **Claim**      one falsifiable sentence
  **Theory**     path to the spec + hypothesis id
  **Method**     what was run (experiment ids)
  **Baseline**   the comparison run + its key metric
  **Results**    measured numbers (params, accuracy, speed)
  **Prior art**  closest published method + the Δ against it
  **Commit**     hash; `git tag` via src/_tag.py for runs worth citing
Entries without a baseline number are drafts, not proof.
-->

## Supported

### BlackGoose Channel-Mix Replacement (2024-07-20)

**Claim**: A single `nn.Linear(2560, 2560)` (BlackGoose CMix) can replace the full RWKV-7 channel-mix FFN (time-mix shift + ReLU² + key/value projections) in the frozen g1g 2.9B model, with cosine similarity >0.80 to the original output.

**Method**: Offline distillation. Run frozen g1g on text → record (LN2, FFN_output) pairs for target layer. Train linear with MSE loss. No backbone involved during training.

**Results** (layer 0, 3104 samples, 2793 train / 311 val):
- **Val MSE**: 0.0032 after 500 steps (2s training)
- **Cosine Similarity**: 0.84 vs original FFN output
- **End-to-end**: Full model forward works at 2.28GB peak GPU

**Speed** (dim=2560, single token):
| Component | Params | Relative |
|-----------|--------|----------|
| Original FFN | 52.43M | 1× |
| BlackGoose | 6.55M | 0.125× (12.5%) |

**Files**:
- `src/g1g_blackgoose_nf4.py` — NF4 quantized backbone + BlackGoose replacement
- `src/gen_blackgoose_data.py` — generate (LN2, FFN_out) pairs
- `src/train_blackgoose_offline.py` — train Linear to match FFN
- `experiments/blackgoose_data/layer_0.pt` — generated dataset
- `layer_0_trained.pt` — trained weights

**Commit**: Pending

## Refuted

<!--
Refuted claims live here permanently — this is the project's memory of what
does NOT work. Before proposing anything, grep this section first.
Format is identical to Supported entries.

### <Title> (<date>)
**Claim** ...
**Theory** <path>
**Method** <experiment ids>
**Baseline** <run + metric>
**Results** <measured numbers vs prediction>
**Lesson** <what belief changed; what to try instead>
**Commit** <hash>
-->

(none recorded yet in the new ledger format — historical dead ends are in
`theories/archive/` only; port them here as they become relevant)
