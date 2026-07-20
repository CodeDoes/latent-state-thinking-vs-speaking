# Prior Art Map

**Read this before designing anything.** Every theory doc must name previous
attempts at the same problem (AGENTS.md, step 2). This file is the curated
starting point; the detailed paper summaries live in the per-topic files of
`research/` (see `research/README.md`), and new queries go through
`python src/arxiv_query.py`.

How to use it: find the row whose *Problem* matches yours. That row (plus the
topic file it links) is your minimum citation set. If a method here already
solves your problem, your theory must be framed as a **reimplementation**,
**simplification**, **budget reduction**, or **falsification** of it — and it
becomes your baseline.

---

## 1. Tokenizer removal / byte-level modeling

| Method | Ref | What they did | Relevance here |
|--------|-----|----------------|----------------|
| **BLT** (Byte Latent Transformer, Meta) | arXiv:2412.09871 | Entropy-gated dynamic patching + local byte encoder/decoder around a large latent transformer | Direct prior art for our patch-based byte front-end; our entropy/dynamic-patch theories are budget-reduced BLTs. See `research/byte_level_models.md` |
| **MegaByte** | arXiv:2305.07185 | Fixed patches + two-level transformer | The "fixed patch" ablation every dynamic scheme must beat (`dyn_patch_fixed_*` runs) |
| **MambaByte** | arXiv:2401.13660 | Pure SSM over raw bytes, no patching | Shows byte-level without patching is viable; cites in `research/state_space_models_mamba.md` |
| **ByT5** | arXiv:2105.13626 | Token-free T5 on UTF-8 bytes | Canonical "just use bytes" baseline; heavy compute cost is the motivation for patching |
| **CANINE** / **Charformer** | arXiv:2103.06874 / arXiv:2106.12672 | Downsampling byte/char encoders feeding a transformer | Design source for our encoder→patch interface (`threads/byte_state_byte/`) |
| **MrT5** | arXiv:2401.14740 | Learned merge gate deleting tokens at runtime | Prior art for gate/merge decisions; compare gate sparsity behavior |
| **SpaceByte** | arXiv:2404.08335 | Byte-level with space-aligned patching | Alternative patching heuristic to entropy |
| **T-FREE** | arXiv:2406.19223 | Tokenizer-free sparse embeddings over char trigram hashes | Prior art for hash-based lookup like `src/hybrid_tokenizer.py` |

## 2. Adaptive compute ("thinking more where needed")

| Method | Ref | What they did | Relevance here |
|--------|-----|----------------|----------------|
| **ACT** (Adaptive Computation Time) | arXiv:1603.08983 | Learned halting per position | The origin of every loop/halting gate we build |
| **Universal Transformer** | arXiv:1807.03819 | Recurrent shared-weight depth | "Loop the block" prior art; baseline for shared-state loops (`src/shared_state_*`) |
| **Mixture-of-Depths** | arXiv:2404.02258 | Top-k routing: which tokens get full compute | Static budget version of our adaptive-exit theories (`threads/adaptive_compute/`) |
| **PonderNet** | arXiv:2107.05407 | Probabilistic halting with geometric prior | Loss design for our entropy/lambda on gates |
| **CALM** / early-exit classifiers | arXiv:2207.07061 | Confidence-based early exit | Prior art for `adaptive-exit-entropy.md` |
| **LayerSkip** | arXiv:2404.16710 | Dropout over layers + early-exit head | Engineering reference for exit supervision |

## 3. Latent reasoning / looped "thinking vs speaking"

| Method | Ref | What they did | Relevance here |
|--------|-----|----------------|----------------|
| **COCONUT** (Chain of Continuous Thought) | arXiv:2412.06769 | Feed last hidden state back as next input embedding | Closest published relative of "latent-state thinking"; if a theory claims latent reasoning improvements, this is the comparison |
| **HRM** (Hierarchical Reasoning Model) | arXiv:2506.21734 | Two-timescale recurrent loops (H/L modules) in latent space | Neighboring architecture to our encoder↔decoder loops |
| **Looped Transformers / weight-tied recurrence** | arXiv:2309.12427 | Iterated weight-tied blocks emulate iterative algorithms | Theory anchor for our loop models' expressivity claims |
| **Quiet-STaR** | arXiv:2403.09629 | Learned "pause tokens" / internal rationale tokens | Prior art for inserting compute without emitting tokens |

## 4. Memory & architectural growth

| Method | Ref | What they did | Relevance here |
|--------|-----|----------------|----------------|
| **DeltaNet / delta rule fast weights** | arXiv:2102.11174 | Associative memory updated by delta rule | Direct parent of `threads/delta_mem/delta-mem.md` and RWKV-7's WKV update |
| **kNN-LM** / retrieval LM | arXiv:1911.00172 | Non-parametric memory at inference | The "store, don't grow" alternative to dendrite growth theories |
| **Memory Layers** (Meta) | arXiv:2412.09764 | Trainable key-value memory at scale | Scaled version of adapter-memory ideas; cite in dendrite theories |
| **LoRA / adapters** | arXiv:2106.09685 / arXiv:1902.00751 | Frozen trunk + tiny trainable branches | What our dendrite work budget-reduces; `research/modular_dendritic_networks.md` |
| **PathNet** / **Progressive Networks** / **Net2Net** | arXiv:1701.08734 / 1606.04671 / 1511.05641 | Architectures that grow with new tasks | Prior art for `progressive-expansion.md` and `dendrite_growth.md`; the comparison is cost-per-new-skill |
| **MoE (sparse)** | arXiv:1701.06538 | Routed expert growth | The routed-vs-grown memory axis |

## 5. Module replacement / distillation (g1g front-end work)

| Method | Ref | What they did | Relevance here |
|--------|-----|----------------|----------------|
| **Knowledge Distillation** | arXiv:1503.02531 | Train small net to match big net's outputs | The frame for every "small learned replacement" claim (BlackGoose, tokenizer, layer-0) |
| **TinyStories / GPT-nano distills** | arXiv:2305.07759 | Small models memorize narrow distributions | Why our probes use synthetic tasks: isolate capability from data difficulty |

---

## Empty corners worth claiming (check first!)

These looked uncovered as of the last research pass — verify with a fresh
arxiv query before claiming novelty:

- Entropy-gated patching feeding an **RWKV** (not transformer) core.
- **Growth keyed by measured bottleneck** (loss-probe triggered) rather than
  task boundaries.
- Full byte front-end for a **frozen retail RWKV-7** that reproduces its
  tokenizer exactly at the state level.

## Changelog

- 2026-07-20 — created from `research/*.md` topic files. Extend per-topic
  files first, then mirror key rows here.
