# Research

ArXiv literature search results for the byte-level RWKV project. Each file covers one topic area, summarizing relevant papers with their arxiv IDs, key results, and connections to this project's theory files.

## Topic Index

| File | Coverage | Links to theories |
|------|----------|-------------------|
| [`rwkv_overview.md`](rwkv_overview.md) | RWKV core papers, RWKV-6/Finch, GoldFinch, VisualRWKV, DREAMSTATE, DeltaProduct, WuNeng | `core/rwkv.md`, `memory/delta-mem.md`, `memory/dendrite_growth.md` |
| [`byte_level_models.md`](byte_level_models.md) | BLT (Meta), MambaByte, ByteFlow, Charformer, HoloByte | `architecture/byte-state-byte.md`, `analysis/byte-patch-preview.md` |
| [`state_space_models_mamba.md`](state_space_models_mamba.md) | Mamba, Mamba-2, MambaByte, Bi-Mamba+, Differential Mamba | `core/rwkv.md`, `spatial/b3d-rwkv-nano.md` |
| [`adaptive_compute.md`](adaptive_compute.md) | DART, RAViT, Input-Conditioned Layer Dropping | `adaptive/adaptive-exit-entropy.md` |
| [`memory_associative.md`](memory_associative.md) | DeltaNet, HOLA, Simple Linear Attention, HRM-RWKV-Text, DREAMSTATE | `memory/delta-mem.md`, `memory/dendrite_growth.md`, `memory/rwkv-state-carry.md` |
| [`diffusion_rnn_hybrids.md`](diffusion_rnn_hybrids.md) | B3D-RWKV, DREAMSTATE, RDM, Recurrent Autoregressive Diffusion | `spatial/b3d-rwkv-nano.md`, `spatial/diffusion-grid-terminal.md` |
| [`modular_dendritic_networks.md`](modular_dendritic_networks.md) | LoRA, PathNet, Mixture of Experts, Progressive Networks, Net2Net | `archive/dendrite_memory.md`, `memory/dendrite_growth.md`, `memory/progressive-expansion.md` |

## How to Use

- Each file lists papers with: title, arxiv ID, authors, key results, and direct relevance to this project
- **Key Gaps** sections highlight what *isn't* covered by published work — potential novel contributions
- Theory files in `theories/` link to these research files in relevant sections

## How to Refresh

```bash
# Query a new topic
python src/arxiv_query.py -n 10 "all:<arxiv search query>"

# Save for inclusion
python src/arxiv_query.py -n 10 -o research/<topic>.json "all:<query>"
```

See [`src/arxiv_query.py`](../src/arxiv_query.py) for options.
