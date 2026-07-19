# Byte-Level Models: Research Overview

Papers covering token-free, byte-level, and patch-based architectures.

---

## Byte Latent Transformer (BLT) — Meta
- **arXiv**: [2412.09871](https://arxiv.org/abs/2412.09871)
- **Authors**: Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, et al. (Meta)
- **Date**: 2024-12-13
- **Categories**: cs.CL

**Summary**: The BLT architecture processes bytes directly but dynamically groups them into *patches* that serve as the primary computation units. Patch boundaries are determined by the entropy of the next-byte prediction — predictable spans form larger patches, surprising spans form smaller patches. This means compute is allocated based on data complexity.

Key results:
- Matches tokenized LLM performance at scale (8B params) for the first time
- Significant inference efficiency gains (fewer patches than tokens for the same content)
- More robust to input noise (no tokenizer to fail on)
- Patch sizes adapt: ~4.6 bytes/patch on average for English text

**Relevance**: This is the closest published work to the project's [`byte-state-byte.md`](byte-state-byte.md) architecture. The BLT's entropy-guided patching is exactly the mechanism proposed in [`byte-patch-preview.md`](byte-patch-preview.md). Key difference: BLT uses a transformer for patch-level computation; this project uses RWKV for the patch-level model.

**Open questions BLT doesn't answer** (that this project explores):
1. Can an RWKV (linear-time) patch-level model replace the transformer in BLT?
2. Can the encoder/decoder be recurrent (step-function or RWKV-based)?
3. Does entropy-guided patching work with recurrent state passing at byte level?

---

## MambaByte: Token-free Selective State Space Model
- **arXiv**: [2401.13660](https://arxiv.org/abs/2401.13660)
- **Authors**: Junxiong Wang, Tushaar Gangavarapu, Jing Nathan Yan, Alexander M. Rush
- **Date**: 2024-01-24
- **Categories**: cs.CL, cs.LG

**Summary**: MambaByte adapts the Mamba SSM to operate directly on raw byte sequences without tokenization. Key insight: SSMs have fixed-size memory state and linear-time decoding, making them naturally suited for the longer sequences that come from byte-level processing.

Key results:
- Competitive with subword-tokenized Transformers on language modeling
- 2.6× inference speedup using speculative decoding (tokenized draft, byte-level verify)
- No tokenizer bias, robust to noise

**Relevance**: Directly relevant to this project's core thesis — can a recurrent/SSM model learn efficiently from raw bytes? MambaByte proves yes. The project's RWKV-based byte models differ in using two-scale processing (encoder → patch-level → decoder) rather than MambaByte's direct byte-by-byte processing.

**The distinction**: MambaByte processes every byte equally (one forward pass per byte). This project's byte-state-byte architecture compresses byte spans into patches first, reducing the number of recurrent steps. BLT also uses patches, but with transformers. This project is unique in combining *patch compression* with *recurrent (RWKV) patch processing*.

---

## ByteFlow
- **arXiv**: [2603.03583](https://arxiv.org/abs/2603.03583) (2026)
- **Categories**: cs.CL

Adaptive byte compression without a tokenizer. Learns variable-length byte-to-embedding mappings dynamically. Related to the dynamic patching idea in `byte-state-byte.md`.

---

## Charformer
- **arXiv**: [2106.12672](https://arxiv.org/abs/2106.12672) (2021)
- **Authors**: Google

Fast character-level transformers using gradient-based subword tokenization. A precursor to the BLT line of work — showed that learning to group characters into latent subword units is feasible and efficient.

---

## HoloByte
- **arXiv**: [2603.16917](https://arxiv.org/abs/2603.16917) (2026)
- **Categories**: cs.LG

Continuous hyperspherical distillation for tokenizer-free modeling. Pushes the token-free approach further with representation-level distillation.

---

## Key Takeaway for the Project

1. **Byte-level works** (MambaByte, BLT) — the premise is proven at scale.
2. **Patches are the bridge** (BLT) — grouping bytes into patches before heavy computation is the right approach.
3. **RWKV for patch-level is novel** — no published work combines BLT-style patching with RWKV-style recurrence at the patch level. This project occupies a unique position.
4. **Dynamic patching is open** — whether boundary comes from entropy (BLT), fixed stride (current project), or learned gate is still an unresolved design question.

### Key gap this project can fill
"Can an RNN-based (RWKV) patch-level processor match or exceed a transformer patch-level processor when operating on BLT-style entropy-patched byte spans?" — No paper answers this.

