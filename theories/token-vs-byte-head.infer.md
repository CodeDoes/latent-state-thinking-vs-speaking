# token-vs-byte-head.infer

Interpretation of token-vs-byte-head.md.

What .md leaves implicit:

**Why this is the user's loop's one-variable delta from B5**: reports/user_theory_mapping.md says user's architecture differs from B5 in exactly two variables: decoder head vocab and patch boundaries. This theory isolates (1). It's the "ponytail recommendation" from that report.

**Detokenize-to-byte loop closure**: Full user loop requires detokenize predicted tokens → bytes → feed encoder. .md's minimal test trains token CE only, not the loop. Full loop adds exposure bias (teacher forcing vs autoregressive). The minimal test deliberately avoids loop to keep single variable; loop closure is follow-up T2.

**Byte_vocab vs BPE**: src/byte_vocab.py + tokenizer.py exist but BPE vocab size ~50k at large scale is too big for 228K model (head dominates). Using 1k vocab keeps head ~64k params, manageable. Trade-off: 1k BPE is still byte-ish, not semantic token, so effect underestimated.

**Evaluation nuance**: Byte model loss directly comparable to token model's detokenized byte loss only if detokenizer is deterministic. Need to compute byte-equivalent CE by projecting token logits to bytes via BPE decoder's byte representation (approx). Simpler: generate tokens, detokenize, then compute Levenshtein vs target bytes as secondary metric, while primary is still CE (token vs byte) – not directly comparable but trend tells story.

**What negative means**: If token head matches byte head, dense byte supervision is NOT necessary, and we can jump to token output for speed. That would unlock scaling to multi-byte tokens quickly.

**What to log**: token accuracy, byte recovery accuracy after detokenize, compression rho (tokens per byte for decoder), samples detokenized.
