# token-vs-byte-head

Architecture maps bytes → bytes with vocab=258. User's proposed loop (reports/user_theory_mapping.md) suggests decoder could output **tokens** (BPE, ~1k-50k vocab) instead of bytes, then detokenize to bytes for loop closure.

Hypothesis: byte head's dense supervision makes B5 work; token head will degrade byte-equivalent recon at small scale (~228K) because token prediction is sparser and vocab larger.

Why token head might help: tokens are concept-level; predicting 1 token = ~4 bytes compressed, so decoder does less work per semantic unit. At scale, faster.

Why byte head might be necessary: at nano scale, every byte position gives gradient; token head gives gradient only at token boundaries (~4x less signal). B5's success relied on dense byte CE.

Minimal test (single variable = head vocab):
- Shared: AdaptiveLoopModel encoder+core identical, dim=64, 2+2 layers
- Arm A: byte head 258 (existing), input bytes, target bytes
- Arm B: token head 1024 using byte_vocab.py BPE trained on same text.txt, input bytes → encoder, decoder outputs tokens, loss = token CE, but we also compute byte-equivalent loss by detokenizing greedy prediction and measuring byte CE post-hoc (or just compare token CE vs byte CE trend)
- Matched params: adjust head size difference by shrinking dim slightly in B (token head larger: 64*1024 vs 64*258 = ~49k extra params, so reduce dim 64→60 to compensate). Or add dummy linear to A for fair count.
- Train 2k steps, same data stream (byte stream tokenized for B).

Win condition for byte hypothesis: A byte CE (or detokenized byte CE for B) lower than B by >=0.1.

Single variable: output vocabulary.
