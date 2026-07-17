# token-vs-byte-head.status

Proof chain for output head vocabulary.

## Claims

- **T1** — *byte head (258) beats token head (1024) at matched ~228K params on dense byte recon.* Arm A byte CE vs Arm B detokenized byte CE (or token CE trend). Proven if A loss < B_equivalent -0.1 after 2k steps. Status: **open**. Exp: `tok_byte_bytehead_001` vs `tok_byte_tokenhead_001`.

- **T2** — *token head enables faster generation (fewer steps per byte).* Measure generation throughput: bytes/sec when decoding via byte head (1 step per byte) vs token head (1 step per ~4 bytes). Even if loss slightly worse, token may win on speed/quality Pareto. Status: **open**, depends on T1 setup.

- **T3** — *loop closure (detokenize→encode) stable with token head.* Train full byte→encode→core→token_decode→detokenize→again for 2 cycles, measure drift (does loss explode after 1 loop?). Proven stable if 2nd loop loss < 1st loop +0.2. Status: **open**.

## Follow-ups

1. BPE vs purely entropy-derived patches for token boundaries
2. Scale token head to 8k vocab at 1M params
3. Hybrid head: byte head for training, token head for inference via distillation
