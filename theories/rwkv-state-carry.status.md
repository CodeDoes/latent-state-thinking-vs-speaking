# rwkv-state-carry.status

Proof chain for RWKV state carry.

## Claims

- **S1** — *stateful carry across segments improves long-horizon accuracy over zero init at matched params.* RWKVNano 128d/3L, max_len 256, noise_max 10, 2k steps. Proven if carry accuracy > zero +0.1 on WHERE/hard queries. Status: **open**. Exp: `rwkv_carry_zero_001` vs `rwkv_carry_stateful_001`.

- **S2** — *learned initial state helps but less than full carry.* Arm C learned init parameter. Proven if learned > zero +0.05 but < carry. Status: **open**. Exp: `rwkv_carry_learnedinit_001`.

- **S3** — *carry induces longer decay channels.* Measure time_decay distribution: mean w_ closer to 1 for carry vs zero. Proven if mean w_ carry > zero +0.05. Status: **open**, metric from checkpoint.

## Follow-ups

1. Full BPTT vs truncated (detach) – does gradient through carry help?
2. Carry across byte→state→byte loops (adaptive_loop with stateful core)
3. Scale carry benefit with context length 512, 1024
