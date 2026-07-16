# byte-state-byte.infer

Interpretation of [`byte-state-byte.md`](byte-state-byte.md).

## What the .md leaves out

### Why two parallel attempts at all
The repo has two shapes — step-function (explicit patch_lengths
from a surrogate entropy), and RWKV-based — that are doing the same
thing. The .md treats them as siblings. They were not started in
parallel by plan; they exist because the step-function attempt
diverged in phase 2 (B2), and we *swapped in* the RWKV-based
version before confiming the step-function result. Concretely:
`rnn_patch_002` was the last step-function run on disk; everything
after that commit (`276b72c`, `6395f33`, `4abe185`) is
encoder-patcher-decoder variants that overlap with the rwnk
unrolled attempts in motivation but use the learned-gate encoder
spec. We never reconciled which one is canonical.

### What "encoder_state_ablation" actually swept
The 7-arms are state *topology*, not state *content*. The arms
mix and match four boolean state components (byte-level, patch,
encoder-side, mutable-vs-static) — a hand-built 2^4 minus the
trivial combos. It is not a parameter ablation; it is constant
parameter count, varying state subgraph. Read the summary.json
as a graph-search result, not a capacity-vs-quality curve.

### The decoder stall isn't proven
B3 says decoder stalls at loss 0.156. But the *encoder* loss in
that same run is 0.035 — meaning the encoder is essentially
solved. The decoder is doing a different task (it generates
bytes), sees a different input distribution (its own previous
output plus the patched state), and the training schedule is
unmodified across the two heads. None of those differences have
been controlled for. It's plausible (not proven) that the
decoder stall is just "different task, same lr/cosine schedule,
less steps apparently needed for encoder" — i.e., the right fix
is two optimisers or asymmetric step counts, not a model change.

### What's missing from the open follow-ups
The four open follow-ups in the .md are hypothesis-generating,
not benchmarked. They are organised cheap-to-expensive, not by
likelihood of distinguishing claims. On reflection, follow-up 1
(diagnose decoder stall) is the most likely to be informative
for the lowest cost — but it tests an *implementation detail*
of the existing impl, not the architecture family. Follow-up 4
(unrolled vs recurrent) is the one that would actually test a
mechanism from the .md narrative.

## Risks the .md doesn't flag
- Both attempts result in samples that are blanks / repeated chars.
  When measuring *training loss* goes down, *sample quality* does
  not move. We don't have any penalty metric that catches this.
- We have not run any controlled compare between the two stacks
  on identical scripts + identical data. They live on different
  training files (`train_rnn_patch.py` vs `train_shared_state_*`).
  A claim like "RWKV-based beats step-function" is not on disk.
