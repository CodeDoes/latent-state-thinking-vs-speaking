# read-twice

## What you said

> i think RNN's can do "read twice" instead

Instead of inserting new layers when you detect a capacity bottleneck, run the
same input through additional recurrent forward passes using the existing
weights. The recurrent state accumulates across passes — no new parameters.

This is "thinking longer" instead of "thinking wider."

## Why this fits the project frame

It inherits directly from the think-once vs re-encode frame you already proved:

- **exp001** proved that a single think pass beats per-query re-encoding for
  WHERE-style long-horizon recall. The mechanism was *amortised thinking*:
  the model builds a state once and queries it cheaply.
- **read-twice** extends this: when amortised thinking is insufficient for a
  particular input, run a second think pass before querying. The cost is still
  amortised (only when needed) but the representational capacity increases
  without new weights.
- **progressive-expansion** adds new layers. Read-twice adds new passes.
  Both are surgical; one changes topology, the other changes compute depth.

## What the math already predicts

The bottleneck metrics you already have (saturation, EDD shift) are measuring
*how hard the current representation is working*. If you correlate those scores
with *how many extra passes an input needs before the downstream task succeeds*,
you get a direct read-out of "required thinking depth" from the activation
signature alone.

## The key hypothesis

> A model given 2 recurrent passes learns harder tasks faster (at matched
> compute) than one that does 1 pass + a new layer.

Or equivalently:

> The capacity-pressure signals measured by progressive-expansion metrics
> predict the number of additional passes needed to recover accuracy on
> hard inputs.

## What this changes in the proof chain

| Progressive expansion step | Read-twice equivalent |
|---------------------------|----------------------|
| Detect bottleneck via activations | Same — metrics still needed |
| Insert new layer at detected junction | Run N additional recurrent passes |
| Train new layer on hard task | No new training needed for existing weights |
| Risk: disturbs existing capabilities | No risk — identical weights, same input |

## Open questions

1. **Monotonicity**: does additional pass depth always help, or does the
   model start *unlearning* the easy task after too many passes?
2. **Per-example vs uniform**: should every input get the same number of
   passes, or should the model learn *when* to stop (early-exit)?
3. **GRU vs RWKV**: does read-twice work equally well for both recurrence
   types, or does one architecture benefit more?

## Minimal test skeleton

1. Train a small GRU on task A (single pass) until converged.
2. Run it on task B (harder, same domain) with 1, 2, 3, 4 recurrent passes.
3. Record accuracy vs pass depth.
4. Compare to: same model with an additional Linear layer inserted at the
   detected bottleneck position (from progressive expansion metrics).
5. Single outcome: pass-depth-to-accuracy curve vs layer-insertion accuracy.
   Winner = whichever learns the hard task faster at matched compute.
