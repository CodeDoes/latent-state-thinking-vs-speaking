# progressive-expansion.status

Proof chain for the progressive-expansion thread. Linked experiment
proofs in [`theories/proofs.md`](proofs.md).

## Claims

- **A1** — capacity pressure is observable per-channel/per-layer.
  Need: a metric that has dynamic range on real (not ceiling-saturating)
  easy/hard gaps. Currently `effective_dimensionality` (per-channel
  SVD loading shift) shows dynamic range; saturation and range hit
  ceiling.
  Status: **partially proven**. Need calibration to produce partial
  degradation, not total saturation. See `experiments/prog_exp_001/`.

- **A2** — detected bottleneck location *matters* (insertion there
  beats random insertion beats no insertion).
  Need: per-model ablation, three arms (insert at detected bottleneck,
  at random, no insertion). All else matched.
  Status: **unproven**. Must be tested within a single model instance
  (channel numbers are not stable across training seeds — see
  `assumptions.md`).

- **A3** — expansion achieves target performance in fewer steps than
  fixed-architecture training at the same final capacity.
  Status: **pending A2**. A2 must establish that location is load-
  bearing before "fewer steps" can be attributed to the *right*
  expansion rather than to capacity added anywhere.

- **A4** — added block does not obstruct prior capacity (loss does not
  spike after expansion).
  Status: **pending first expansion experiment**.

## Layer hierarchy (from prog_exp_001 anomaly map)
`head` > `ln_out` ≈ `embed` > `blocks.*`. Recorded as observation,
not yet confirmed against A2's three-arm test.

## Open follow-ups
- Re-run with a calibrated task where A1 metric does not saturate.
- Once A1 is solid: design A2 run. Same model, three seeds of
  expansion-point selection (detected bottleneck vs two random
  controls).
