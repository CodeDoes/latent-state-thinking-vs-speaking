# realtime_ai.status

Proof chain for "Realtime constantly-learning device-stream AI."

## Claims

- **R1** — *Byte-level recurrent model can ingest multi-channel device streams and learn co-occurrence.*
  Status: **not proven** — no synthetic stream generator yet.

- **R2** — *Trained synthetic-pattern model adapts to real device data via branch-addition (no retraining).*
  Status: blocked on R1 + dendrite-growth proof.

- **R3** — *Realtime latency budget (~10–50ms / decision) achieved on CPU for small RWKV-nano.*
  Status: open — not measured.

- **R4** — *Branch-addition preserves past knowledge (no catastrophic forgetting).*
  Status: blocked on dendrite-growth proof.

## Mechanism Gaps

- What does a "device stream" look like as a byte sequence? (no spec yet)
- How does a model *check its own answer* without ground truth? (you said it needs a verification mechanism)
- What's the latency budget on the *real* hardware vs CPU?

## Hardware / Data

| Need | Status |
|---|---|
| Wifi scanner logger | not started |
| Bluetooth monitor | not started |
| PC telemetry capture | not started |
| Synthetic stream generator | not started |

## Follow-Ups (Cheap → Expensive)

1. `realtime_001`: Synthetic multichannel byte-stream generator
2. `realtime_002`: Train RWKV-nano on streams, measure co-occurrence learning (R1)
3. `realtime_003`: Latency measurement on CPU (R3)
4. `realtime_004`: Branch-addition continual test (R2, R4, depends on dendrite-growth)
5. `realtime_005`: Real-device capture pipeline

## Implementation Status

| Component | Status |
|---|---|
| Synthetic stream generator | not started |
| Co-occurrence metric | not started |
| Realtime eval harness | not started |
| Wifi capture | not started |
| Bluetooth capture | not started |
