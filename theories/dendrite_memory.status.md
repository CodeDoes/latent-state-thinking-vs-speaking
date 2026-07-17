# dendrite_memory.status

Proof chain for the Dendrite Memory Registry on RWKV backbone.

## Claims

- **D1** — *Frozen RWKV + independent LoRA branches + routing + lifecycle gates yields causally isolated functional memories.*
  Status: **not proven** — awaiting experiment `dendrite_rwkv_001`.

- **D1a** — *Independent LoRA adapters prevent interference between memories.*
  Status: open.

- **D1b** — *Address head + PPCA verifier reduces wrong-adapter activation vs always-first baseline.*
  Status: open.

- **D1c** — *Hash-gated install/verify catches silent corruption on reload.*
  Status: open.

- **D1d** — *Same registry logic ports to RWKV backbone with parity to Transformer PoC.*
  Status: open.

## Mechanism Gaps (What We Know We Don't Know)

- Whether RWKV hidden states at tap_index (~35% depth) provide sufficient signal for PPCA routing.
- Whether LoRA on RWKV `key/value/receptance/output` projections is as expressive as on Transformer `q/k/v/o`.
- Whether the registry gates' thresholds (from Dendritron SmolLM2) transfer without retuning.

## Follow-Ups (Cheap → Expensive)

1. `dendrite_rwkv_001`: 4 synthetic rules, 1 adapter each, prove isolation + routing + gates.
2. `dendrite_rwkv_002`: Same on RWKV-8-ROSA (if local weights available) — test portability.
3. `dendrite_rwkv_003`: Dynamic composition (2+ adapters active) with AND/OR gating.
4. `dendrite_rwkv_004`: Real rules (code lint, API schema) instead of synthetic.