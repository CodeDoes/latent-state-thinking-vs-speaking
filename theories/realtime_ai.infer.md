# Realtime AI — Inferred Interpretation

> **Source**: `theories/realtime_ai.md` (verbatim)  
> **Date**: 2025-07-17

---

## What This Theory Is About

Two related but distinct ideas came up together:

### 1. **Constantly-Learning AI**
*"i want something where i can use ai in realtime and that constantly learns"*

- A model that doesn't just *infer* — it *updates itself* inference-by-inference.
- Contrasts with: "train big model, freeze, serve, hope it's good enough."
- Suggests a *continual learning / online learning / streaming learning* framing.

### 2. **PC-Device-Stream Listener AI**
*"an AI that can listen to Wifi and Bluetooth and all the different devices and streams on the pc and find information from that"*  

- Multisource sensor ingestion as input
- Wifi/Bluetooth packets as signal
- Co-occurrence of devices = "who is around"
- Output: "is there someone with a phone around or ..." etc.
- Model gets more accurate *over time* — long-term memory of "what my devices do normally."

### Common Thread
*"meta-information that is normally hidden"* — both ideas want to surface *signals you're not currently seeing*. The AI is a *sensor fusion + interpretation* engine, not a chatbot.

---

## Connections to Existing Work in This Repo

| Existing | Connection |
|---|---|
| **Dendrite Growth** | Constant learning = add a new branch each time you discover a new pattern (rather than retraining whole model) |
| **RWKV-7 (Rosa)** | Linear-time inference = suitable for realtime streams |
| **Byte-state-byte** | Multisource streams → could be byte-encodable → byte-level model with extension heads |
| **Adaptive Exit Entropy** | Realtime latency budget = exit early when confident |
| **Generator-based synthetic data** | Before any real device data, learn on synthetic patterns |

---

## Hypotheses To Test

**R1** — *A byte-level recurrent model can ingest multiple parallel device streams (concatenated or interleaved byte channels) and learn co-occurrence patterns.*

**R2** — *A model trained on synthetic device-like patterns can be **adapted** to real device data with only the dendrite-growth / branch-on-routing mechanism (no retraining).*

**R3** — *Realtime latency budget (~10–50 ms / decision) is achievable on CPU for small RWKV-nano models with adaptive exit.*

**R4** — *Constant learning via branch-addition preserves past knowledge (no catastrophic forgetting).*

---

## What's NOT Addressed Yet

- What does an AI "checked answer" look like for device streams? (you said *"it might need a way to check its answer"*) — needs verification / supervision mechanism
- Online label creation: how does "is the answer right?" decide *without ground truth*?
- Privacy: streaming wifi/bluetooth data implies personal/observed-info constraints

---

## Latent Assumptions (Infer Notes)

- You assume the model trains fast enough to keep up with realtime signal volume. RWKV-nano on CPU may be fast enough; verify.
- "Constantly learns" is ambiguous — could mean (a) every batch updates params, (b) per-day fine-tuning cycle, (c) expandable knowledge base without retraining. *Branch-addition* (G-theory) gives (c) for free.
- "Gets more accurate as it goes on" sounds like a continual-learning claim — needs measurement protocol (accumulating accuracy over time, not single-shot accuracy).
- The bifurcation between "language tasks" and "analytics tasks" suggests this is *not the same model class as your byte-state-byte*. But same RWKV substrate could be reused.

---

## Open Follow-Ups

1. `realtime_001`: Build a synthetic BinaryStream generator (multiple byte channels of noise + occasional bursts)
2. `realtime_002`: Train byte-level RWKV on synthetic streams, measure co-occurrence detection accuracy (R1)
3. `realtime_003`: Add dendrite branch per *new* pattern class, test no-forgetting (R2, R4)
4. `realtime_004`: Measure per-decision latency on CPU; verify R3 feasibility
5. `realtime_005`: Build a verification protocol for "no ground truth" predictions (self-consistency? shadow model?)

---

## What This Needs From Hardware

Real device streams need:
- A `wigle`-like wifi scanner feed
- `btmon` or similar for bluetooth
- PC telemetry (`/proc`, power, network, USB)

You haven't yet captured any of this. So step 1 is *capture* (write data to indexed log files), step 2 is *synth*, step 3 is *fit*. This is a multi-month thread.
