# Theory 07 — Capacity is NOT the bottleneck (refuted)

## Claim (REFUTED)
The hypothesis that the latent state "isn't accurate enough" due to
insufficient dimension/capacity, and that *increasing `d_state` would fix
accuracy*.

## Why it was worth testing
- Intuition: a bigger state vector should hold more world information.
- `L* = 77.9 bits → 5 floats` is the info-theoretic floor to store a whole
  world (6 items × (log2 80 objs + log2 32 locs) + rel + 8 bits).

## Refutation evidence
1. `d_state = 48` is **~9.6× the 5-float floor** → capacity is amply sufficient.
2. Latent has **579K params vs baseline 399K** (40% more) yet still loses →
   not a capacity fight.
3. Increasing `d_state` 32 → 48 did **not** improve WHERE (0.018 vs 0.06) →
   the deficit is architectural, not dimensional.
4. The latent's failure mode is **collapse to majority-class** (frozen AT/SAME
   at the always-NONE value, random WHERE), i.e. *starvation of gradient
   signal*, not *starvation of capacity*.

## Conclusion
The problem is **collapse / objective design** (T03 label non-uniqueness,
T04 Normally-Empty slots, T06 missing reconstruction signal) — **not** raw
capacity. Increasing `d_state` alone will not un-collapse the model; it can
even make collapse easier (more dead space to ignore).

## Status
❌ **Refuted.**

## Related
T01 (core hypothesis), T03 (label non-uniqueness), T06 (auxiliary loss).
