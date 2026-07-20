# injection-frequency

Architecture that maps bytes → byte via encoder → core → decoder has one free variable: **how often** the patch-level state `h_core` is injected into the decoder.

Current B5 (`adaptive_loop_001`) injects once:
  x = encoder_out + core_out_broadcast
  then decoder loops over x.

Alternative: inject at **every decoder layer** (per-block cross-fusion) like BLT cross-attention:
  for each decoder layer i:
    h = layer(ln(h + core_broadcast_i))

Hypothesis: per-block injection > front-only fusion at matched params.

Why: patch state is global slow dynamics; if fused once, later layers can overwrite it. Per-layer re-injection forces decoder to keep global context alive.

Minimal test: same AdaptiveLoopModel, two variants:
- `front_fusion` (existing): add core_broadcast once at decoder input
- `per_layer_fusion`: add core_broadcast before each decoder layer (learned gate: h = h + gate * core, gate = sigmoid(W * [h, core]))

Both at 228K params (per-layer adds tiny gate projections ~2*dim per layer, subtract same dim from elsewhere to match). Train 2k steps on byte_ts_001 text.txt, compare recon loss and samples. If per-layer < front by >=0.05 loss, claim proven.

Single variable: injection frequency. All else constant (dim, patches, loops, data, lr).
