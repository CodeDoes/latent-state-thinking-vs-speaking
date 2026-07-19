# Where Token Encode/Decode Space Lives in RWKV-7

Analysis based on the actual RWKV-7 model code (`rwkv/model.py`) and
the original RWKV paper (2305.13048).

---

## Data Flow in a Real RWKV-7

Every token goes through the same path:

```
token_id → embed.weight → [layers 0..L-1] → ln_out → head.weight → logits
```

Each layer does:

```
x → ln1 → TMix (WKV recurrence) → +x → ln2 → CMix (FFN) → +x
```

There is **no structural distinction** between "encoder layers" and
"decoder layers" in the code. Every layer applies the identical TMix+CMix
operation. The model is a uniform stack.

## So Where Is the Tokenizer?

The tokenizer's two functions — **encode bytes→token IDs** and
**decode token IDs→bytes** — are handled by components *outside* the
model:

| Tokenizer function | Model component |
|---|---|
| Encode: bytes → token IDs | **The TRIE** (not in the model) |
| Decode: token IDs → bytes | **TRIE's idx2token lookup** (not in the model) |
| Encode: token IDs → state vectors | **embed.weight** (layer -1) |
| Decode: state vectors → token IDs | **ln_out + head.weight** (layer L) |

So the model itself only does token→state and state→token. The
byte→token and token→byte boundaries live in the TRIE, which is
external to the neural network.

## The Value Residual: Layer 0's Special Role

The **only** operation that distinguishes layer 0 from other layers is
the value residual (`v_first`):

```python
if layer_id == 0:
    v_first = v  # store layer 0's value vectors
else:
    v = v + (v_first - v) * sigmoid(v0 + (xv @ v1) @ v2)  # blend with stored
```

Layer 0's value vectors are preserved and blended into all subsequent
layers via a learned gate. This means:

- **Layer 0 establishes the initial "reading" of each token.**
  Its value vectors are a first-pass encoding that later layers can
  reference through the residual.
- **The v_first blend is trainable per-layer.** Each later layer learns
  how much of the original token encoding to mix into its own values.

This is the closest thing to a dedicated "encoder layer" in RWKV-7.
But it's a residual, not a replacement — layer 0's output is mixed in,
not overwritten.

## What Probing Literature Says

For transformer LMs, probing studies consistently find:

| Layer depth | Represented information |
|---|---|
| Early (0-2) | Surface form, token identity, POS tags |
| Middle (3- N) | Semantics, syntax, relations |
| Late (N-2..N) | Next-token prediction features, output-oriented |

For RWKV specifically, there is minimal probing literature. The
DREAMSTATE paper (2601.19221) confirms RWKV state is an editable
knowledge representation but doesn't probe individual layer roles.

## Implications for Tokenizer Surgery

The uniform stack structure means **there is no natural split point**
for "encoder" vs "core" vs "decoder" layers. The shunt approach
(bypassing layer 0 and feeding into layer 1) is arbitrary — every
layer does the same computation. The choice is pragmatic, not
architecturally justified.

The cleanest interpretation: the tokenizer is **entirely outside** the
model. Its encoder (TRIE) produces token IDs, its decoder (idx2token)
consumes token IDs. The model only operates on vectors. A learned
front-end (loopy RNN) that replaces the TRIE is conceptually the
simplest replacement — it occupies the same position in the data flow
as the original tokenizer, without modifying the model at all.

## References

- RWKV paper: [2305.13048](https://arxiv.org/abs/2305.13048)
- DREAMSTATE: [2601.19221](https://arxiv.org/abs/2601.19221)
- RWKV-7 source: `rwkv/model.py` (pip package)
- Probing BERT: [1906.00906](https://arxiv.org/abs/1906.00906) — findings
  on layer roles in transformers, hypothesized to generalize to linear
  RNNs but not yet verified
