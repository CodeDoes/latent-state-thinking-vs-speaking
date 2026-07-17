# Dendritron: An RWKV That Grows — Architectural Extensions to a Frozen Core

> **Report**: 2025-07-17  
> **Core idea**: RWKV as a **frozen trunk**. Functional capabilities added as **physical architectural extensions** (branches) that attach to the core without modifying it. Each branch is a self-contained module: new layers, cross-attention, memory, output heads. The core never changes; the network grows by accretion.

---

## 1. The Metaphor: Tree Growth, Not Weight Patching

```
                    ┌─ Branch: Code Generation ──► [Cross-Attn + Decoder Head]
                    │
   ┌────────────────┼──────────────────────────────► Frozen RWKV Core (Trunk)
   │  (frozen,      │
   │   never        ├─ Branch: Math Reasoning ───► [Extra Layers + Scratchpad Head]
   │   changes)     │
   │                ├─ Branch: API Schema ───────► [Memory Module + Validator Head]
   │  Trunk =       │
   │  RWKV blocks   └─ Branch: Tool Use ────────► [Tool Router + Action Head]
   │  0..L-1                        (each branch trained independently)
   └──────────────────────────────────────────────────────► Token Stream
```

**Key principle**: The trunk (RWKV layers 0..L-1) is **cast in stone**. No LoRA, no fine-tuning, no state injection. Growth happens by **attaching new modules** to well-defined interfaces on the trunk.

---

## 2. Extension Interfaces (Where Branches Attach)

| Interface | What it exposes | Branch types that attach here |
|---|---|---|
| **Hidden states at layer k** | `h_k ∈ ℝ^{B×T×C}` | Cross-attention branches, extra FFN layers, probing heads |
| **Recurrent state (WKV)** | `(num, den, xx) ∈ ℝ^{B×C}` | State-memory branches, long-range retrieval, state editors |
| **Output logits** | `logits ∈ ℝ^{B×T×V}` | Output heads (specialized vocab), rerankers, verifiers |
| **Token embeddings** | `embed ∈ ℝ^{V×C}` | Vocabulary expansion branches, multimodal projectors |
| **Attention scores (ROSA)** | Suffix automaton state | Symbolic memory branches, exact retrieval modules |

Each interface is a **stable contract** — the trunk guarantees the tensor shape and semantics. Branches are compiled against this contract.

---

## 3. Branch Types (Growth Primitives)

### 3.1 Cross-Attention Branch (Read from Trunk)
```
Trunk h_k ──────► CrossAttn(Q=branch, K=trunk, V=trunk) ──────► Branch Layers ──────► Head
```
- Reads trunk representations without modifying them
- Adds `k` new layers with their own params
- Example: Code generation branch attends to trunk's semantic understanding

### 3.2 Residual Expansion Branch (Deepen the Trunk)
```
Trunk h_k ──────► [New Block k+1] ──────► [New Block k+2] ──────► ... ──────► Head
```
- Adds new RWKV blocks *after* the frozen trunk
- Branch has its own recurrent state
- Example: Math reasoning adds 4 deep reasoning layers

### 3.3 State-Memory Branch (Attach to Recurrence)
```
Trunk WKV state ──────► Memory Module (key-value store, episodic) ──────► Modified state ──────► Trunk continues
```
- Intercepts recurrent state, augments it, passes back
- Does not modify trunk weights — only reads/writes state tensors
- Example: Long-context memory, fact retrieval, user preference store

### 3.4 Output Head Branch (Specialize the Vocabulary)
```
Trunk logits ──────► Head: Linear(C → V_specialized) ──────► Specialized logits
```
- New output projection for domain-specific tokens
- Can be routed: main vocab vs. branch vocab
- Example: SQL generator head, JSON schema validator head

### 3.5 Vocabulary Expansion Branch (Grow the Embedding)
```
New tokens ──────► Embedding Expansion (C) ──────► Concatenated to trunk embed ──────► Trunk
```
- Adds new token embeddings without retraining old ones
- Trunk sees extended vocab; new tokens initialized from similar old tokens
- Example: Domain terminology, user-defined symbols, multimodal tokens

---

## 4. Growth Protocol (How to Add a Branch)

```
1. DEFINE INTERFACE
   - Which trunk layer(s) to attach to
   - What tensors to read/write
   - Input/output shapes

2. INITIALIZE BRANCH
   - Random init or distill from trunk (probe trunk, fit branch to mimic)
   - Branch params = 0 initially (identity) or small

3. TRAIN BRANCH (trunk frozen)
   - Freeze all trunk params
   - Train only branch params on branch-specific data
   - Loss: branch task + consistency regularization (don't drift trunk outputs)

4. VERIFY & INSTALL
   - Hash-gated integrity check (sha256 of branch weights)
   - Functional equivalence on probe set
   - Regression test: trunk accuracy on *other* branches unchanged
   - Register in branch registry

5. COMPOSE (optional)
   - Multiple branches active: route per input, or merge outputs
   - AND/OR gating: branch A AND branch B → both process, merge logits
```

---

## 5. Why This Is Not LoRA / State Injection

| Approach | Modifies | Growth mechanism | Reversibility |
|---|---|---|---|
| **LoRA** | Weight deltas (same layers) | Add rank-r matrices to existing projections | Delete LoRA weights |
| **State injection** | Recurrent state tensors | Swap WKV state at runtime | Restore previous state |
| **Architectural extensions (this)** | **Network topology** | **Add new layers/modules with new params** | **Delete branch module** |

**Key difference**: LoRA and state injection work *within* the fixed architecture. Extensions **grow the architecture itself** — new depth, new width, new pathways, new outputs. The trunk is truly immutable; growth is accretive.

---

## 6. Connection to Prior Work

| Work | Similarity | Difference |
|---|---|---|
| **Progressive Neural Networks** (Rusu et al., 2016) | Frozen columns, new columns for new tasks | Columns are full networks; here trunk is shared RWKV, branches are lightweight |
| **Mixture of Experts** | Modular experts added over time | Experts replace FFN; branches attach at *any* interface (state, hidden, output, embed) |
| **DMoE** (arXiv:2606.14243) | Decoupled experts + router | DMoE experts on Transformer FFN; branches on RWKV *any interface* |
| **AdapterSoup / Merger** | Compose adapters | Adapters = weight deltas; branches = new architecture |
| **RWKV-ROSA** | Built-in symbolic memory | ROSA is *inside* trunk; branches are *outside*, attachable |

---

## 7. Minimal Proof (Single-Variable Ablation)

**Claim G1**: A frozen RWKV trunk + one architectural branch (cross-attention + 2 layers + head) trained on Task A achieves Task A accuracy without degrading trunk's Task B accuracy.

| Exp | Variable | Baseline | Test | Measure |
|---|---|---|---|---|
| **G1a** | Branch necessity | Trunk only (multi-task) | Trunk + branch for Task A | Task A acc, Task B acc (interference) |
| **G1b** | Interface depth | Attach at layer L-1 (top) | Attach at layer L/2 (middle) | Branch acc, trunk preservation |
| **G1c** | Branch depth | 1 extra layer | 4 extra layers | Scaling of branch capability |
| **G1d** | Composition | Branch A only | Branch A + Branch B (both active) | Multi-task acc, routing accuracy |

**Minimal scale:**
- Trunk: RWKV-nano, dim=128, 4 layers (frozen, ~400K params)
- Branch: 2 RWKV blocks + cross-attn + head (~200K params)
- Data: 2 synthetic rules × 2000 train
- Compute: CPU, <5 min per branch

---

## 8. Implementation Sketch

```python
class RWKVTrunk(nn.Module):
    """Frozen RWKV core. No gradients ever."""
    def __init__(self, vocab_size, dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RWKVBlock(dim) for _ in range(num_layers)])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.freeze()
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, input_ids, return_hiddens=False):
        x = self.embed(input_ids)
        hiddens = []
        for block in self.blocks:
            x, state = block(x)
            hiddens.append(x)
        logits = self.head(self.ln_out(x))
        return (logits, hiddens) if return_hiddens else logits


class Branch(nn.Module):
    """Base class for architectural extensions."""
    def __init__(self, trunk_dim, attach_layer):
        super().__init__()
        self.trunk_dim = trunk_dim
        self.attach_layer = attach_layer  # which trunk hidden to read
    
    def forward(self, trunk_hidden, trunk_state=None):
        # trunk_hidden: [B, T, C] from trunk.blocks[self.attach_layer]
        # trunk_state: optional WKV state dict
        raise NotImplementedError


class CrossAttentionBranch(Branch):
    """Reads trunk hidden, processes through own layers, outputs logits."""
    def __init__(self, trunk_dim, attach_layer, num_layers=2, head_vocab=None):
        super().__init__(trunk_dim, attach_layer)
        self.cross_attn = nn.MultiheadAttention(trunk_dim, num_heads=4, batch_first=True)
        self.blocks = nn.ModuleList([RWKVBlock(trunk_dim) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(trunk_dim)
        self.head = nn.Linear(trunk_dim, head_vocab or trunk_dim)
    
    def forward(self, trunk_hidden, trunk_state=None):
        # Cross-attend: branch queries trunk
        x, _ = self.cross_attn(trunk_hidden, trunk_hidden, trunk_hidden)
        for block in self.blocks:
            x, _ = block(x)
        return self.head(self.ln(x))  # [B, T, vocab_branch]


class GrowingRWKV(nn.Module):
    """Trunk + dynamic branches."""
    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk
        self.branches = nn.ModuleDict()  # name -> Branch
    
    def add_branch(self, name, branch):
        self.branches[name] = branch
    
    def forward(self, input_ids, active_branches=None):
        logits, hiddens = self.trunk(input_ids, return_hiddens=True)
        outputs = {'trunk': logits}
        
        if active_branches:
            for name in active_branches:
                branch = self.branches[name]
                h = hiddens[branch.attach_layer]
                outputs[name] = branch(h)
        
        return outputs
```

---

## 9. The Ideal: Continuous Growth

```
Day 1:  Trunk only (general language)
Day 7:  + Code branch (attaches at layer 2, 3 layers, code vocab head)
Day 14: + Math branch (attaches at layer 3, 4 layers, scratchpad head)  
Day 30: + Memory branch (intercepts WKV state, key-value store)
Day 60: + Tool branch (cross-attn + router + action head)
Day 90: + Vocab expansion (new tokens for user's domain)
...
Year 5: Trunk unchanged. 50 branches. Each added, verified, never broke others.
```

**This is the vision**: An RWKV that *grows* like a tree — trunk fixed, branches added as capabilities are needed. No catastrophic forgetting (trunk frozen, branches isolated). No retraining (only new branch trains). No architecture search (interface contract fixed).

---

## 10. Next Steps

1. **Implement `src/rwkv_growing.py`** — Trunk + Branch base classes + 3 branch types
2. **Experiment `grow_001`** — Prove G1a: one branch, no interference
3. **Experiment `grow_002`** — Test attachment depth (G1b)
4. **Experiment `grow_003`** — Two branches, routing (G1d)
5. **Branch registry** — Hash-gated install/verify/delete (like Dendritron v0 but for modules)

---

*Per AGENTS.md: "Prove one thing at a time." First proof: **G1a** — a single cross-attention branch on a frozen trunk learns its task without touching the trunk.*