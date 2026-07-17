# Theory: Correcting RWKV State Passing for Long-Term Memory (T=1 and Sequence Boundary Continuity)

## 1. Executive Summary
This theory identifies and resolves a critical, silent bug in the `RWKVBlock` recurrence logic. While parallel sequence-level training functions correctly, sequential evaluation/generation (where token-by-token processing is done with sequence length $T = 1$) completely discards the `num` and `den` recurrent channel states. This turns the WKV recurrent channel-mixing into a memory-less layer during autoregressive decoding.

We propose and derive a mathematically exact, fully vectorized, autograd-friendly correction that seamlessly integrates historical `num` and `den` states for any sequence length $T \ge 1$.

---

## 2. Mathematical Foundation & Recurrence

In standard RWKV-4, the WKV (weighted key-value) recurrence at time step $t \ge 0$ maintains two running sums (numerator $num_t$ and denominator $den_t$) with decay $w\_ = \exp(-\exp(w)) \in (0, 1)$ and bias $u$:

$$num_t = w\_ \cdot num_{t-1} + e^{k_t} v_t$$
$$den_t = w\_ \cdot den_{t-1} + e^{k_t}$$

The WKV output $wkv_t$ is defined as:

$$wkv_t = \frac{w\_ \cdot num_{t-1} + e^{u + k_t} v_t}{w\_ \cdot den_{t-1} + e^{u + k_t}}$$

---

## 3. The Silent Memory-Loss Bug

In the current codebase, `RWKVBlock.forward` computes WKV output in parallel over a sequence of length $T$:

```python
log_term_num = -t_idx * log_w + k + torch.log(torch.abs(v) + 1e-8)
log_term_den = -t_idx * log_w + k
cum_num = _cumlogsumexp(log_term_num)
cum_den = _cumlogsumexp(log_term_den)
num = torch.exp(cum_num + t_idx * log_w) * sign_num
den = torch.exp(cum_den + t_idx * log_w)
```

During generation or step-by-step unrolled execution, the model is called sequentially with a single token at a time ($T = 1$). Under this condition:
- `t_idx = 0`
- `cum_num = log_term_num = k + log(|v|)`
- `num = exp(k + log(|v|)) * sign_num = e^k * v`
- `den = e^k`

Although the previous `num` and `den` states from earlier tokens are passed inside the `state` dict and saved inside `new_state`, they are **never read** at the beginning of `forward`!
Consequently, **WKV recurrence has zero memory across step-by-step transitions**, completely undermining the temporal capacity of the model during sequential generation/evaluation.

---

## 4. The Unified Vectorized State-Passing Solution

To preserve mathematical equivalence between parallel training and sequential generation, we must incorporate `num_prev` and `den_prev` into the running sums across a sequence of length $T \ge 1$.

For any step $t \in [0, T-1]$, the exact contribution of the initial states is decayed by $w\_^{t+1}$:

$$num_t = w\_^{t+1} \cdot num_{prev} + S^{num}_t$$
$$den_t = w\_^{t+1} \cdot den_{prev} + S^{den}_t$$

where $S^{num}_t$ and $S^{den}_t$ are the cumulative sums computed solely from the current sequence of tokens (which the existing vectorized code already computes correctly).

### Derivation for $T = 1$:
At $t = 0$:
$$num_0 = w\_ \cdot num_{prev} + e^{k_0} v_0$$
$$den_0 = w\_ \cdot den_{prev} + e^{k_0}$$

This perfectly matches the step-by-step RWKV definition.

### Vectorized Implementation:
1. Load `num_prev` and `den_prev` from `state` (defaulting to zero).
2. Compute the element-wise exponential decay factor for each position $t$:
   $$\text{decay\_factor}_t = \exp((t + 1) \cdot \log w\_)$$
3. Blend the initial states with the current sequence sums:
   `num = decay_factor * num_prev.unsqueeze(1) + S_num`
   `den = decay_factor * den_prev.unsqueeze(1) + S_den`

This correction is fully vectorized, adds no extra parameters, preserves parallel training efficiency, and restores perfect WKV memory during sequential decoding.
