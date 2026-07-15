# progressive-expansion — automated detection metrics

This document lists **mathematical quantities** you can compute fully
automatically from activations/grads. No manual threshold tuning. Each metric
produces a single number per channel, per layer, per token position — then
you rank-order by that number and pick the top-k.

No hypothesis testing, no p-values, no confidence intervals. Just: *compute,
sort, pick the worst*. That is deterministic and reproducible.

## M1. Saturation ratio (forward-pass only)

$$S_{c,l} = \frac{1}{N}\sum_i \mathbb{I}(|a_{i,c,l}| > \tau_c)$$

Count fraction of stimuli where channel $c$ in layer $l$ exceeds magnitude
threshold $\tau_c$. The threshold is **not tuned per stimulus** — it comes
from the model's happy-domain calibration:

$$\tau_c = \mu_c^{\text{easy}} + 4\sigma_c^{\text{easy}}$$

where $\mu_c^{\text{easy}}$ and $\sigma_c^{\text{easy}}$ are the mean and
standard deviation of channel $c$ computed over a held-out set of easy
(training-distribution) examples. This makes the threshold data-driven,
not hand-picked.

**Output**: scalar $S_{c,l} \in [0,1]$ for every channel-layer pair.
Higher = more saturated under stress. Top-k channels = detected bottleneck.

**Cost**: one forward pass over hard samples + one pre-computed from easy
samples. Negligible.

---

## M2. Activation-range expansion ratio (forward-pass only)

$$R_{c,l} = \frac{\text{range}_c^{\text{hard}}(l)}{\text{range}_c^{\text{easy}}(l)}$$

Where range is $\max - \min$ across the sample batch for each channel.
If the model uses rectified activations, take $\max$ instead of range.

A channel that expands dramatically on hard inputs while staying tight on
easy inputs is being pushed into unexplored territory — either because it's
trying to encode new structure or because it's saturating. Either way, it's
atypical behavior.

**Output**: scalar $R_{c,l}$. Values near 1.0 = same behavior.
Values >> 1.0 = unusual activity under stress.

**Cost**: one forward pass each over easy and hard batches. Negligible.

---

## M3. Gradient-norm concentration (backward pass needed)

$$G_{c,l} = \frac{\|\nabla_{w_{c,l}} L\|^2}{\sum_{c',l'} \|\nabla_{w_{c',l'}} L\|^2}$$

Fraction of total squared gradient norm flowing through channel $c$ in
layer $l$, computed on the hard-sample mini-batch. Then compare to the
same quantity on an easy mini-batch:

$$\Delta G_{c,l} = G_{c,l}^{\text{hard}} - G_{c,l}^{\text{easy}}$$

Channels with large $\Delta G$ are those that suddenly become important
(or irrelevant) when the problem gets harder. This captures gradient-level
pressure, not just activation-level pressure.

**Output**: scalar $\Delta G_{c,l}$ for every channel-layer. Top positive
values = channels whose importance shifted under load.

**Cost**: standard backward pass. Cheap relative to training.

---

## M4. Singular-value spectral decay (forward-pass only, layer-level)

Compute the covariance matrix of layer-$l$ activations over the hard batch:

$$\Sigma_l = \frac{1}{N}\sum_i a_i a_i^\top$$

Take its eigenvalues (or SVD singular values): $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d$.

Define **effective dimensionality**:

$$D_{\text{eff},l} = \exp\left(-\sum_j p_j \log p_j\right), \quad p_j = \frac{\lambda_j}{\sum_k \lambda_k}$$

Entropy-based effective dimensionality: if activations occupy a full
$d$-dimensional subspace evenly, $D_{\text{eff}} \approx d$. If they collapse
to a line, $D_{\text{eff}} \approx 1$.

Compare $D_{\text{eff}}^{\text{hard}}$ vs $D_{\text{eff}}^{\text{easy}}$:

$$\Delta D_l = \frac{D_{\text{eff},l}^{\text{hard}}}{D_{\text{eff},l}^{\text{easy}}}$$

Small $\Delta D$ (< 1) = representation is degenerating, channels are
correlating, information is being lost to fewer dimensions. That's a
bottleneck in a different form — not saturation, but **collapsing
expressivity**.

**Output**: scalar $\Delta D_l$ per layer. Values < 1 = collapsing.
More collapsed = worse.

**Cost**: $O(N d^2)$ per layer for covariance, $O(d^3)$ for eigendecomp.
$d$ is small (~64-256 in your models), so this is fast. Batch size $N$
can be moderate (100-500).

---

## M5. Channel-output mutual-information proxy (forward-only, approximate)

True mutual information estimation requires density estimation — too expensive
and noisy for small-scale experiments. Instead, use a **linear predictability**
proxy:

For each channel $c$ in layer $l$, fit a linear map from the channel's
activations (over easy examples) to the final output:

$$\hat{y}_{c,l} = W_{out} a_{c,l} + b_{out}$$

Then measure $R^2$ of this map on the hard examples. If a channel that
was predictive of output during easy examples suddenly loses predictive
power on hard examples, that channel has entered a regime where the
output head can no longer read it reliably — another form of bottleneck.

$$U_{c,l} = R^2_{c,l}(\text{easy}) - R^2_{c,l}(\text{hard})$$

Positive $U_{c,l}$ = channel broke in a way the output head can't recover.

**Output**: scalar $U_{c,l}$ per channel-layer. Positive = useful info
now unreadable.

**Cost**: one linear regression per channel per layer. Analytic closed-form
solution ($W = (X^\top X)^{-1} X^\top y$). Very cheap.

---

## How to use these (the algorithm)

```
Input: trained model M, easy_batch, hard_batch (both labeled)
Output: ranked list of (layer, channel) tuples

Step 1: Run M on easy_batch, collect activations {A_easy}
Step 2: For each channel c in each layer l:
        Compute μ_c, σ_c over easy_batch         ← calibration
        Store A_easy[c] for later regression       ← for M5

Step 3: Run M on hard_batch, collect activations {A_hard}

Step 4: Choose one metric M_i. Compute for all (c,l):
        score[c,l] = M_i(A_easy, A_hard)[c,l]

Step 5: Rank all (c,l) by score[c,l], descending.

Step 6: Return top-K channels as the bottleneck set.
        Default K = 5% of total channels (or specify absolute count).
```

No threshold tuning. No manual inspection. Fully deterministic.

Pick **one** metric for the first experiment. M1 (saturation ratio) is
cheapest and most interpretable. M4 (spectral decay) catches a different
failure mode (collapsing expressivity vs channel-level saturation) and is
a good second choice if M1 finds nothing interesting.
