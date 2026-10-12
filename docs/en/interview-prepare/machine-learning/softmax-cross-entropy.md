---
title: Softmax and Cross-Entropy
description: Compute stable class probabilities, mean cross-entropy, and the gradient with respect to logits.
---
# Softmax and Cross-Entropy

## Problem in plain words

Implement row-wise softmax and cross-entropy using only NumPy. Given logits of shape `(n, c)` and integer class labels of shape `(n,)`, return class probabilities, the mean cross-entropy loss, and the loss gradient with respect to each logit. Large positive or negative logits should not overflow the exponential.

For example, logits `log([1, 2, 1])` produce probabilities `[0.25, 0.5, 0.25]`. If the correct class is index `1`, the loss is `log(2)` and, for this one-example batch, the gradient is `[0.25, -0.5, 0.25]`.

This implementation accepts nonempty finite real logits and integer labels in `[0, c)`. `softmax` returns probabilities; `softmax_cross_entropy` returns `(mean_loss, gradient)`. Inputs are not modified. Labels must have an integer dtype, so `[1.0]` is rejected rather than silently converted.

## Math in brief

For each row, subtract its maximum before exponentiating:

```text
shifted[j] = logits[j] - max(logits)
log_p[j] = shifted[j] - log(sum(exp(shifted)))
p[j] = exp(log_p[j])
L = -mean(log_p[i, labels[i]])
dL/dlogits[i, j] = (p[i, j] - indicator(j == labels[i])) / n
```

Subtracting a constant from every logit in a row leaves its probabilities unchanged. The largest shifted logit is zero, so the denominator contains at least one `exp(0) = 1`.

## Key idea

Work in log-probability space for the loss. Taking `log(softmax(logits))` can lose information when a very small probability underflows to zero, even if the true log probability is representable. The gradient simplifies to probabilities minus the target indicator, divided by the batch size.

## Step by step

1. Validate a nonempty two-dimensional logits matrix and finite values.
2. Subtract each row's maximum, retaining the singleton axis for broadcasting.
3. Compute the log of the sum of shifted exponentials and subtract it from the shifted logits.
4. Exponentiate the log probabilities when probabilities are requested.
5. Validate integer target indices and average the negative target log probabilities.
6. Copy probabilities into the gradient, subtract one at each target position, and divide by `n`.

For logits `[1000, -1000]` and target class `1`, the shifted logits are `[0, -2000]`. The second probability may round to zero, but its log probability remains `-2000`, giving finite loss `2000` and gradient `[1, -1]`.

## Python solution

<<< ./solutions/softmax_cross_entropy.py

## Complexity

For `n` examples and `c` classes, time is **O(nc)** and working/output space is **O(nc)**. Indexing the target classes takes **O(n)** time and space. No `(n, c, c)` softmax Jacobian is constructed.

## Common mistakes

- Subtracting one global maximum can still leave a different row with all exponentials underflowing. Stabilize each row separately.
- Normalizing over the batch axis mixes independent examples instead of producing a distribution over classes.
- Clipping probabilities before taking a logarithm changes the objective and can disagree with the gradient being returned.
- Forgetting `/ n` gives a summed-loss gradient while reporting a mean loss.
- Building the full softmax Jacobian wastes memory. Combining softmax with cross-entropy gives the direct expression above.
- NaN and infinite logits are rejected. Even finite values near opposite float64 extremes can overflow their difference; shift invariance does not provide arbitrary-precision arithmetic.

## Interview follow-ups

- How would the gradient change for soft targets or label smoothing?
- How would you support ignored labels or per-example weights while defining the reduction precisely?
- What does temperature scaling do to probabilities and gradients?
- How are two-class softmax and binary logistic regression related?
