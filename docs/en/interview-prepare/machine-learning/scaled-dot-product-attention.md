---
title: Scaled Dot-Product Attention
description: Implement a single attention head with stable normalization and an optional boolean mask.
---
# Scaled Dot-Product Attention

## Problem in plain words

Implement single-head scaled dot-product attention using only NumPy. Given query rows `Q`, key rows `K`, and value rows `V`, return the weighted values and the attention weights. Support an optional mask that excludes certain query-key pairs before normalization.

The shapes are `Q: (q, d)`, `K: (k, d)`, and `V: (k, v)`. The output has shape `(q, v)` and the weights have shape `(q, k)`. Query and key counts may differ, allowing cross-attention as well as self-attention.

For example, a zero query gives equal scores for two allowed keys. If their values are `[2]` and `[6]`, the weights are `[0.5, 0.5]` and the output is `[4]`. Masking the second key changes the weights to `[1, 0]` and the output to `[2]`.

This implementation requires nonempty finite real matrices. The optional mask is boolean and must broadcast to `(q, k)`; `True` means allowed. A fully masked row returns zero weights and zero output as an explicit policy. Inputs are not changed, and there is no batch dimension or dropout.

## Math in brief

```text
scores = Q @ K.T / sqrt(d)
scores[disallowed] = -infinity
weights = row_softmax(scores)
output = weights @ V
```

When independent query and key components have mean zero and variance one, their dot product has variance `d`. Dividing by `sqrt(d)` keeps this scale near one, reducing the tendency for softmax to become overly concentrated as the head dimension grows. It does not force every input distribution to have unit variance.

## Key idea

Each query chooses a probability distribution over the allowed keys, then uses those weights to mix the corresponding value vectors. Apply the mask before softmax, so forbidden keys receive exactly zero weight and never enter the normalization denominator.

## Step by step

1. Check matrix dimensions, finite values, the shared query/key feature width, and matching key/value row counts.
2. Divide queries by `sqrt(d)` and multiply by `K.T` to obtain scaled scores.
3. Broadcast the optional boolean mask and replace disallowed scores with negative infinity.
4. Subtract the maximum score from each row before exponentiating. Use a maximum of zero for fully masked rows to avoid subtracting negative infinity from itself.
5. Normalize rows with a nonzero exponential sum. Leave fully masked rows at zero.
6. Multiply the weights by `V` and return `(output, weights)`.

A causal self-attention mask is a lower-triangular boolean matrix: query position `i` may use key positions up to `i`. A one-dimensional key mask can instead hide padding positions from every query.

## Python solution

<<< ./solutions/scaled_dot_product_attention.py

## Complexity

Score computation takes **O(qkd)** time and mixing values takes **O(qkv)** time. Masking and normalization take **O(qk)**. Working/output space is **O(qk + qd + qv)**, including the score/weight matrices, scaled queries, and output. Converting input representations to float64 may also copy the input matrices.

## Common mistakes

- Dividing by the number of tokens instead of `sqrt(d)` uses the wrong scaling factor.
- Softmax must run over keys for each query, not over queries.
- Masking weights after softmax without renormalizing changes their sum. Multiplying scores by zero also fails because `exp(0)` is positive.
- Mask conventions differ across libraries. This function uses `True` for allowed positions and rejects numeric masks.
- A fully masked row has no mathematically defined probability distribution. Handle it explicitly rather than allowing `0 / 0` to produce NaNs.
- Large score offsets require a stable softmax. Extremely large matrix products can still overflow; this function rejects nonfinite scores.

## Interview follow-ups

- How does multi-head attention extend this operation? Each head uses its own learned projections `QW_Q`, `KW_K`, and `VW_V`, then runs attention independently. Concatenate the head outputs along the feature axis and apply a learned output projection. Heads can learn different relationships, though they are not guaranteed to specialize.
- How would you add a batch dimension and an explicit head dimension without Python loops over examples?
- How would causal masking and cached keys/values support autoregressive decoding?
- Why does materializing the full attention matrix become expensive for long sequences?
