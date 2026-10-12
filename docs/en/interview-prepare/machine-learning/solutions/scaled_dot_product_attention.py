"""Single-head scaled dot-product attention with a boolean keep mask."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def scaled_dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    mask: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (output, weights) for Q:(q,d), K:(k,d), V:(k,v).

    A boolean mask must broadcast to (q,k); True means a key is allowed.
    Fully masked rows have zero weights and zero output. There is no batch
    dimension, projection, dropout, or mutation of the supplied arrays.
    """
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (query, key, value))
    for name, array in (("query", q), ("key", k), ("value", v)):
        if array.ndim != 2 or 0 in array.shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a nonempty finite matrix")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Q/K feature dimensions and K/V row counts must match")
    scores = (q / np.sqrt(q.shape[1])) @ k.T
    if not np.all(np.isfinite(scores)):
        raise ValueError("attention scores exceed the float64 range")
    allowed = np.ones(scores.shape, dtype=np.bool_)
    if mask is not None:
        raw_mask = np.asarray(mask)
        if raw_mask.dtype != np.bool_:
            raise ValueError("mask must be boolean; True means allowed")
        try:
            allowed = np.broadcast_to(raw_mask, scores.shape)
        except ValueError as exc:
            raise ValueError("mask must broadcast to (queries, keys)") from exc
    scores = np.where(allowed, scores, -np.inf)
    row_max = np.max(scores, axis=1, keepdims=True)
    row_max = np.where(np.any(allowed, axis=1, keepdims=True), row_max, 0.0)
    weights = np.exp(scores - row_max)
    totals = np.sum(weights, axis=1, keepdims=True)
    np.divide(weights, totals, out=weights, where=totals != 0)
    return weights @ v, weights
