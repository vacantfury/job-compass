"""Stable row-wise softmax and mean cross-entropy for integer class labels."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _log_probabilities(logits: ArrayLike) -> NDArray[np.float64]:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or 0 in z.shape or not np.all(np.isfinite(z)):
        raise ValueError("logits must be a nonempty finite (samples, classes) matrix")
    shifted = z - np.max(z, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def softmax(logits: ArrayLike) -> NDArray[np.float64]:
    """Return probabilities of shape (samples, classes), normalized by row."""
    return np.exp(_log_probabilities(logits))


def softmax_cross_entropy(
    logits: ArrayLike, labels: ArrayLike
) -> tuple[float, NDArray[np.float64]]:
    """Return mean loss and its gradient with respect to the input logits.

    labels must be an integer vector with values in [0, classes). The loss
    is calculated from log probabilities, never from log(rounded softmax).
    """
    log_probs = _log_probabilities(logits)
    n, classes = log_probs.shape
    y = np.asarray(labels)
    if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be an integer vector matching the sample count")
    if np.any(y < 0) or np.any(y >= classes):
        raise ValueError("labels must be valid class indices")
    rows = np.arange(n)
    loss = float(-np.mean(log_probs[rows, y]))
    gradient = np.exp(log_probs)
    gradient[rows, y] -= 1.0
    gradient /= n
    return loss, gradient
