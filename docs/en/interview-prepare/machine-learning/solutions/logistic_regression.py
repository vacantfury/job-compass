"""Binary logistic regression with mean loss and full-batch gradient descent."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sigmoid(logits: ArrayLike) -> NDArray[np.float64]:
    """Map finite logits to probabilities without overflowing exp."""
    z = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite")
    small = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + small), small / (1.0 + small))


def _targets(y: ArrayLike, n: int) -> NDArray[np.float64]:
    target = np.asarray(y, dtype=np.float64)
    if target.shape != (n,) or not np.all((target == 0) | (target == 1)):
        raise ValueError("y must be a vector of binary labels matching X")
    return target


def _features(X: ArrayLike) -> NDArray[np.float64]:
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2 or 0 in data.shape or not np.all(np.isfinite(data)):
        raise ValueError("X must be a nonempty, finite (samples, features) matrix")
    return data


def binary_cross_entropy(logits: ArrayLike, y: ArrayLike) -> float:
    """Return mean binary cross-entropy from logits, not probabilities."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("logits must be a nonempty finite vector")
    target = _targets(y, z.size)
    signed = np.where(target == 1, -z, z)
    return float(np.mean(np.logaddexp(0.0, signed)))


def predict_proba(
    X: ArrayLike, weights: ArrayLike, bias: float
) -> NDArray[np.float64]:
    """Return P(y=1) for each row; weights has shape (features,)."""
    data = _features(X)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (data.shape[1],) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite vector matching X's features")
    if not np.isfinite(bias):
        raise ValueError("bias must be finite")
    return sigmoid(data @ w + bias)


def loss_and_gradient(
    X: ArrayLike, y: ArrayLike, weights: ArrayLike, bias: float
) -> tuple[float, NDArray[np.float64], float]:
    """Return mean BCE, its weight gradient, and its bias gradient."""
    data = _features(X)
    target = _targets(y, data.shape[0])
    error = predict_proba(data, weights, bias) - target
    logits = data @ np.asarray(weights, dtype=np.float64) + bias
    return (
        binary_cross_entropy(logits, target),
        data.T @ error / data.shape[0],
        float(np.mean(error)),
    )


def fit_logistic_regression(
    X: ArrayLike,
    y: ArrayLike,
    *,
    learning_rate: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[NDArray[np.float64], float]:
    """Fit from zero weights; stop at gradient infinity norm <= tol or the cap.

    No regularization is applied. Returns (weights, bias), leaving inputs intact.
    Scale features and tune the learning rate for the data before training.
    """
    data = _features(X)
    target = _targets(y, data.shape[0])
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be nonnegative and finite")
    weights = np.zeros(data.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(max_iter):
        error = sigmoid(data @ weights + bias) - target
        dw = data.T @ error / data.shape[0]
        db = float(np.mean(error))
        if max(float(np.max(np.abs(dw))), abs(db)) <= tol:
            break
        weights -= learning_rate * dw
        bias -= learning_rate * db
    # Also reject an overflowing final update when the iteration cap is reached.
    predict_proba(data, weights, bias)
    return weights, bias


def predict(
    X: ArrayLike, weights: ArrayLike, bias: float, *, threshold: float = 0.5
) -> NDArray[np.int64]:
    """Predict 1 when its probability is >= threshold (including equality)."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    return (predict_proba(X, weights, bias) >= threshold).astype(np.int64)
