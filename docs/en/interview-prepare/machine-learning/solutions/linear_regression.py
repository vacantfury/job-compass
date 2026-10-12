"""Ridge regression with a normal-equation solve or batch gradient descent."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _problem(
    X: ArrayLike, y: ArrayLike, l2: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    data = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if data.ndim != 2 or 0 in data.shape or not np.all(np.isfinite(data)):
        raise ValueError("X must be a nonempty finite (samples, features) matrix")
    if target.shape != (data.shape[0],) or not np.all(np.isfinite(target)):
        raise ValueError("y must be a finite vector matching the sample count")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be nonnegative and finite")
    design = np.column_stack((np.ones(data.shape[0]), data))
    penalty = np.ones(design.shape[1])
    penalty[0] = 0.0
    return design, target, penalty


def ridge_regression(
    X: ArrayLike, y: ArrayLike, *, l2: float = 1.0
) -> tuple[NDArray[np.float64], float]:
    """Minimize ||Xw + b - y||^2 / (2n) + l2 * ||w||^2 / 2.

    The intercept is not penalized. Solve the normal equation rather than
    forming an inverse. With l2=0, use least squares directly, which also
    defines a minimum-norm answer for rank-deficient augmented designs.
    Returns (weights, bias), leaving inputs intact.
    """
    design, target, penalty = _problem(X, y, l2)
    n = design.shape[0]
    if l2 == 0:
        theta = np.linalg.lstsq(design, target, rcond=None)[0]
    else:
        hessian = design.T @ design / n + l2 * np.diag(penalty)
        theta = np.linalg.solve(hessian, design.T @ target / n)
    if not np.all(np.isfinite(theta)):
        raise ValueError("nonfinite coefficients; rescale the inputs")
    return theta[1:], float(theta[0])


def gradient_descent(
    X: ArrayLike,
    y: ArrayLike,
    *,
    l2: float = 1.0,
    learning_rate: float = 0.1,
    max_iter: int = 10000,
    tol: float = 1e-8,
) -> tuple[NDArray[np.float64], float]:
    """Minimize the same ridge objective with zero-initialized batch updates.

    Stop at gradient infinity norm <= tol or the iteration cap. A sufficient
    learning-rate bound is 0 < rate < 2 / largest_eigenvalue(H), where H is
    the objective's Hessian. A finite result at the cap is not a convergence
    guarantee. Feature scaling makes an appropriate rate easier to choose.
    """
    design, target, penalty = _problem(X, y, l2)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be nonnegative and finite")
    theta = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(max_iter):
        gradient = design.T @ (design @ theta - target) / design.shape[0]
        gradient += l2 * penalty * theta
        if not np.all(np.isfinite(gradient)):
            raise ValueError("gradient overflow; reduce learning_rate or rescale X")
        if np.max(np.abs(gradient)) <= tol:
            break
        theta -= learning_rate * gradient
    if not np.all(np.isfinite(theta)):
        raise ValueError("coefficient overflow; reduce learning_rate or rescale X")
    return theta[1:], float(theta[0])
