"""Seeded k-means++ initialization followed by Lloyd's assignment/update steps."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _data_and_k(X: ArrayLike, n_clusters: int) -> NDArray[np.float64]:
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2 or 0 in data.shape or not np.all(np.isfinite(data)):
        raise ValueError("X must be a nonempty finite (samples, features) matrix")
    if isinstance(n_clusters, (bool, np.bool_)) or not isinstance(n_clusters, (int, np.integer)):
        raise ValueError("n_clusters must be an integer")
    if not 1 <= n_clusters <= data.shape[0]:
        raise ValueError("n_clusters must be between 1 and the number of samples")
    return data


def _squared_distances(
    data: NDArray[np.float64], centers: NDArray[np.float64]
) -> NDArray[np.float64]:
    # One center at a time avoids allocating a (samples, clusters, features) cube.
    distances = np.empty((data.shape[0], centers.shape[0]), dtype=np.float64)
    for index, center in enumerate(centers):
        delta = data - center
        distances[:, index] = np.einsum("ij,ij->i", delta, delta)
    if not np.all(np.isfinite(distances)):
        raise ValueError("squared distances exceed the float64 range; rescale X")
    return distances


def kmeans_plus_plus(
    X: ArrayLike, n_clusters: int, *, seed: int | None = None
) -> NDArray[np.float64]:
    """Choose data rows with probability proportional to nearest squared distance.

    The first center is uniform. If all distances become zero (fewer distinct
    points than clusters), choose another row uniformly; duplicate centers
    are allowed. A local generator leaves NumPy's global RNG untouched.
    """
    data = _data_and_k(X, n_clusters)
    rng = np.random.default_rng(seed)
    centers = np.empty((n_clusters, data.shape[1]), dtype=np.float64)
    centers[0] = data[rng.integers(data.shape[0])]
    nearest = _squared_distances(data, centers[:1])[:, 0]
    for index in range(1, n_clusters):
        scale = float(np.max(nearest))
        if scale == 0:
            chosen = rng.integers(data.shape[0])
        else:
            scaled = nearest / scale
            chosen = rng.choice(data.shape[0], p=scaled / np.sum(scaled))
        centers[index] = data[chosen]
        distance = _squared_distances(data, centers[index:index + 1])[:, 0]
        nearest = np.minimum(nearest, distance)
    return centers


def k_means(
    X: ArrayLike,
    n_clusters: int,
    *,
    seed: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.int64], float]:
    """Return (centers, labels, inertia) without modifying X.

    Stop when the maximum Euclidean center movement is <= tol, or at the
    iteration cap. Empty clusters retain their previous center. Distance
    ties select the lowest center index. Final labels always refer to the
    returned centers, even when the iteration cap interrupts convergence.
    """
    data = _data_and_k(X, n_clusters)
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be nonnegative and finite")
    centers = kmeans_plus_plus(data, n_clusters, seed=seed)
    for _ in range(max_iter):
        labels = np.argmin(_squared_distances(data, centers), axis=1)
        updated = centers.copy()
        for index in range(n_clusters):
            members = data[labels == index]
            if members.shape[0]:
                updated[index] = np.mean(members, axis=0)
        movement = float(np.max(np.linalg.norm(updated - centers, axis=1)))
        centers = updated
        if movement <= tol:
            break
    distances = _squared_distances(data, centers)
    labels = np.argmin(distances, axis=1).astype(np.int64)
    inertia = float(np.sum(distances[np.arange(data.shape[0]), labels]))
    return centers, labels, inertia
