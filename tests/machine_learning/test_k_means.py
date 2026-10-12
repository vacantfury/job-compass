"""Check seeded initialization, Lloyd updates, and degenerate clusters."""

import numpy as np
import pytest


@pytest.fixture
def solution(load_ml_solution):
    return load_ml_solution("k_means.py")


def test_hand_computed_clusters_and_inertia(solution):
    X = np.array([[0, 0], [0, 2], [10, 0], [10, 2]])
    original = X.copy()
    centers, labels, inertia = solution.k_means(X, 2, seed=7)
    np.testing.assert_allclose(centers[np.argsort(centers[:, 0])], [[0, 1], [10, 1]])
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert inertia == pytest.approx(4)
    for index, center in enumerate(centers):
        np.testing.assert_allclose(center, X[labels == index].mean(axis=0))
    np.testing.assert_array_equal(X, original)


def test_seed_reproduces_initialization_and_fit_without_global_rng_changes(solution):
    X = np.random.default_rng(10).normal(size=(20, 3))
    global_before = np.random.get_state()
    initial = solution.kmeans_plus_plus(X, 4, seed=29)
    np.testing.assert_array_equal(initial, solution.kmeans_plus_plus(X, 4, seed=29))
    assert len(np.unique(initial, axis=0)) == 4
    assert all(np.any(np.all(X == center, axis=1)) for center in initial)
    first, second = solution.k_means(X, 4, seed=29), solution.k_means(X, 4, seed=29)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    global_after = np.random.get_state()
    for left, right in zip(global_before, global_after):
        np.testing.assert_array_equal(left, right)


def test_single_cluster_and_single_sample(solution):
    centers, labels, inertia = solution.k_means([[0], [2], [4]], 1, seed=1)
    np.testing.assert_allclose(centers, [[2]])
    np.testing.assert_array_equal(labels, [0, 0, 0])
    assert inertia == 8
    centers, labels, inertia = solution.k_means([[3, 4]], 1)
    np.testing.assert_array_equal(centers, [[3, 4]])
    np.testing.assert_array_equal(labels, [0])
    assert inertia == 0


def test_duplicate_points_and_empty_clusters(solution):
    centers, labels, inertia = solution.k_means([[2, 3]] * 4, 3, seed=4)
    np.testing.assert_array_equal(centers, [[2, 3]] * 3)
    np.testing.assert_array_equal(labels, [0, 0, 0, 0])
    assert inertia == 0


def test_final_labels_match_centers_even_at_iteration_cap(solution):
    X = np.random.default_rng(21).normal(size=(30, 2))
    centers, labels, inertia = solution.k_means(X, 3, seed=5, max_iter=1, tol=0)
    distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    np.testing.assert_array_equal(labels, distances.argmin(axis=1))
    assert inertia == pytest.approx(distances.min(axis=1).sum())
    initial = solution.kmeans_plus_plus(X, 3, seed=5)
    initial_inertia = np.sum(np.min(np.sum((X[:, None] - initial) ** 2, axis=2), axis=1))
    assert inertia <= initial_inertia + 1e-12
    _, _, later_inertia = solution.k_means(X, 3, seed=5, max_iter=50, tol=0)
    assert later_inertia <= inertia + 1e-12


def test_large_tolerance_stops_after_first_update(solution):
    X = [[0], [1], [4], [8], [9]]
    capped = solution.k_means(X, 2, seed=0, max_iter=1)
    stopped = solution.k_means(X, 2, seed=0, tol=1e6)
    for left, right in zip(capped, stopped):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("kwargs", [
    {"n_clusters": 0}, {"n_clusters": 3}, {"n_clusters": 1.5},
    {"n_clusters": True}, {"n_clusters": 1, "max_iter": 0},
    {"n_clusters": 1, "max_iter": 2.5}, {"n_clusters": 1, "tol": -1},
    {"n_clusters": 1, "tol": np.nan},
])
def test_invalid_controls(solution, kwargs):
    with pytest.raises(ValueError):
        solution.k_means([[0], [1]], **kwargs)


@pytest.mark.parametrize("X", [[], [[]], [1, 2], [[np.nan]], [[np.inf]]])
def test_invalid_data(solution, X):
    with pytest.raises(ValueError):
        solution.k_means(X, 1)
