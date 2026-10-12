"""Check the ridge objective, intercept policy, rank deficiency, and GD agreement."""

import numpy as np
import pytest


@pytest.fixture
def solution(load_ml_solution):
    return load_ml_solution("linear_regression.py")


@pytest.mark.parametrize("method", ["ridge_regression", "gradient_descent"])
@pytest.mark.parametrize(("l2", "expected_weight"), [(0, 2), (1, 1), (3, 0.5)])
def test_hand_computed_fit_and_unpenalized_intercept(solution, method, l2, expected_weight):
    # Mean x=0, mean y=3, variance(x)=1, covariance(x,y)=2.
    w, b = getattr(solution, method)([[-1], [1]], [1, 5], l2=l2)
    np.testing.assert_allclose(w, [expected_weight], atol=1e-7)
    assert b == pytest.approx(3, abs=1e-7)


@pytest.mark.parametrize("l2", [0.0, 0.1, 2.0])
def test_gradient_descent_matches_normal_equation(solution, l2):
    X = np.array([[0, 1], [1, 0], [2, -1], [-1, 2], [1, 2]], dtype=float)
    y = np.array([2, 4, 5, -1, 6], dtype=float)
    original_X, original_y = X.copy(), y.copy()
    closed_w, closed_b = solution.ridge_regression(X, y, l2=l2)
    gd_w, gd_b = solution.gradient_descent(X, y, l2=l2, learning_rate=0.1, tol=1e-10)
    np.testing.assert_allclose(gd_w, closed_w, atol=1e-7)
    assert gd_b == pytest.approx(closed_b, abs=1e-7)
    residual = X @ closed_w + closed_b - y
    np.testing.assert_allclose(X.T @ residual / len(y) + l2 * closed_w, 0, atol=1e-12)
    assert residual.mean() == pytest.approx(0, abs=1e-12)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


def test_one_batch_update(solution):
    w, b = solution.gradient_descent([[0], [2]], [1, 5], l2=1, max_iter=1, learning_rate=0.1)
    np.testing.assert_allclose(w, [0.5])
    assert b == pytest.approx(0.3)


def test_unregularized_rank_deficient_design(solution):
    X = np.array([[-1, -1], [0, 0], [1, 1]], dtype=float)
    w, b = solution.ridge_regression(X, [1, 3, 5], l2=0)
    np.testing.assert_allclose(w, [1, 1])
    np.testing.assert_allclose(X @ w + b, [1, 3, 5])
    gd_w, gd_b = solution.gradient_descent(X, [1, 3, 5], l2=0)
    np.testing.assert_allclose(gd_w, w, atol=1e-7)
    assert gd_b == pytest.approx(b, abs=1e-7)


def test_duplicating_samples_preserves_mean_loss_regularization(solution):
    X, y = [[-2], [1], [3]], [0, 3, 4]
    first = solution.ridge_regression(X, y, l2=2)
    repeated = solution.ridge_regression(X * 2, y * 2, l2=2)
    for left, right in zip(first, repeated):
        np.testing.assert_allclose(left, right)


@pytest.mark.parametrize("method", ["ridge_regression", "gradient_descent"])
@pytest.mark.parametrize(("X", "y", "l2"), [
    ([], [], 1), ([1, 2], [1, 2], 1), ([[1]], [1, 2], 1),
    ([[np.nan]], [1], 1), ([[1]], [np.inf], 1),
    ([[1]], [1], -1), ([[1]], [1], np.nan),
])
def test_invalid_data_and_regularization(solution, method, X, y, l2):
    with pytest.raises(ValueError):
        getattr(solution, method)(X, y, l2=l2)


@pytest.mark.parametrize("kwargs", [
    {"learning_rate": 0}, {"learning_rate": np.inf}, {"max_iter": 0},
    {"max_iter": 1.5}, {"max_iter": True}, {"tol": -1}, {"tol": np.nan},
])
def test_invalid_descent_controls(solution, kwargs):
    with pytest.raises(ValueError):
        solution.gradient_descent([[0], [1]], [0, 1], **kwargs)
