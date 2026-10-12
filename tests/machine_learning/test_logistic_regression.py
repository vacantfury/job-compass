"""Check stable BCE, analytical gradients, and full-batch training."""

import numpy as np
import pytest


@pytest.fixture
def solution(load_ml_solution):
    return load_ml_solution("logistic_regression.py")


def test_sigmoid_and_loss_at_extreme_logits(solution):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        np.testing.assert_allclose(solution.sigmoid([-1000, 0, 1000]), [0, 0.5, 1])
        assert solution.binary_cross_entropy([-1000, 1000], [1, 0]) == 1000
        assert solution.binary_cross_entropy([-1000, 1000], [0, 1]) == 0
        assert solution.binary_cross_entropy([50], [1]) > 0
    assert solution.sigmoid(0) == 0.5


def test_hand_computed_gradient_and_first_update(solution):
    X, y = [[0], [2]], [0, 1]
    loss, dw, db = solution.loss_and_gradient(X, y, [0], 0)
    assert loss == pytest.approx(np.log(2))
    np.testing.assert_allclose(dw, [-0.5])
    assert db == 0
    w, b = solution.fit_logistic_regression(X, y, max_iter=1, learning_rate=0.1)
    np.testing.assert_allclose(w, [0.05])
    assert b == 0


def test_gradient_matches_finite_differences(solution):
    X = np.array([[1, -2], [0.5, 3], [-1, 1]])
    y, theta = [1, 0, 1], np.array([0.3, -0.2, 0.4])
    _, dw, db = solution.loss_and_gradient(X, y, theta[:2], theta[2])
    numerical = []
    for index in range(3):
        step = np.zeros(3)
        step[index] = 1e-6
        plus, minus = theta + step, theta - step
        high = solution.binary_cross_entropy(X @ plus[:2] + plus[2], y)
        low = solution.binary_cross_entropy(X @ minus[:2] + minus[2], y)
        numerical.append((high - low) / 2e-6)
    np.testing.assert_allclose(np.r_[dw, db], numerical, atol=1e-8)


def test_separates_toy_data_and_learns_intercept(solution):
    X = np.array([[0.0], [1.0], [3.0], [4.0]])
    original, y = X.copy(), np.array([0, 0, 1, 1])
    w, b = solution.fit_logistic_regression(X, y, learning_rate=0.2, max_iter=2000)
    np.testing.assert_array_equal(solution.predict(X, w, b), y)
    assert b < 0
    assert solution.binary_cross_entropy(X @ w + b, y) < 0.05
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(solution.predict([[0]], [0], 0), [1])


def test_zero_gradient_stops_at_initial_parameters(solution):
    w, b = solution.fit_logistic_regression([[0], [0]], [0, 1])
    np.testing.assert_array_equal(w, [0])
    assert b == 0


@pytest.mark.parametrize("kwargs", [
    {"learning_rate": 0}, {"learning_rate": np.inf}, {"tol": -1},
    {"tol": np.nan}, {"max_iter": 0}, {"max_iter": 1.5}, {"max_iter": True},
])
def test_rejects_invalid_training_controls(solution, kwargs):
    with pytest.raises(ValueError):
        solution.fit_logistic_regression([[0], [1]], [0, 1], **kwargs)


@pytest.mark.parametrize(("X", "y"), [
    ([], []), ([1, 2], [0, 1]), ([[1], [2]], [0]),
    ([[1], [2]], [0, 2]), ([[np.nan]], [0]), ([[1]], [[1]]),
])
def test_rejects_invalid_data(solution, X, y):
    with pytest.raises(ValueError):
        solution.fit_logistic_regression(X, y)


def test_rejects_invalid_prediction_and_loss_inputs(solution):
    for weights, bias in [([1, 2], 0), ([np.nan], 0), ([0], np.inf)]:
        with pytest.raises(ValueError):
            solution.predict_proba([[1]], weights, bias)
    with pytest.raises(ValueError):
        solution.predict([[1]], [1], 0, threshold=1.1)
    for logits, labels in [([], []), ([np.inf], [1]), ([1], [0.5])]:
        with pytest.raises(ValueError):
            solution.binary_cross_entropy(logits, labels)
