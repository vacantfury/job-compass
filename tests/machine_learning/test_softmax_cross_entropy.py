"""Check normalization, stable log loss, and gradients against finite differences."""

import numpy as np
import pytest


@pytest.fixture
def solution(load_ml_solution):
    return load_ml_solution("softmax_cross_entropy.py")


def test_hand_computed_probabilities_loss_and_gradient(solution):
    logits = np.log([[1, 2, 1], [3, 1, 2]])
    original = logits.copy()
    expected = np.array([[0.25, 0.5, 0.25], [0.5, 1 / 6, 1 / 3]])
    np.testing.assert_allclose(solution.softmax(logits), expected)
    loss, gradient = solution.softmax_cross_entropy(logits, [1, 0])
    assert loss == pytest.approx(np.log(2))
    np.testing.assert_allclose(gradient, [[0.125, -0.25, 0.125], [-0.25, 1 / 12, 1 / 6]])
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-15)
    np.testing.assert_array_equal(logits, original)


def test_stability_and_row_shift_invariance(solution):
    logits = np.array([[10000, 10001, 9999], [-10000, -9999, -10001]])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        probs = solution.softmax(logits)
        loss, gradient = solution.softmax_cross_entropy([[1000, -1000]], [1])
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    np.testing.assert_allclose(probs[0], probs[1])
    np.testing.assert_allclose(probs, solution.softmax(logits + [[12345], [-23456]]))
    assert loss == 2000
    np.testing.assert_array_equal(gradient, [[1, -1]])


def test_gradient_matches_finite_differences(solution):
    logits = np.array([[0.5, -1, 2], [1, 0.3, -0.1]])
    _, gradient = solution.softmax_cross_entropy(logits, [2, 1])
    numerical = np.empty_like(logits)
    for index in np.ndindex(logits.shape):
        step = np.zeros_like(logits)
        step[index] = 1e-6
        high, _ = solution.softmax_cross_entropy(logits + step, [2, 1])
        low, _ = solution.softmax_cross_entropy(logits - step, [2, 1])
        numerical[index] = (high - low) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=1e-8)


def test_single_class(solution):
    loss, gradient = solution.softmax_cross_entropy([[100], [-100]], [0, 0])
    assert loss == 0
    np.testing.assert_array_equal(gradient, [[0], [0]])


@pytest.mark.parametrize("logits", [[], [[]], [1, 2], [[np.inf]], [[np.nan]]])
def test_invalid_logits(solution, logits):
    with pytest.raises(ValueError):
        solution.softmax(logits)


@pytest.mark.parametrize("labels", [[0, 1], [[1]], [0.5], [1.0], [-1], [2], [True]])
def test_invalid_labels(solution, labels):
    with pytest.raises(ValueError):
        solution.softmax_cross_entropy([[1, 2]], labels)
