"""Check scaled attention, masking semantics, and fully masked rows."""

import numpy as np
import pytest


@pytest.fixture
def solution(load_ml_solution):
    return load_ml_solution("scaled_dot_product_attention.py")


def test_hand_computed_attention_includes_scaling(solution):
    # QK^T / sqrt(4) = [log(3), 0], giving weights [3/4, 1/4].
    q = [[2 * np.log(3), 0, 0, 0]]
    k = [[1, 0, 0, 0], [0, 0, 0, 0]]
    output, weights = solution.scaled_dot_product_attention(q, k, [[2, 4], [6, 0]])
    np.testing.assert_allclose(weights, [[0.75, 0.25]])
    np.testing.assert_allclose(output, [[3, 3]])


def test_uniform_weights_average_values(solution):
    output, weights = solution.scaled_dot_product_attention([[0], [0]], [[1], [9]], [[2], [6]])
    np.testing.assert_allclose(weights, [[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(output, [[4], [4]])


def test_causal_mask_and_input_preservation(solution):
    q = np.ones((3, 2))
    mask = np.tril(np.ones((3, 3), dtype=bool))
    q_before, mask_before = q.copy(), mask.copy()
    output, weights = solution.scaled_dot_product_attention(q, q, [[1], [3], [8]], mask)
    np.testing.assert_array_equal(weights[~mask], 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    np.testing.assert_allclose(output[:, 0], [1, 2, 4])
    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_broadcast_key_mask_and_all_masked_row(solution):
    q, k, v = [[1], [2]], [[1000], [-1000]], [[3], [7]]
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        output, weights = solution.scaled_dot_product_attention(q, k, v, [False, True])
        empty_output, empty_weights = solution.scaled_dot_product_attention(
            q, k, v, [[False, False], [True, False]]
        )
    np.testing.assert_array_equal(weights, [[0, 1], [0, 1]])
    np.testing.assert_array_equal(output, [[7], [7]])
    np.testing.assert_array_equal(empty_weights, [[0, 0], [1, 0]])
    np.testing.assert_array_equal(empty_output, [[0], [3]])


def test_large_scores_remain_stable(solution):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        output, weights = solution.scaled_dot_product_attention([[1000]], [[1000], [-1000]], [[2], [6]])
    np.testing.assert_array_equal(weights, [[1, 0]])
    np.testing.assert_array_equal(output, [[2]])


@pytest.mark.parametrize(("q", "k", "v", "mask"), [
    ([], [[1]], [[1]], None),
    ([[1, 2]], [[1]], [[1]], None),
    ([[1]], [[1]], [[1], [2]], None),
    ([[np.nan]], [[1]], [[1]], None),
    ([[1]], [[1]], [[np.inf]], None),
    ([[1]], [[1]], [[1]], [[1]]),
    ([[1]], [[1]], [[1]], [[True, False]]),
])
def test_invalid_inputs(solution, q, k, v, mask):
    with pytest.raises(ValueError):
        solution.scaled_dot_product_attention(q, k, v, mask)
