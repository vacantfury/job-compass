"""Check hand counts, undefined cases, and an independent pairwise AUC oracle."""

import numpy as np
import pytest


@pytest.fixture
def solution(load_ml_solution):
    return load_ml_solution("classification_metrics.py")


def test_metrics_match_hand_counts(solution):
    # TP=2, FP=1, FN=2, TN=1.
    result = solution.classification_metrics([1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0])
    assert result == pytest.approx({"precision": 2 / 3, "recall": 0.5, "f1": 4 / 7})


@pytest.mark.parametrize(("truth", "predicted", "policy", "expected"), [
    ([0, 0], [0, 0], 0, {"precision": 0, "recall": 0, "f1": 0}),
    ([0, 0], [0, 0], 1, {"precision": 1, "recall": 1, "f1": 1}),
    ([1, 0], [0, 0], 1, {"precision": 1, "recall": 0, "f1": 0}),
    ([0, 0], [1, 0], 1, {"precision": 0, "recall": 1, "f1": 0}),
    ([1, 0], [1, 0], 0, {"precision": 1, "recall": 1, "f1": 1}),
])
def test_zero_division_policies(solution, truth, predicted, policy, expected):
    assert solution.classification_metrics(truth, predicted, zero_division=policy) == expected


@pytest.mark.parametrize(("truth", "scores", "expected"), [
    ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.75),
    ([0, 1, 1, 0], [0.1, 0.5, 0.5, 0.5], 0.75),
    ([0, 1], [5, 5], 0.5),
    ([0, 1, 0, 1], [1, 1, 1, 1], 0.5),
    ([0, 1], [-100, 100], 1),
    ([0, 1], [100, -100], 0),
])
def test_auc_hand_examples(solution, truth, scores, expected):
    assert solution.roc_auc(truth, scores) == expected


def test_auc_matches_pairwise_oracle_and_order_invariance(solution):
    rng = np.random.default_rng(12)
    truth = np.tile([0, 1], 12)
    scores = rng.integers(-2, 3, size=truth.size)
    original = scores.copy()
    positives, negatives = scores[truth == 1], scores[truth == 0]
    favorable = sum(float(p > n) + 0.5 * float(p == n) for p in positives for n in negatives)
    expected = favorable / (len(positives) * len(negatives))
    assert solution.roc_auc(truth, scores) == pytest.approx(expected)
    permutation = rng.permutation(truth.size)
    assert solution.roc_auc(truth[permutation], scores[permutation]) == pytest.approx(expected)
    assert solution.roc_auc(truth, scores * 3 + 8) == pytest.approx(expected)
    assert solution.roc_auc(truth, -scores) == pytest.approx(1 - expected)
    np.testing.assert_array_equal(scores, original)


@pytest.mark.parametrize(("truth", "scores"), [
    ([0, 0], [0.2, 0.4]), ([1, 1], [0.2, 0.4]), ([], []),
    ([0, 1], [0.2]), ([0, 1], [0, np.nan]), ([0, 1], [0, np.inf]),
    ([0, 2], [0, 1]), ([[0, 1]], [0, 1]),
])
def test_invalid_auc_inputs(solution, truth, scores):
    with pytest.raises(ValueError):
        solution.roc_auc(truth, scores)


@pytest.mark.parametrize(("truth", "predicted", "policy"), [
    ([], [], 0), ([0], [0, 1], 0), ([0, 1], [0, 0.5], 0),
    ([0], [0], 0.5), ([np.nan], [0], 0),
])
def test_invalid_metric_inputs(solution, truth, predicted, policy):
    with pytest.raises(ValueError):
        solution.classification_metrics(truth, predicted, zero_division=policy)
