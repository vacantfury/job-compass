"""Binary precision, recall, F1, and ROC-AUC with explicit undefined cases."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _binary_vector(values: ArrayLike, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be a nonempty vector containing only 0 and 1")
    return array.astype(np.int64)


def classification_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, *, zero_division: float = 0.0
) -> dict[str, float]:
    """Return binary precision, recall, and F1 from hard labels.

    An undefined metric receives zero_division, which must be 0 or 1.
    F1 uses counts directly, so its zero-denominator case is independent
    of the precision and recall policies. Label 1 is the positive class.
    """
    truth = _binary_vector(y_true, "y_true")
    predicted = _binary_vector(y_pred, "y_pred")
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1")
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else float(zero_division)

    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def roc_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Return P(positive score > negative score) + half the tie probability.

    Larger finite scores mean stronger evidence for label 1. Sort once and
    scan equal-score groups. Raise ValueError if either class is absent,
    because ROC-AUC is undefined in that case.
    """
    truth = _binary_vector(y_true, "y_true")
    score = np.asarray(scores, dtype=np.float64)
    if score.shape != truth.shape or not np.all(np.isfinite(score)):
        raise ValueError("scores must be a finite vector matching y_true")
    positives = int(np.sum(truth))
    negatives = truth.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both positive and negative labels")
    order = np.argsort(score, kind="stable")
    sorted_scores, sorted_labels = score[order], truth[order]
    favorable = 0.0
    negatives_before = 0
    start = 0
    while start < truth.size:
        end = start + 1
        while end < truth.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_positives = int(np.sum(sorted_labels[start:end]))
        group_negatives = end - start - group_positives
        favorable += group_positives * (negatives_before + 0.5 * group_negatives)
        negatives_before += group_negatives
        start = end
    return float(favorable / (positives * negatives))
