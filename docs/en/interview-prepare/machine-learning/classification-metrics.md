---
title: Classification Metrics
description: Compute binary precision, recall, F1, and score-based ROC-AUC with explicit ties and undefined cases.
---
# Classification Metrics

## Problem in plain words

Implement binary precision, recall, and F1 from true and predicted labels using only NumPy. Also implement ROC-AUC from true labels and real-valued scores, where a larger score means stronger evidence for the positive class. Handle tied scores and zero denominators explicitly.

For example, true labels `[1, 1, 1, 1, 0, 0]` and predictions `[1, 1, 0, 0, 1, 0]` have two true positives, one false positive, and two false negatives. Precision is `2/3`, recall is `1/2`, and F1 is `4/7`.

This implementation accepts nonempty binary label vectors of equal length, with label `1` designated positive. `classification_metrics` returns a dictionary with `precision`, `recall`, and `f1`. A zero denominator returns `zero_division`, which defaults to `0` and may be set to `1`. `roc_auc` requires finite scores and both classes; it raises `ValueError` when AUC is undefined. Inputs are not changed.

## Math in brief

```text
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * TP / (2 * TP + FP + FN)
AUC = (winning positive-negative pairs + 0.5 * tied pairs)
      / (number of positives * number of negatives)
```

For AUC, a positive wins when its score is greater than a negative's score. This pairwise definition is equivalent to the area under the ROC curve when tied thresholds are grouped and straight segments join successive ROC points.

## Key idea

Thresholded metrics need only confusion counts. AUC needs the ordering of scores, not a chosen decision threshold. Sort scores ascending and process equal-score groups together. Each positive in a group beats all negatives in earlier groups and ties every negative in its own group.

## Step by step

1. Validate the binary vectors and count true positives, false positives, and false negatives.
2. Compute precision, recall, and F1 directly from those counts, applying the zero-denominator policy to each metric separately.
3. For AUC, validate scores and require at least one example of each class.
4. Sort labels by ascending score and start a count of negatives in earlier groups.
5. For each equal-score group, add `group_positives * (earlier_negatives + 0.5 * group_negatives)` to the favorable pair count.
6. Add the group's negative count to the earlier-negative total and continue.
7. Divide favorable pairs by the total number of positive-negative pairs.

For labels `[0, 0, 1, 1]` with scores `[0.1, 0.4, 0.35, 0.8]`, the positive at `0.35` beats one negative, and the positive at `0.8` beats both. Three of four pairs win, so AUC is `0.75`. If every score is equal and both classes exist, every pair ties and AUC is `0.5`.

## Python solution

<<< ./solutions/classification_metrics.py

## Complexity

Precision, recall, and F1 take **O(n)** time and **O(n)** auxiliary space for NumPy's vectorized masks and label copies. AUC takes **O(n log n)** time for sorting plus **O(n)** for the grouped scan, with **O(n)** auxiliary space. Explicitly enumerating all positive-negative pairs would take **O(n_positive * n_negative)** time.

## Common mistakes

- Passing thresholded predictions to AUC throws away ranking information. Supply the scores for the positive class; probabilities are not required.
- Counting ties as all wins or all losses makes AUC depend on arbitrary ordering. Each tied pair contributes one half.
- Treating approximately equal floats as ties changes the metric. This implementation groups exactly equal scores.
- One-class AUC is undefined. Returning zero would confuse missing information with a completely reversed ranking.
- When there are actual positives but no predicted positives, precision may be undefined while recall and F1 are valid zeros. Compute F1 from counts so a configurable precision fallback does not distort it.
- AUC is unchanged by strictly increasing score transformations. It measures ranking, not probability calibration or performance at a particular threshold.

## Interview follow-ups

- How would you compute the full ROC curve while handling tied thresholds together?
- How do micro, macro, and weighted averaging extend precision, recall, and F1 to multiple classes?
- When might a precision-recall curve be more informative than ROC-AUC?
- Which quantities can be updated from streaming confusion counts, and why does exact AUC need more information?
