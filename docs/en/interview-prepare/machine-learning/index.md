---
title: Machine Learning Interview Implementations
description: Six NumPy implementations with mathematical intuition, numerical safeguards, walkthroughs, and tested source code.
---
# Machine Learning Interview Implementations

These six problems cover common requests to implement machine-learning operations from scratch. Each page includes an interviewer-style question, an original example, brief math, a key idea, a walkthrough, complexity, numerical pitfalls, and follow-up questions.

## Binary classification

- [Logistic Regression](/en/interview-prepare/machine-learning/logistic-regression): compute stable probabilities and binary cross-entropy, then train with batch gradient descent.

## Clustering

- [K-Means Clustering](/en/interview-prepare/machine-learning/k-means): initialize with seeded k-means++ and alternate nearest-center assignments with mean updates.

## Multiclass loss

- [Softmax and Cross-Entropy](/en/interview-prepare/machine-learning/softmax-cross-entropy): normalize logits stably and derive the loss gradient without a full Jacobian.

## Attention

- [Scaled Dot-Product Attention](/en/interview-prepare/machine-learning/scaled-dot-product-attention): mix values using masked query-key similarities and explain the extension to multiple heads.

## Evaluation

- [Classification Metrics](/en/interview-prepare/machine-learning/classification-metrics): compute precision, recall, F1, and ROC-AUC with explicit ties and zero-denominator policies.

## Regression and optimization

- [Linear Regression with Ridge Regularization](/en/interview-prepare/machine-learning/linear-regression): solve a regularized normal equation and compare it with gradient descent on the same objective.

## How to practice

1. State the shapes, positive-class convention, reduction, and output before coding.
2. Derive the objective or invariant and trace a small hand-calculated example.
3. Implement the operation with NumPy and explain its numerical safeguards.
4. Test extreme logits, repeated points or scores, masks, invalid shapes, and undefined cases.
5. Derive time and space, then discuss how the algorithm changes at larger scale.

The solution files use only NumPy and Python's standard syntax. They require Python 3.11 or later with the NumPy version pinned in `requirements-dev.txt`. Pages import the complete solution files directly; tests under `tests/machine_learning/` load those same files. Each page states its behavior for edge cases beyond the core interview question. Learning rates, tolerances, and iteration limits are exposed as function arguments; tune them for the input scale rather than treating the defaults as universal guarantees.
