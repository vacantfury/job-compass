---
title: Logistic Regression
description: Train a binary classifier from scratch with stable sigmoid probabilities and batch gradient descent.
---
# Logistic Regression

## Problem in plain words

Implement binary logistic regression using only NumPy. Given a feature matrix `X` of shape `(n, d)` and labels `y` of shape `(n,)`, learn a weight vector and an intercept with batch gradient descent. Implement sigmoid, binary cross-entropy, the loss gradient, probabilities, and class predictions.

For example, with `X = [[0], [2]]`, `y = [0, 1]`, and zero parameters, both probabilities are `0.5`. One update with learning rate `0.1` produces weight `0.05` and intercept `0`.

This implementation requires nonempty finite real features and labels in `{0, 1}`. It returns `(weights, bias)` without changing the inputs. Predictions use `probability >= threshold`, with a default threshold of `0.5`. There is no regularization.

## Math in brief

For each row, `z = Xw + b` and `p = sigmoid(z) = 1 / (1 + exp(-z))`. The objective is mean binary cross-entropy:

```text
L = -mean(y * log(p) + (1 - y) * log(1 - p))
dw = X.T @ (p - y) / n
db = mean(p - y)
w = w - learning_rate * dw
b = b - learning_rate * db
```

The equivalent per-example loss is `logaddexp(0, -z)` for label `1` and `logaddexp(0, z)` for label `0`. This computes the loss directly from logits, without taking the logarithm of a probability rounded to zero or one.

## Key idea

The derivative of sigmoid combined with binary cross-entropy simplifies to `p - y` with respect to a logit. Multiply that residual by each feature and average to obtain the weight gradient. The intercept gradient is the average residual.

## Step by step

1. Validate feature and label shapes, the learning rate, iteration limit, and tolerance.
2. Initialize the weights and intercept to zero.
3. Compute every example's logit and probability. Evaluate sigmoid using `exp(-abs(z))`, whose exponent is never positive.
4. Compute the full batch's residuals, weight gradient, and intercept gradient.
5. Stop if the largest absolute gradient component is at most `tol`. Otherwise, subtract the scaled gradients and repeat up to `max_iter` times.
6. Use the learned parameters to return probabilities or thresholded binary labels.

In the example, residuals are `[0.5, -0.5]`. The weight gradient is `(0 * 0.5 + 2 * -0.5) / 2 = -0.5`, and the intercept gradient is zero. Subtracting the gradient increases the weight, making the positive example more likely.

## Python solution

<<< ./solutions/logistic_regression.py

## Complexity

For `n` samples, `d` features, and `T` iterations, training takes **O(Tnd)** time. Prediction and one loss/gradient evaluation take **O(nd)** time. With an existing float64 feature matrix, working arrays use **O(n + d)** auxiliary space. Converting another input representation can additionally require **O(nd)** space.

## Common mistakes

- Directly computing `exp(-z)` can overflow for large negative logits. Computing `log(sigmoid(z))` can still fail after sigmoid rounds to zero.
- Using a summed gradient with a mean loss changes the effective learning rate by a factor of `n`.
- Labels of shape `(n, 1)` can broadcast residuals into an unintended `(n, n)` matrix.
- Forgetting the intercept makes an offset decision boundary impossible to learn.
- A large learning rate or badly scaled features can cause unstable updates. The default rate is an example, not a convergence guarantee; tune it on the data.
- On strictly separable data, unregularized logistic regression has no finite loss-minimizing parameter vector. An iteration cap matters even when classification is already perfect.
- Stable exponentials do not prevent overflow in `X @ w` when features or coefficients exceed float64's useful range. Rescale the inputs.

## Interview follow-ups

- How do L2 regularization and its gradient change the objective? Should the intercept be penalized?
- How would mini-batch or stochastic updates change memory use and convergence behavior?
- How would you choose the decision threshold for unequal false-positive and false-negative costs?
- Why can zero initialization work here even though it is unsuitable for all weights in a multilayer neural network?
