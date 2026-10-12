---
title: Linear Regression with Ridge Regularization
description: Compare a normal-equation solve and batch gradient descent for the same ridge objective.
---
# Linear Regression with Ridge Regularization

## Problem in plain words

Implement linear regression using only NumPy in two ways: a closed-form normal-equation solve with ridge regularization, and batch gradient descent. Both methods must minimize the same objective and include an unregularized intercept. Explain when you would choose each method.

For example, `X = [[-1], [1]]` and `y = [1, 5]` fit the line `y = 2x + 3` without regularization. Under the mean-loss convention below, setting `l2 = 1` shrinks the weight to `1` while keeping the intercept at `3`.

Both functions return `(weights, bias)` and leave the inputs unchanged. Features must be a nonempty finite real `(n, d)` matrix, and targets a finite `(n,)` vector. `l2` must be nonnegative. With `l2 = 0`, the direct solver uses `numpy.linalg.lstsq` to support rank-deficient designs instead of inverting a singular matrix.

## Math in brief

Append a column of ones to form `A = [1, X]`, put the intercept first in `theta = [b, w]`, and let `R = diag(0, 1, ..., 1)`:

```text
L = ||A @ theta - y||^2 / (2n) + l2 * ||w||^2 / 2
H = A.T @ A / n + l2 * R
Normal equation: H @ theta = A.T @ y / n
Gradient: A.T @ (A @ theta - y) / n + l2 * R @ theta
Update: theta = theta - learning_rate * gradient
```

The zero entry of `R` excludes the intercept from regularization. Equivalently, the unnormalized normal equation adds `n * l2 * R` to `A.T @ A`. Omitting that factor would make the two methods optimize different objectives.

## Key idea

Setting the quadratic objective's gradient to zero gives a linear system. Solve that system directly when its size and conditioning make this practical. Gradient descent instead approaches the same optimum through repeated matrix-vector products, avoiding a dense feature-by-feature matrix.

## Step by step

1. Validate inputs and create the augmented design matrix with an intercept column.
2. For the direct ridge solution, form `H` and the right-hand side, then call `numpy.linalg.solve`. Do not explicitly form an inverse.
3. If regularization is zero, call `numpy.linalg.lstsq(A, y)` instead. A rank-deficient system receives a minimum-norm least-squares answer for the augmented parameters.
4. For gradient descent, initialize all parameters to zero and compute the residuals over the full batch.
5. Average the data gradient, add the ridge gradient only to the weights, and stop if the largest absolute gradient component is at most `tol`.
6. Otherwise, subtract the scaled gradient and repeat until convergence or `max_iter`. Return the weight vector and intercept separately.

In the example, the mean feature is zero, its mean squared value is one, and its covariance with the target is two. The optimal weight is therefore `2 / (1 + l2)`, while the intercept remains the mean target, `3`.

## Python solution

<<< ./solutions/linear_regression.py

## Complexity

For `n` samples and `d` features, the regularized normal-equation method takes **O(nd^2 + d^3)** time and **O(nd + d^2)** space, including the augmented design matrix. The unregularized least-squares branch works directly with the design matrix; its decomposition cost depends on matrix shape, typically **O(nd * min(n, d))**.

With `T` iterations, gradient descent takes **O(Tnd)** time. This implementation uses **O(nd + n + d)** space for the augmented design matrix, residuals, and parameter vectors. It does not form the **O(d^2)** Hessian.

## Common mistakes

- Explicit matrix inversion costs work and amplifies numerical error. Solve the system instead.
- Mixing a mean squared-error gradient with a sum-based regularization coefficient changes the objective as the sample count changes.
- Regularizing the intercept pulls predictions toward zero unnecessarily when the targets have a nonzero mean.
- Forming `A.T @ A` squares the design matrix's condition number before regularization. Positive ridge helps, but severely ill-conditioned data may still call for a QR/SVD solve on an augmented least-squares formulation.
- Gradient descent requires a suitable learning rate. For this quadratic, `0 < learning_rate < 2 / largest_eigenvalue(H)` is sufficient for convergence to an optimum. Feature scaling improves conditioning; the default rate is not universal.
- Reaching the iteration cap does not certify convergence. A finite answer can still be inaccurate if the rate is too small or the problem is poorly conditioned.
- Large features can overflow products even when each input is finite. Rescale data before either method.

## Interview follow-ups

- When should you use each method? Prefer a direct solve for a moderate feature count and a well-conditioned system when a precise fit is needed. Use iterative optimization when forming or factoring a dense feature matrix is too expensive, especially when extending to sparse data or mini-batches. Compare their actual costs rather than choosing by sample count alone.
- How would an augmented least-squares formulation implement ridge without forming `A.T @ A`?
- How would feature standardization affect the meaning of an L2 penalty and the recovered coefficients?
- How would mini-batch descent or conjugate gradients change the computational tradeoffs?
