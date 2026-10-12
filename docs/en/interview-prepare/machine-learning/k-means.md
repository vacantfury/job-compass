---
title: K-Means Clustering
description: Use seeded k-means++ initialization and alternating assignment and mean updates to cluster points.
---
# K-Means Clustering

## Problem in plain words

Implement k-means using only NumPy. Given a finite matrix `X` of shape `(n, d)` and a requested number of clusters `k`, initialize centers with k-means++, alternate assignment and update steps, and stop when the centers move sufficiently little. Accept a seed so the run can be reproduced.

For example, `[[0, 0], [0, 2], [10, 0], [10, 2]]` with two clusters can converge to centers `[[0, 1], [10, 1]]`. Each point is distance `1` from its center, so the sum of squared distances is `4`.

This implementation returns `(centers, labels, inertia)` and does not change `X`. It requires `1 <= k <= n`, but permits repeated points and fewer than `k` distinct values. Empty clusters keep their previous center. Equal distances choose the lowest center index; cluster numbers have no inherent meaning.

## Math in brief

```text
Objective: sum over i of ||X[i] - centers[labels[i]]||^2
Assignment: labels[i] = argmin_j ||X[i] - centers[j]||^2
Update: centers[j] = mean of the points assigned to j
```

For k-means++ initialization, choose the first center uniformly from the data. For each later center, let `D[i]^2` be the squared distance from point `i` to its nearest chosen center. Sample the next point with probability `D[i]^2 / sum(D^2)`.

## Key idea

With fixed centers, nearest-center assignment minimizes the objective over labels. With fixed labels, each nonempty cluster's mean minimizes its squared distances. Alternating these two steps cannot increase the objective in exact arithmetic, although it can converge to a local optimum.

## Step by step

1. Validate the data, cluster count, iteration limit, and nonnegative tolerance.
2. Create a local NumPy random generator from the seed and choose the first center uniformly.
3. Track nearest squared distances and sample each later center with k-means++ probabilities. If every distance is zero, choose a row uniformly because no uncovered distinct point remains.
4. Compute the point-to-center squared distances and assign each point to its closest center.
5. Replace every nonempty cluster's center with its mean. Preserve empty clusters' centers.
6. Stop if the largest Euclidean center movement is at most `tol`, or after `max_iter` updates.
7. Recompute labels and inertia using the returned centers, so both outputs describe the final centers even at the iteration cap.

For the example's left pair, the update averages `[0, 0]` and `[0, 2]` into `[0, 1]`. The right pair similarly averages to `[10, 1]`. Once those assignments remain fixed, the next update moves neither center.

## Python solution

<<< ./solutions/k_means.py

## Complexity

Initialization takes **O(nkd)** time. With `T` assignment/update rounds, the full fit takes **O((T + 1)nkd)** time. Working storage is **O(nk + nd + kd)**, including the distance matrix, temporary point arrays, and centers. Computing distances one center at a time avoids an **O(nkd)** temporary array.

## Common mistakes

- Uniformly choosing every initial center implements random initialization, not k-means++. Sampling must use squared distances to the nearest chosen center.
- Taking the mean of an empty cluster creates NaNs. Keeping its old center is this implementation's explicit policy; reseeding is another policy to discuss.
- Duplicate points can make all initialization distances zero. Dividing by their sum then produces invalid probabilities.
- Normalizing very large distance sums can overflow. The implementation first divides distances by their maximum; individual squared distances must still fit in float64.
- Raw features on different scales can make one feature dominate Euclidean distance. Choose meaningful scaling before fitting.
- Returning old labels after updating centers can make labels and inertia disagree with the returned centers.
- A seed gives reproducibility in the same NumPy environment, not a guarantee of a globally optimal clustering. At the iteration cap, centers need not be means of the recomputed final labels.

## Interview follow-ups

- How would multiple initializations and selection by inertia improve robustness?
- What are the tradeoffs between retaining and reseeding empty clusters?
- How would you choose `k`, and when is the spherical-cluster assumption inappropriate?
- How would mini-batch updates reduce the cost for very large datasets?
