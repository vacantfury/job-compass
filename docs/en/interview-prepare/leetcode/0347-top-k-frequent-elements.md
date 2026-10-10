---
title: 347. Top K Frequent Elements
description: Group distinct values by frequency and read the most populated frequencies first.
---
# 347. Top K Frequent Elements

## Problem in plain words

Given integers and a number `k`, return `k` distinct values with the highest occurrence counts. Repeated occurrences affect ranking, but each selected value appears only once in the result. Output order does not matter.

For example, `[6, 9, 6, 4, 9, 6, 9, 6]` with `k = 2` returns `6` and `9`. Their frequencies are four and three, compared with one for `4`.

The usual problem guarantees a unique selected set. For broader practice, this implementation also allows ties at the cutoff: any tied value is valid, and this code selects them in first-occurrence order. It accepts `k = 0`, returning `[]` even for empty input. Negative `k` or `k` larger than the number of distinct values raises `ValueError`. The input is unchanged.

## Key idea

Use frequency buckets. In an input of length `n`, a frequency must lie between `1` and `n`. Count each value, then place it into the bucket indexed by its count. Reading buckets from high frequency to low avoids sorting all distinct values.

## Step by step

1. Use `Counter` to compute each distinct value's frequency.
2. Check that `k` is valid, and return immediately if it is zero.
3. Create `n + 1` separate empty buckets; index zero stays unused.
4. Put each distinct value in the bucket matching its frequency.
5. Visit buckets in descending order. Append each value once.
6. Return as soon as the result contains exactly `k` values, including when a bucket contains more tied candidates than needed.

In the example, bucket `4` contains `6`, bucket `3` contains `9`, and bucket `1` contains `4`. The descending scan selects `6`, then `9`, and stops.

Each distinct value appears in one bucket. Before we move to a lower bucket, all values with higher frequency have been considered. Thus no omitted value has a higher count than a selected one. Equal frequencies are interchangeable at the cutoff.

## Python solution

<<< ./solutions/p0347_top_k_frequent_elements.py

## Complexity

Let `n` be the input length and `u` the number of distinct values. Time is **O(n)** on average: counting is O(n), bucket creation is O(n), and placement plus traversal is O(n + u), with `u <= n`. Auxiliary space is **O(n)** for the counts and buckets. The output uses **O(k)** space.

## Common mistakes

- Sorting by numeric value instead of by occurrence count answers a different question.
- Appending a whole tied bucket can return more than `k` values.
- Using `[[]] * (n + 1)` aliases the same inner list across all buckets.
- Creating only `n` buckets fails when one value occurs `n` times.
- Tests should compare selected sets or frequency validity, not require an arbitrary output order.

## Interview follow-ups

- How would a size-`k` min-heap reduce bucket storage? With precomputed counts, it uses O(u + k) auxiliary space and O(n + u log(k + 1)) time.
- Could quickselect on the distinct values give expected linear time without allocating `n + 1` buckets?
- How would you maintain frequent values as a stream changes their counts?
