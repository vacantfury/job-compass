---
title: 1. Two Sum
description: Find a pair of indices in one pass by remembering previously seen values.
---
# 1. Two Sum

## Problem in plain words

Given a list of integers and a target, return the positions of two different elements whose values add up to that target. Equal values at different positions are allowed. The usual problem guarantees exactly one pair; the order of the two returned positions does not matter.

For example, `[8, 1, 6, 4]` with target `10` gives positions `[2, 3]`, since `6 + 4 = 10`.

This implementation also defines behavior outside that guarantee: it raises `ValueError` when no pair exists, including for empty or single-element input. If several pairs exist, it returns the first one found by the scan. It does not change the input.

## Key idea

Use a hash map from a value to an earlier position. For the current value, the only useful partner is `target - value`. A lookup tells us whether that partner has already appeared.

## Step by step

1. Start with an empty map called `seen`.
2. At each position, compute the complement of the current value.
3. If the complement is in `seen`, return its earlier position and the current position.
4. Otherwise, store the current value and position, then continue.
5. If the scan finishes, report that no pair exists.

In the example, the map accumulates `8`, `1`, and `6`. At position `3`, the value is `4`, whose complement `6` is already stored at position `2`.

The map contains only earlier positions when we query it, so a match always uses two different elements. Any valid pair will be found when its later element is reached, because its earlier partner is already in the map.

## Python solution

<<< ./solutions/p0001_two_sum.py

## Complexity

Let `n` be the number of elements. Time is **O(n)** on average, assuming average constant-time dictionary operations. Auxiliary space is **O(n)** for the map. The returned pair uses constant space.

## Common mistakes

- Inserting before checking can match an element with itself when the target is twice its value.
- Returning the values instead of their positions answers a different question.
- A set can detect the complement but does not remember the position needed for the answer.
- Sorting without retaining original positions loses the required indices.

## Interview follow-ups

- How would you return all distinct pairs without reporting the same pair twice?
- If the input were already sorted, how would two pointers change the space usage?
- For a stream of values, what must you retain to find a pair when it first becomes available?
