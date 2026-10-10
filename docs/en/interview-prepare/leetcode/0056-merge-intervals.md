---
title: 56. Merge Intervals
description: Sort closed intervals by their starts and combine overlapping ranges.
---
# 56. Merge Intervals

## Problem in plain words

Given closed intervals `[start, end]`, combine intervals that overlap and return disjoint intervals covering the same points. Endpoints are included, so intervals sharing an endpoint must merge. Assume each interval has two endpoints with `start <= end`.

For example, `[[7, 10], [2, 5], [4, 8], [13, 13]]` becomes `[[2, 10], [13, 13]]`.

This implementation returns `[]` for empty input. It returns intervals in ascending order and creates fresh output lists, leaving the input order and endpoints unchanged.

## Key idea

Sort by start position. After sorting, the next interval can only extend the last output interval or begin a new one. Earlier completed output intervals cannot overlap it.

## Step by step

1. Create a sorted copy of the input and start an empty result.
2. If the result is empty, append a fresh interval.
3. If the next start is greater than the last output end, append a new interval.
4. Otherwise, extend the last output end to the larger of the two ends.
5. Return the result after all intervals have been processed.

The example sorts to `[2, 5]`, `[4, 8]`, `[7, 10]`, `[13, 13]`. The first three ranges overlap in a chain, producing `[2, 10]`. The final point interval starts after `10`, so it remains separate.

At each step, the result is the merged union of the intervals already scanned. Sorted starts guarantee that a gap before the next interval can never be filled by a later interval. Taking the maximum end preserves the whole union when intervals overlap, including nested intervals.

## Python solution

<<< ./solutions/p0056_merge_intervals.py

## Complexity

For `n` intervals, sorting takes **O(n log n)** time and the scan takes O(n). Auxiliary space is **O(n)** for the sorted copy and Python's sorting workspace. The returned intervals also take up to **O(n)** space.

## Common mistakes

- Scanning unsorted input can miss overlaps between intervals far apart in the list.
- Using `start >= previous_end` to detect a gap incorrectly separates touching closed intervals.
- Replacing the previous end with the current end can shrink a containing interval.
- Appending an original inner list and modifying it later can unexpectedly mutate the caller's input.

## Interview follow-ups

- If the intervals arrive already sorted, what work can you eliminate?
- How does the overlap rule change for half-open intervals `[start, end)`?
- How would you insert one interval into an already merged list?
