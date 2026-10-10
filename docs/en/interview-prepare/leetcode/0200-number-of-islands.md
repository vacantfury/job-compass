---
title: 200. Number of Islands
description: Count connected land regions with an iterative breadth-first traversal.
---
# 200. Number of Islands

## Problem in plain words

A rectangular grid contains land (`"1"`) and water (`"0"`). Count separate regions of land. Two land cells are connected through their shared top, bottom, left, or right edge; diagonal contact does not connect them. Cells outside the grid are not land.

For example, this grid has three islands:

```text
1 0 1
1 0 0
0 1 1
```

The left column's two land cells form one island, the upper-right cell forms another, and the bottom row's two land cells form the third. Empty input and grids with no columns return `0`. The implementation preserves the input grid.

## Key idea

Scan every cell. When an unvisited land cell appears, count one new island and use breadth-first search (BFS) to mark every cell reachable from it. Those cells cannot start another island later.

## Step by step

1. Handle an empty grid, then create a visited set and a zero island count.
2. Scan rows and columns. Skip water and previously visited land.
3. For new land, increment the count, mark the cell visited, and put it in a queue.
4. Remove the next queued cell with `deque.popleft()`.
5. Examine its four neighbors. For each neighbor inside the grid that is unvisited land, mark it visited immediately and enqueue it.
6. Continue until the queue is empty, then resume the outer scan.

Marking on enqueue ensures each land cell enters the queue only once, even when several neighbors lead to it. A traversal reaches exactly one connected component. The first scanned cell in each component starts a traversal, so every island is counted exactly once.

An iterative queue avoids Python's recursion-depth limit on a long connected island.

## Python solution

<<< ./solutions/p0200_number_of_islands.py

## Complexity

For `r` rows and `c` columns, time is **O(r * c)** on average with set operations. Each cell is scanned once, and each land cell's four neighbors are inspected once. Auxiliary space is **O(r * c)** in the worst case for the visited set and queue.

## Common mistakes

- Counting diagonal neighbors joins islands that should stay separate.
- Delaying the visited mark until dequeue can enqueue the same cell repeatedly.
- Using `list.pop(0)` shifts elements and can make queue operations expensive.
- Recursive DFS can exceed the call stack on a large connected component.
- Replacing land with water saves the visited set but changes the caller's input. That tradeoff needs agreement.

## Interview follow-ups

- How would an explicit DFS stack change the traversal?
- If modifying the grid is allowed, how could you store visited state in the cells? What space would the traversal still need?
- How would you track islands as new land cells are added over time?
