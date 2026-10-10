---
title: 146. LRU Cache
description: Combine direct key lookup with a doubly linked list to track recent use.
---
# 146. LRU Cache

## Problem in plain words

Build a cache holding at most `capacity` integer key-value entries. `get(key)` returns the stored value or `-1` for a miss. `put(key, value)` inserts a new entry or replaces an existing value. A successful read or any write makes that entry the most recently used. When a new entry exceeds capacity, discard the least recently used entry. Both operations should take average constant time.

For a capacity of `2`, write `(4, 40)` and `(9, 90)`, read key `4`, then write `(6, 60)`. Key `9` is evicted because key `4` was used more recently.

The usual problem uses positive capacity. This implementation also accepts zero capacity, which stores nothing, and rejects negative capacity with `ValueError`.

## Key idea

A hash map finds each key's node directly. A doubly linked list keeps nodes in order of use. With a direct node reference, removing or appending that node requires only a fixed number of pointer updates.

One circular sentinel node, `_root`, marks both ends. Its `next` link leads to the least recently used entry; its `prev` link leads to the most recently used entry. In an empty cache both links point back to the sentinel. The sentinel is never stored in the map.

## Step by step

1. Initialize the empty map and circular sentinel.
2. For `get`, look up the node. On a miss, return `-1` without changing the order.
3. On a hit, detach the node and append it just before the sentinel, then return its value.
4. For `put` on an existing key, replace its value, detach it, and append it at the recent end.
5. For `put` on a new key, create a node, add it to the map, and append it at the recent end.
6. If capacity is exceeded, detach the sentinel's next node and delete its key from the map.

In the example, the order from oldest to newest is `[4, 9]` after the two writes. Reading `4` changes it to `[9, 4]`. Adding `6` temporarily gives `[9, 4, 6]`, so evicting the front leaves `[4, 6]`.

Every cached key has exactly one node in both structures. Successful accesses move that node to the recent end, and all other entries retain their relative order. Therefore the front is always the correct eviction candidate.

## Python solution

<<< ./solutions/p0146_lru_cache.py

## Complexity

Each `get` and `put` takes **O(1)** average time, assuming average constant-time dictionary operations. Each list operation updates a fixed number of links. Space is **O(c)** for positive capacity `c`, plus the constant-size sentinel. A zero-capacity cache uses constant space.

## Common mistakes

- Updating a value without refreshing its recency evicts the wrong key later.
- Treating a failed read as an access changes eviction order incorrectly.
- Removing a node from only the list or only the map makes the structures disagree.
- Creating another node for an existing key leaves duplicates in the order.
- Searching a Python list for a key makes each access linear in cache size.

## Interview follow-ups

- How would Python's `OrderedDict` simplify this implementation?
- What synchronization is required if several threads access the cache?
- How would expiration times or a memory-size limit change the eviction policy?
