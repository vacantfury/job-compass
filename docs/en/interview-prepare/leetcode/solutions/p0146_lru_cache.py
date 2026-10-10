"""Implement an LRU cache with a hash map and a doubly linked list."""

from __future__ import annotations


class _Node:
    def __init__(self, key: int = 0, value: int = 0) -> None:
        self.cache_key = key
        self.value = value
        self.prev: _Node = self
        self.next: _Node = self


class LRUCache:
    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("Capacity must be nonnegative")
        self._capacity = capacity
        self._nodes: dict[int, _Node] = {}
        self._root = _Node()

    def _remove(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _append(self, node: _Node) -> None:
        # Append just before the sentinel, at the most recently used end.
        node.prev = self._root.prev
        node.next = self._root
        node.prev.next = node
        self._root.prev = node

    def get(self, key: int) -> int:
        node = self._nodes.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._append(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if self._capacity == 0:
            return
        node = self._nodes.get(key)
        if node is not None:
            node.value = value
            self._remove(node)
        else:
            node = _Node(key, value)
            self._nodes[key] = node
        self._append(node)

        if len(self._nodes) > self._capacity:
            oldest = self._root.next
            self._remove(oldest)
            del self._nodes[oldest.cache_key]
