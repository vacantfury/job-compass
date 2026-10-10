"""Merge overlapping closed intervals without modifying the input."""


class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return merged
