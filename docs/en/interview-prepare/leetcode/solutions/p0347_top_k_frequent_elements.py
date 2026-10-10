"""Select the most frequent distinct values with frequency buckets."""

from collections import Counter


class Solution:
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        counts = Counter(nums)
        if not 0 <= k <= len(counts):
            raise ValueError("k must be between zero and the distinct value count")
        if k == 0:
            return []

        buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]
        for value, frequency in counts.items():
            buckets[frequency].append(value)

        result: list[int] = []
        for bucket in reversed(buckets):
            for value in bucket:
                result.append(value)
                if len(result) == k:
                    return result
        return result
