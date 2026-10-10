"""Measure the longest distinct-character substring with a sliding window."""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last_seen: dict[str, int] = {}
        left = 0
        longest = 0
        for right, character in enumerate(s):
            left = max(left, last_seen.get(character, -1) + 1)
            last_seen[character] = right
            longest = max(longest, right - left + 1)
        return longest
