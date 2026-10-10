"""Check contiguous windows, repeated characters, and stale last-seen positions."""

from itertools import product

import pytest


@pytest.fixture
def solution(load_solution):
    return load_solution(
        "p0003_longest_substring_without_repeating_characters.py"
    ).Solution()


@pytest.mark.parametrize(("text", "expected"), [
    ("", 0),
    ("z", 1),
    ("kkkkk", 1),
    ("cadcae", 4),
    ("abbaa", 2),
    ("aab", 2),
    ("abcdef", 6),
    (" a! a", 3),
    ("aA", 2),
    ("\u00e9a\u00e9b", 3),
    ("\U0001f642a\U0001f642bc", 4),
])
def test_window_lengths(solution, text, expected):
    assert solution.lengthOfLongestSubstring(text) == expected


def test_small_strings_match_brute_force(solution):
    for length in range(6):
        for characters in product("abc", repeat=length):
            text = "".join(characters)
            expected = max(
                (
                    end - start
                    for start in range(length)
                    for end in range(start + 1, length + 1)
                    if len(set(text[start:end])) == end - start
                ),
                default=0,
            )
            assert solution.lengthOfLongestSubstring(text) == expected, text
