"""Check sorting, closed endpoints, nesting, and independent output lists."""

from copy import deepcopy

import pytest


@pytest.fixture
def solution(load_solution):
    return load_solution("p0056_merge_intervals.py").Solution()


@pytest.mark.parametrize(("intervals", "expected"), [
    ([], []),
    ([[4, 9]], [[4, 9]]),
    ([[6, 6]], [[6, 6]]),
    ([[7, 10], [2, 5], [4, 8], [13, 13]], [[2, 10], [13, 13]]),
    ([[6, 9], [2, 6]], [[2, 9]]),
    ([[1, 12], [3, 5], [7, 8]], [[1, 12]]),
    ([[4, 7], [4, 7], [4, 5]], [[4, 7]]),
    ([[9, 11], [5, 6], [1, 2]], [[1, 2], [5, 6], [9, 11]]),
    ([[-8, -3], [-4, 2], [5, 8]], [[-8, 2], [5, 8]]),
    ([[2, 2], [2, 5], [5, 5]], [[2, 5]]),
])
def test_merged_ranges(solution, intervals, expected):
    original = deepcopy(intervals)
    assert solution.merge(intervals) == expected
    assert intervals == original


def test_output_does_not_alias_input(solution):
    intervals = [[7, 9], [2, 4]]
    result = solution.merge(intervals)
    result[0][1] = 100
    assert intervals == [[7, 9], [2, 4]]
    intervals[0][0] = 0
    assert result[1] == [7, 9]
