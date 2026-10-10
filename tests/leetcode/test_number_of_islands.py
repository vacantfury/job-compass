"""Check four-directional connectivity and preservation of the input grid."""

from copy import deepcopy

import pytest


@pytest.fixture
def solution(load_solution):
    return load_solution("p0200_number_of_islands.py").Solution()


@pytest.mark.parametrize(("rows", "expected"), [
    ([], 0),
    ([""], 0),
    (["", ""], 0),
    (["0"], 0),
    (["1"], 1),
    (["0000", "0000"], 0),
    (["1111", "1111"], 1),
    (["101", "100", "011"], 3),
    (["10", "01"], 2),
    (["10101", "01010", "10101"], 8),
    (["1101101"], 3),
    (["1", "1", "0", "1", "0", "1"], 3),
    (["111", "101", "111"], 1),
])
def test_components_and_input_preservation(solution, rows, expected):
    grid = [list(row) for row in rows]
    original = deepcopy(grid)
    assert solution.numIslands(grid) == expected
    assert grid == original
    assert solution.numIslands(grid) == expected


def test_long_connected_island_does_not_need_recursion(solution):
    grid = [["1"] for _ in range(2500)]
    assert solution.numIslands(grid) == 1
    assert all(row == ["1"] for row in grid)
