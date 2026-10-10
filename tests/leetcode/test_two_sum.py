"""Check distinct indices, complements, and inputs outside the pair guarantee."""

import pytest


@pytest.fixture
def solution(load_solution):
    return load_solution("p0001_two_sum.py").Solution()


@pytest.mark.parametrize(("nums", "target", "expected"), [
    ([8, 1, 6, 4], 10, [2, 3]),
    ([7, 7], 14, [0, 1]),
    ([0, 8, 0], 0, [0, 2]),
    ([-8, -2, -5], -7, [1, 2]),
    ([-6, 4, 9], 3, [0, 2]),
    ([5, 1, 9], 10, [1, 2]),
    ([10**20, 1, -(10**20)], 0, [0, 2]),
])
def test_pair_and_input_preservation(solution, nums, target, expected):
    original = nums.copy()
    result = solution.twoSum(nums, target)
    assert result == expected
    assert result[0] != result[1]
    assert nums[result[0]] + nums[result[1]] == target
    assert nums == original


@pytest.mark.parametrize(("nums", "target"), [
    ([], 5),
    ([5], 10),
    ([0], 0),
    ([2, 4, 8], 13),
    ([6, 6], 15),
])
def test_no_pair_raises(solution, nums, target):
    with pytest.raises(ValueError, match="No pair"):
        solution.twoSum(nums, target)


def test_multiple_valid_pairs(solution):
    nums = [1, 8, 2, 7, 3, 6]
    left, right = solution.twoSum(nums, 9)
    assert 0 <= left < right < len(nums)
    assert nums[left] + nums[right] == 9
