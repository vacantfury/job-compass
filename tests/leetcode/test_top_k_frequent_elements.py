"""Check selected frequencies, boundary k values, and ties without ordering bias."""

from collections import Counter
from random import Random

import pytest


@pytest.fixture
def solution(load_solution):
    return load_solution("p0347_top_k_frequent_elements.py").Solution()


@pytest.mark.parametrize(("nums", "k", "expected"), [
    ([6, 9, 6, 4, 9, 6, 9, 6], 2, {6, 9}),
    ([8], 1, {8}),
    ([5, 5, 5, 5], 1, {5}),
    ([-7, -7, -7, 0, 0, 3], 2, {-7, 0}),
    ([8, 2, 5, 8], 3, {8, 2, 5}),
    ([1, 1, 2, 2, 3], 2, {1, 2}),
    ([], 0, set()),
    ([4, 4, 9], 0, set()),
])
def test_selected_values(solution, nums, k, expected):
    original = nums.copy()
    result = solution.topKFrequent(nums, k)
    assert len(result) == k
    assert set(result) == expected
    assert nums == original


@pytest.mark.parametrize(("nums", "k"), [
    ([], 1),
    ([], -1),
    ([4, 4], 2),
    ([4, 9], -1),
])
def test_invalid_k(solution, nums, k):
    with pytest.raises(ValueError, match="distinct value count"):
        solution.topKFrequent(nums, k)


@pytest.mark.parametrize(("nums", "k"), [
    ([8, 3, 5, 7], 2),
    ([8, 8, 3, 3, 5, 5, 7], 2),
    ([8, 8, 8, 3, 3, 5, 5, 7], 2),
])
def test_ties_at_cutoff(solution, nums, k):
    counts = Counter(nums)
    result = solution.topKFrequent(nums, k)
    assert len(result) == len(set(result)) == k
    assert set(result) <= set(counts)
    assert sorted((counts[value] for value in result), reverse=True) == sorted(
        counts.values(), reverse=True
    )[:k]


def test_generated_inputs_match_sorted_frequency_ranks(solution):
    rng = Random(347)
    for _ in range(100):
        nums = [rng.randrange(-5, 6) for _ in range(rng.randrange(30))]
        counts = Counter(nums)
        k = rng.randrange(len(counts) + 1)
        result = solution.topKFrequent(nums, k)
        assert len(result) == len(set(result)) == k
        assert set(result) <= set(counts)
        assert sorted((counts[value] for value in result), reverse=True) == sorted(
            counts.values(), reverse=True
        )[:k]
