"""Exercise eviction order through public operations and a reference cache."""

from collections import OrderedDict
from random import Random

import pytest


@pytest.fixture
def cache_type(load_solution):
    return load_solution("p0146_lru_cache.py").LRUCache


def test_empty_cache(cache_type):
    cache = cache_type(3)
    assert cache.get(4) == -1
    assert cache.get(9) == -1


def test_get_refreshes_eviction_order(cache_type):
    cache = cache_type(2)
    cache.put(4, 40)
    cache.put(9, 90)
    assert cache.get(4) == 40
    cache.put(6, 60)
    assert cache.get(9) == -1
    cache.put(8, 80)
    assert cache.get(4) == -1
    assert cache.get(6) == 60
    assert cache.get(8) == 80


def test_update_refreshes_recency_without_adding_an_entry(cache_type):
    cache = cache_type(2)
    cache.put(4, 40)
    cache.put(9, 90)
    cache.put(4, 44)
    cache.put(6, 60)
    assert cache.get(9) == -1
    assert cache.get(4) == 44
    assert cache.get(6) == 60


def test_miss_does_not_change_order(cache_type):
    cache = cache_type(2)
    cache.put(4, 40)
    cache.put(9, 90)
    assert cache.get(100) == -1
    cache.put(6, 60)
    assert cache.get(4) == -1
    assert cache.get(9) == 90
    assert cache.get(6) == 60


def test_capacity_one_and_repeated_updates(cache_type):
    cache = cache_type(1)
    cache.put(4, 40)
    cache.put(4, 44)
    assert cache.get(4) == 44
    assert cache.get(4) == 44
    cache.put(9, 90)
    assert cache.get(4) == -1
    assert cache.get(9) == 90
    cache.put(4, 48)
    assert cache.get(9) == -1
    assert cache.get(4) == 48


def test_zero_key_and_value(cache_type):
    cache = cache_type(2)
    cache.put(0, 0)
    cache.put(4, 40)
    assert cache.get(0) == 0
    cache.put(6, 60)
    assert cache.get(4) == -1
    assert cache.get(0) == 0


def test_zero_capacity(cache_type):
    cache = cache_type(0)
    for key in [0, 4, 4, 9]:
        cache.put(key, 10)
        assert cache.get(key) == -1


def test_negative_capacity(cache_type):
    with pytest.raises(ValueError, match="nonnegative"):
        cache_type(-1)


def test_instances_are_independent(cache_type):
    first = cache_type(1)
    second = cache_type(1)
    first.put(4, 40)
    second.put(4, 90)
    assert first.get(4) == 40
    assert second.get(4) == 90


@pytest.mark.parametrize("capacity", [0, 1, 2, 5])
def test_operation_sequences_match_ordered_dict(cache_type, capacity):
    rng = Random(146)
    cache = cache_type(capacity)
    reference = OrderedDict()
    for _ in range(400):
        key = rng.randrange(8)
        if rng.randrange(2):
            value = rng.randrange(100)
            cache.put(key, value)
            reference[key] = value
            reference.move_to_end(key)
            if len(reference) > capacity:
                reference.popitem(last=False)
        else:
            expected = reference.get(key, -1)
            if key in reference:
                reference.move_to_end(key)
            assert cache.get(key) == expected

    for key in range(8):
        assert cache.get(key) == reference.get(key, -1)
