---
title: 3. Longest Substring Without Repeating Characters
description: Maintain a window of distinct characters using their most recent positions.
---
# 3. Longest Substring Without Repeating Characters

## Problem in plain words

Find the length of the longest contiguous part of a string in which every character is different. Characters cannot be skipped. An empty string has answer `0`.

For example, `"cadcae"` has answer `4`, from `"dcae"`. This implementation compares Python string characters, which are Unicode code points. Spaces, punctuation, and uppercase letters are ordinary, distinct characters; it performs no normalization.

## Key idea

Maintain a sliding window with no repeated characters. A map records the most recent position of each character. When the right end reaches a duplicate inside the window, jump the left end just past its earlier occurrence.

## Step by step

1. Set the left boundary and best length to zero; start with an empty position map.
2. Visit each character at the right boundary.
3. Move the left boundary to the larger of its current position and one past this character's previous position. For an unseen character, use previous position `-1`.
4. Record the character's new position.
5. Measure `right - left + 1` and update the best length.

For `"cadcae"`, the boundaries evolve as follows:

| Right position | Character | Left position | Current window | Best length |
| --- | --- | --- | --- | --- |
| 0 | c | 0 | c | 1 |
| 1 | a | 0 | ca | 2 |
| 2 | d | 0 | cad | 3 |
| 3 | c | 1 | adc | 3 |
| 4 | a | 2 | dca | 3 |
| 5 | e | 2 | dcae | 4 |

Before adding the next character, the window has no repeats. Only that new character can violate the rule, and moving past its last occurrence fixes the violation. The left boundary stays as far left as validity allows, so each step finds the longest valid window ending at that right position.

## Python solution

<<< ./solutions/p0003_longest_substring_without_repeating_characters.py

## Complexity

For a string of length `n`, time is **O(n)** on average with dictionary operations. Auxiliary space is **O(min(n, A))**, where `A` is the number of possible distinct characters. For unrestricted input, this is O(n) in the worst case.

## Common mistakes

- Moving `left` backward when an earlier occurrence is already outside the window. The `max` prevents this.
- Confusing a substring with a subsequence, which can skip characters.
- Using `right - left` instead of including both endpoints.
- Clearing the whole window on every duplicate can discard a useful suffix.

## Interview follow-ups

- How would you return the substring itself while avoiding a copy at every step?
- How would you adapt the window to allow at most `k` distinct characters?
- What changes if a user-visible character consists of several Unicode code points?
