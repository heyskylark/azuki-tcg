from __future__ import annotations

from collections import Counter

from draft_vs_reference_eval import multiset_jaccard


def test_multiset_jaccard_counts_copies():
  left = Counter({"a": 4, "b": 2, "c": 1})
  right = Counter({"a": 2, "b": 2, "d": 3})

  assert multiset_jaccard(left, right) == 4 / 10


def test_multiset_jaccard_empty_decks():
  assert multiset_jaccard(Counter(), Counter()) == 0.0
