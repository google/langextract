# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for fast fuzzy alignment helpers."""

from absl.testing import absltest

from langextract.core import fuzzy_alignment


class FuzzyAlignmentTest(absltest.TestCase):

  def test_greedy_subsequence_match_skips_missing_tokens(self):
    source = ["a", "b", "c", "d"]
    positions = fuzzy_alignment.build_token_positions(source)

    matched = fuzzy_alignment.greedy_subsequence_match_positions(
        ["x", "b", "y", "d"], positions
    )
    self.assertEqual(matched, [1, 3])

  def test_generate_candidate_starts_respects_max_candidates(self):
    source = ["x", "a", "b", "c", "d"]
    positions = fuzzy_alignment.build_token_positions(source)

    candidates = fuzzy_alignment.generate_candidate_starts(
        ["a", "b"],
        positions,
        max_candidates=1,
        max_anchor_occurrences=10,
    )
    self.assertEqual(candidates, [0])

  def test_find_best_fuzzy_span_prefers_tighter_span_on_tie(self):
    # Two perfect matches exist: [0..2] spans 3 tokens, [3..4] spans 2 tokens.
    source = ["a", "x", "b", "a", "b"]
    best = fuzzy_alignment.find_best_fuzzy_span(
        source,
        ["a", "b"],
        max_candidates=10,
        max_anchor_occurrences=10,
    )
    self.assertEqual(best, (3, 5, 2))


if __name__ == "__main__":
  absltest.main()
