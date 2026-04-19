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

"""Correctness oracle tests for fuzzy alignment.

These planted-span cases serve as regression tests before and after
performance changes to _fuzzy_align_extraction. Each case asserts
exact token_interval, char_interval, and matched substring.
"""

import random
import warnings

from absl.testing import absltest
from absl.testing import parameterized

from langextract import prompt_validation
from langextract import resolver as resolver_lib
from langextract.core import data
from langextract.core import tokenizer as tokenizer_lib

_WORD_POOL = [
    "patient",
    "diagnosed",
    "with",
    "diabetes",
    "hypertension",
    "medication",
    "prescribed",
    "daily",
    "chronic",
    "condition",
    "treatment",
    "history",
    "symptoms",
    "blood",
    "pressure",
    "glucose",
    "insulin",
    "kidney",
    "liver",
    "cardiac",
    "pulmonary",
    "neurological",
    "assessment",
    "examination",
    "laboratory",
    "results",
    "normal",
    "elevated",
    "decreased",
    "follow",
    "appointment",
    "scheduled",
    "monitor",
    "progress",
    "clinical",
    "evaluation",
    "imaging",
    "therapy",
    "dosage",
    "adverse",
    "reaction",
    "prognosis",
    "referral",
    "discharge",
    "admission",
    "surgery",
    "recovery",
    "emergency",
    "outpatient",
    "inpatient",
    "consultation",
    "diagnosis",
    "pathology",
    "specimen",
    "biopsy",
    "cultures",
    "antibiotics",
    "analgesic",
    "sedation",
    "ventilation",
    "intubation",
    "catheter",
    "drainage",
    "infusion",
]


def _generate_source(n, seed=42):
  """Generates deterministic source text from _WORD_POOL."""
  rng = random.Random(seed)
  return " ".join(rng.choice(_WORD_POOL) for _ in range(n))


def _plant_span(source, target, pos):
  """Inserts target tokens at pos in source."""
  words = source.split()
  target_words = target.split()
  p = min(pos, len(words))
  words[p : p + len(target_words)] = target_words
  return " ".join(words)


def _plant_gapped(source, tokens, start, gap):
  """Inserts tokens at intervals of (gap+1) starting at start."""
  words = source.split()
  for i, token in enumerate(tokens):
    p = min(start + i * (gap + 1), len(words) - 1)
    words[p] = token
  return " ".join(words)


def _run(
    source,
    extraction_text,
    tokenizer,
    aligner,
    algorithm="lcs",
    token_offset=0,
    char_offset=0,
):
  """Runs fuzzy alignment using the requested algorithm and returns the result."""
  tokenized = tokenizer.tokenize(source)
  source_tokens = [
      source[t.char_interval.start_pos : t.char_interval.end_pos].lower()
      for t in tokenized.tokens
  ]
  extraction = data.Extraction(
      extraction_class="entity", extraction_text=extraction_text
  )
  if algorithm == "lcs":
    source_tokens_norm = [
        resolver_lib._normalize_token(t) for t in source_tokens
    ]
    return aligner._lcs_fuzzy_align_extraction(
        extraction=extraction,
        source_tokens_norm=source_tokens_norm,
        tokenized_text=tokenized,
        token_offset=token_offset,
        char_offset=char_offset,
        tokenizer_impl=tokenizer,
    )
  return aligner._fuzzy_align_extraction(
      extraction=extraction,
      source_tokens=source_tokens,
      tokenized_text=tokenized,
      token_offset=token_offset,
      char_offset=char_offset,
      tokenizer_impl=tokenizer,
  )


_BASE_200 = _generate_source(200, seed=42)
_PLANTED = _plant_span(_BASE_200, "metformin hydrochloride tablet", 50)
_GAPPED = _plant_gapped(
    _generate_source(200, seed=99),
    ["metformin", "hydrochloride", "tablet"],
    start=40,
    gap=3,
)


_PLANTED_POSITIVE_CASES = (
    dict(
        testcase_name="contiguous_lcs",
        algorithm="lcs",
        source=_PLANTED,
        extraction_text="metformin hydrochloride tablet",
        expect_token_interval=(50, 53),
        expect_char_interval=(451, 481),
        expect_substring="metformin hydrochloride tablet",
    ),
    dict(
        testcase_name="contiguous_legacy",
        algorithm="legacy",
        source=_PLANTED,
        extraction_text="metformin hydrochloride tablet",
        expect_token_interval=(50, 53),
        expect_char_interval=(451, 481),
        expect_substring="metformin hydrochloride tablet",
    ),
    dict(
        testcase_name="fuzzy_stemming_lcs",
        algorithm="lcs",
        source=_PLANTED,
        extraction_text="metformins hydrochlorides tablets",
        expect_token_interval=(50, 53),
        expect_char_interval=(451, 481),
        expect_substring="metformin hydrochloride tablet",
    ),
    dict(
        testcase_name="fuzzy_stemming_legacy",
        algorithm="legacy",
        source=_PLANTED,
        extraction_text="metformins hydrochlorides tablets",
        expect_token_interval=(50, 53),
        expect_char_interval=(451, 481),
        expect_substring="metformin hydrochloride tablet",
    ),
    dict(
        testcase_name="gapped_lcs",
        algorithm="lcs",
        source=_GAPPED,
        extraction_text="metformin hydrochloride tablet",
        expect_token_interval=(40, 49),
        expect_char_interval=(371, 461),
        expect_substring=(
            "metformin pulmonary antibiotics assessment"
            " hydrochloride hypertension pressure with tablet"
        ),
    ),
    dict(
        testcase_name="gapped_legacy",
        algorithm="legacy",
        source=_GAPPED,
        extraction_text="metformin hydrochloride tablet",
        expect_token_interval=(40, 49),
        expect_char_interval=(371, 461),
        expect_substring=(
            "metformin pulmonary antibiotics assessment"
            " hydrochloride hypertension pressure with tablet"
        ),
    ),
)


class FuzzyAlignmentCasesTest(parameterized.TestCase):
  """Planted-span oracle tests for the fuzzy aligners."""

  def setUp(self):
    super().setUp()
    self._tokenizer = tokenizer_lib.RegexTokenizer()
    self._aligner = resolver_lib.WordAligner()
    resolver_lib._normalize_token.cache_clear()

  @parameterized.named_parameters(*_PLANTED_POSITIVE_CASES)
  def test_planted_positive(
      self,
      algorithm,
      source,
      extraction_text,
      expect_token_interval,
      expect_char_interval,
      expect_substring,
  ):
    """Both algorithms agree on the planted oracle spans."""
    result = _run(
        source,
        extraction_text,
        self._tokenizer,
        self._aligner,
        algorithm=algorithm,
    )

    self.assertIsNotNone(result)
    self.assertEqual(result.alignment_status, data.AlignmentStatus.MATCH_FUZZY)
    self.assertEqual(
        (
            result.token_interval.start_index,
            result.token_interval.end_index,
        ),
        expect_token_interval,
    )
    self.assertEqual(
        (result.char_interval.start_pos, result.char_interval.end_pos),
        expect_char_interval,
    )
    matched = source[
        result.char_interval.start_pos : result.char_interval.end_pos
    ]
    self.assertEqual(matched, expect_substring)

  @parameterized.named_parameters(
      dict(testcase_name="lcs", algorithm="lcs"),
      dict(testcase_name="legacy", algorithm="legacy"),
  )
  def test_planted_negative(self, algorithm):
    """Tokens absent from the source produce no alignment."""
    result = _run(
        _BASE_200,
        "warfarin coumadin anticoagulant",
        self._tokenizer,
        self._aligner,
        algorithm=algorithm,
    )
    self.assertIsNone(result)


class LcsBestSpanTest(parameterized.TestCase):
  """Unit tests for the pure LCS DP helper."""

  @parameterized.named_parameters(
      dict(
          testcase_name="repeated_token_min_span",
          source=["a", "a", "b"],
          extraction=["a", "b"],
          expect=(2, 1, 2),
      ),
      dict(
          testcase_name="single_token_reuse_forbidden",
          source=["a"],
          extraction=["a", "a"],
          expect=(1, 0, 0),
      ),
      dict(
          testcase_name="contiguous",
          source=["x", "a", "b", "c", "y"],
          extraction=["a", "b", "c"],
          expect=(3, 1, 3),
      ),
      dict(
          testcase_name="gapped",
          source=["a", "x", "b", "y", "c"],
          extraction=["a", "b", "c"],
          expect=(3, 0, 4),
      ),
      dict(
          testcase_name="negative",
          source=["a", "b", "c"],
          extraction=["x", "y", "z"],
          expect=(0, -1, -1),
      ),
      dict(
          testcase_name="tie_break_earliest_start",
          source=["a", "b", "c", "a", "b", "c"],
          extraction=["a", "b", "c"],
          expect=(3, 0, 2),
      ),
      dict(
          testcase_name="empty_source",
          source=[],
          extraction=["a"],
          expect=(0, -1, -1),
      ),
      dict(
          testcase_name="empty_extraction",
          source=["a"],
          extraction=[],
          expect=(0, -1, -1),
      ),
  )
  def test_best_lcs_span(self, source, extraction, expect):
    result = resolver_lib._best_lcs_span(source, extraction)
    self.assertEqual((result.matches, result.start, result.end), expect)


class LcsAcceptanceGateTest(parameterized.TestCase):
  """Tests for the coverage + density acceptance gate."""

  @parameterized.named_parameters(
      dict(
          testcase_name="perfect_coverage",
          matches=3,
          start=0,
          end=2,
          ext_len=3,
          expect=True,
      ),
      dict(
          testcase_name="density_ok_on_gapped",
          matches=3,
          start=0,
          end=8,
          ext_len=3,
          expect=True,
      ),
      dict(
          testcase_name="density_too_sparse",
          matches=3,
          start=0,
          end=9,
          ext_len=3,
          expect=False,
      ),
      dict(
          testcase_name="coverage_below_ceil",
          matches=2,
          start=0,
          end=2,
          ext_len=3,
          expect=False,
      ),
      dict(
          testcase_name="ceil_boundary",
          matches=3,
          start=0,
          end=2,
          ext_len=4,
          expect=True,
      ),
      dict(
          testcase_name="zero_matches",
          matches=0,
          start=-1,
          end=-1,
          ext_len=3,
          expect=False,
      ),
  )
  def test_accept_lcs_match(self, matches, start, end, ext_len, expect):
    span = resolver_lib.LcsSpan(matches=matches, start=start, end=end)
    self.assertEqual(resolver_lib._accept_lcs_match(span, ext_len), expect)


class LcsFuzzyAlignmentEdgeCasesTest(absltest.TestCase):
  """LCS-specific regression tests on _lcs_fuzzy_align_extraction."""

  def setUp(self):
    super().setUp()
    self._tokenizer = tokenizer_lib.RegexTokenizer()
    self._aligner = resolver_lib.WordAligner()
    resolver_lib._normalize_token.cache_clear()

  def test_repeated_token_selects_min_span(self):
    """With "a a b" and extraction "a b" the second "a" is chosen."""
    result = _run(
        "a a b",
        "a b",
        self._tokenizer,
        self._aligner,
        algorithm="lcs",
    )
    self.assertIsNotNone(result)
    self.assertEqual(
        (
            result.token_interval.start_index,
            result.token_interval.end_index,
        ),
        (1, 3),
    )
    self.assertEqual(
        (result.char_interval.start_pos, result.char_interval.end_pos),
        (2, 5),
    )

  def test_density_gate_rejects_over_sparse(self):
    """A 3-token extraction scattered over a 10-token span fails density."""
    source = (
        "metformin alpha beta gamma delta epsilon zeta hydrochloride eta tablet"
    )
    result = _run(
        source,
        "metformin hydrochloride tablet",
        self._tokenizer,
        self._aligner,
        algorithm="lcs",
    )
    self.assertIsNone(result)

  def test_algorithm_switch_legacy_rejects_lcs_accepts(self):
    source = "alpha beta gamma"
    extraction_text = "alpha beta gamma delta"

    lcs_result = _run(
        source,
        extraction_text,
        self._tokenizer,
        self._aligner,
        algorithm="lcs",
    )
    self.assertIsNotNone(lcs_result)
    self.assertEqual(
        (
            lcs_result.token_interval.start_index,
            lcs_result.token_interval.end_index,
        ),
        (0, 3),
    )

    legacy_result = _run(
        source,
        extraction_text,
        self._tokenizer,
        self._aligner,
        algorithm="legacy",
    )
    self.assertIsNone(legacy_result)

  def test_sparse_max_match_falls_back_to_dense_submatch(self):
    """Sparse k=m span failing density should not hide a denser k=m-1 span."""
    # Source: one leading extraction token, long noise, then a dense cluster
    # of the remaining extraction tokens. The 4-of-4 span is too sparse to
    # pass density, but the 3-of-4 dense suffix should still be accepted.
    source = (
        "alpha noise noise noise noise noise noise noise noise noise noise"
        " beta gamma delta"
    )
    result = _run(
        source,
        "alpha beta gamma delta",
        self._tokenizer,
        self._aligner,
        algorithm="lcs",
    )
    self.assertIsNotNone(result)
    self.assertEqual(
        (
            result.token_interval.start_index,
            result.token_interval.end_index,
        ),
        (11, 14),
    )

  def test_offsets_are_propagated(self):
    """Nonzero token and char offsets are added to the returned intervals."""
    result = _run(
        "metformin hydrochloride tablet",
        "metformin hydrochloride tablet",
        self._tokenizer,
        self._aligner,
        algorithm="lcs",
        token_offset=10,
        char_offset=100,
    )
    self.assertIsNotNone(result)
    self.assertEqual(
        (
            result.token_interval.start_index,
            result.token_interval.end_index,
        ),
        (10, 13),
    )
    self.assertEqual(
        (result.char_interval.start_pos, result.char_interval.end_pos),
        (100, 130),
    )


class PositionalCallCompatTest(absltest.TestCase):
  """Old positional call shapes must not break after adding new params."""

  def test_align_extractions_positional(self):
    """align_extractions(..., threshold, accept_lesser, tokenizer) still binds correctly."""
    aligner = resolver_lib.WordAligner()
    extraction = data.Extraction(
        extraction_class="med", extraction_text="aspirin"
    )
    source = "patient takes aspirin daily"
    groups = aligner.align_extractions(
        [[extraction]],
        source,
        0,  # token_offset
        0,  # char_offset
        "\u241F",  # delim
        True,  # enable_fuzzy_alignment
        0.75,  # fuzzy_alignment_threshold
        True,  # accept_match_lesser
        None,  # tokenizer_impl
    )
    self.assertLen(groups, 1)
    self.assertEqual(
        groups[0][0].alignment_status,
        data.AlignmentStatus.MATCH_EXACT,
    )

  def test_alignment_policy_positional(self):
    """AlignmentPolicy(True, 0.75, True) preserves old 3-arg shape."""
    policy = prompt_validation.AlignmentPolicy(True, 0.75, True)
    self.assertTrue(policy.enable_fuzzy_alignment)
    self.assertEqual(policy.fuzzy_alignment_threshold, 0.75)
    self.assertTrue(policy.accept_match_lesser)
    self.assertEqual(policy.fuzzy_alignment_algorithm, "lcs")

  def test_alignment_policy_rejects_fourth_positional(self):
    """New fields cannot be passed positionally."""
    with self.assertRaises(TypeError):
      prompt_validation.AlignmentPolicy(True, 0.75, True, "legacy")


class ParameterValidationTest(parameterized.TestCase):
  """Parameter validation at the alignment boundary."""

  def setUp(self):
    super().setUp()
    self._aligner = resolver_lib.WordAligner()
    self._extraction = data.Extraction(
        extraction_class="med", extraction_text="aspirin"
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="out_of_range_threshold",
          kwargs={"fuzzy_alignment_threshold": 1.5},
      ),
      dict(
          testcase_name="out_of_range_min_density",
          kwargs={"fuzzy_alignment_min_density": -0.1},
      ),
      dict(
          testcase_name="invalid_algorithm",
          kwargs={"fuzzy_alignment_algorithm": "bogus"},
      ),
  )
  def test_invalid_params_raise(self, kwargs):
    with self.assertRaises(ValueError):
      self._aligner.align_extractions(
          [[self._extraction]],
          "patient takes aspirin",
          **kwargs,
      )

  def test_disabled_fuzzy_skips_param_validation(self):
    """Bogus fuzzy params are ignored when fuzzy alignment is disabled."""
    self._aligner.align_extractions(
        [[self._extraction]],
        "patient takes aspirin",
        enable_fuzzy_alignment=False,
        fuzzy_alignment_algorithm="bogus",
        fuzzy_alignment_min_density=-0.1,
    )

  def test_disabled_fuzzy_skips_deprecation_warning(self):
    """Legacy algorithm selector is silent when fuzzy alignment is off."""
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      self._aligner.align_extractions(
          [[self._extraction]],
          "patient takes aspirin",
          enable_fuzzy_alignment=False,
          fuzzy_alignment_algorithm="legacy",
      )
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    self.assertEmpty(deprecations)

  def test_legacy_dispatch_warns(self):
    """Legacy algorithm emits DeprecationWarning at the dispatch site."""
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      self._aligner.align_extractions(
          [[self._extraction]],
          "patient takes aspirin",
          fuzzy_alignment_algorithm="legacy",
      )
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    self.assertLen(deprecations, 1)
    self.assertIn("legacy", str(deprecations[0].message))


if __name__ == "__main__":
  absltest.main()
