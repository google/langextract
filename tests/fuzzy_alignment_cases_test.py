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

from absl.testing import absltest
from absl.testing import parameterized

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


def _run(source, extraction_text, tokenizer, aligner):
  """Runs _fuzzy_align_extraction and returns the result."""
  tokenized = tokenizer.tokenize(source)
  source_tokens = [
      source[t.char_interval.start_pos : t.char_interval.end_pos].lower()
      for t in tokenized.tokens
  ]
  extraction = data.Extraction(
      extraction_class="entity", extraction_text=extraction_text
  )
  return aligner._fuzzy_align_extraction(
      extraction=extraction,
      source_tokens=source_tokens,
      tokenized_text=tokenized,
      token_offset=0,
      char_offset=0,
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


class FuzzyAlignmentCasesTest(parameterized.TestCase):
  """Planted-span oracle tests for _fuzzy_align_extraction."""

  def setUp(self):
    super().setUp()
    self._tokenizer = tokenizer_lib.RegexTokenizer()
    self._aligner = resolver_lib.WordAligner()
    resolver_lib._normalize_token.cache_clear()

  @parameterized.named_parameters(
      dict(
          testcase_name="contiguous",
          source=_PLANTED,
          extraction_text="metformin hydrochloride tablet",
          expect_token_interval=(50, 53),
          expect_char_interval=(451, 481),
          expect_substring="metformin hydrochloride tablet",
      ),
      dict(
          testcase_name="fuzzy_stemming",
          source=_PLANTED,
          extraction_text="metformins hydrochlorides tablets",
          expect_token_interval=(50, 53),
          expect_char_interval=(451, 481),
          expect_substring="metformin hydrochloride tablet",
      ),
      dict(
          testcase_name="gapped",
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
  def test_planted_positive(
      self,
      source,
      extraction_text,
      expect_token_interval,
      expect_char_interval,
      expect_substring,
  ):
    """Planted spans align to their expected token and char positions."""
    result = _run(source, extraction_text, self._tokenizer, self._aligner)

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

  def test_planted_negative(self):
    """Tokens absent from the source produce no alignment."""
    result = _run(
        _BASE_200,
        "warfarin coumadin anticoagulant",
        self._tokenizer,
        self._aligner,
    )
    self.assertIsNone(result)


if __name__ == "__main__":
  absltest.main()
