# Copyright 2026 Google LLC.
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

"""Regression tests for stored resolver parameters and per-call overrides."""

import unittest

from langextract import resolver as resolver_lib
from langextract.core import data


class ResolverParamsTest(unittest.TestCase):

  @staticmethod
  def _align(resolver, **kwargs):
    # Two of three extraction tokens match, so 0.6 succeeds and 0.8 fails.
    # Use fresh extractions because alignment mutates their status and spans.
    return list(
        resolver.align(
            [
                data.Extraction(
                    extraction_class="symptom",
                    extraction_text="headache and fever",
                )
            ],
            "Patient reports back pain and a fever.",
            token_offset=0,
            **kwargs,
        )
    )

  def test_resolver_accepts_alignment_params(self):
    resolver = resolver_lib.Resolver(
        fuzzy_alignment_threshold=0.6, enable_fuzzy_alignment=True
    )
    self.assertEqual(resolver.fuzzy_alignment_threshold, 0.6)
    self.assertTrue(resolver.enable_fuzzy_alignment)

  def test_align_uses_instance_threshold(self):
    resolver = resolver_lib.Resolver(
        fuzzy_alignment_threshold=0.6,
        enable_fuzzy_alignment=True,
        accept_match_lesser=False,
    )

    aligned = self._align(resolver)

    self.assertEqual(len(aligned), 1)
    self.assertEqual(
        aligned[0].alignment_status, data.AlignmentStatus.MATCH_FUZZY
    )

  def test_align_uses_instance_threshold_fail(self):
    resolver = resolver_lib.Resolver(
        fuzzy_alignment_threshold=0.8,
        enable_fuzzy_alignment=True,
        accept_match_lesser=False,
    )

    aligned = self._align(resolver)

    self.assertIsNone(aligned[0].alignment_status)

  def test_align_threshold_override_does_not_change_instance_setting(self):
    for stored, override, expected, stored_expected in [
        (0.6, 0.8, None, data.AlignmentStatus.MATCH_FUZZY),
        (0.8, 0.6, data.AlignmentStatus.MATCH_FUZZY, None),
    ]:
      with self.subTest(stored=stored, override=override):
        resolver = resolver_lib.Resolver(
            fuzzy_alignment_threshold=stored,
            enable_fuzzy_alignment=True,
            accept_match_lesser=False,
        )

        aligned = self._align(resolver, fuzzy_alignment_threshold=override)

        self.assertEqual(aligned[0].alignment_status, expected)
        self.assertEqual(
            self._align(resolver)[0].alignment_status, stored_expected
        )

  def test_align_fuzzy_override_does_not_change_instance_setting(self):
    for stored, override, expected, stored_expected in [
        (True, False, None, data.AlignmentStatus.MATCH_FUZZY),
        (False, True, data.AlignmentStatus.MATCH_FUZZY, None),
    ]:
      with self.subTest(stored=stored, override=override):
        resolver = resolver_lib.Resolver(
            fuzzy_alignment_threshold=0.6,
            enable_fuzzy_alignment=stored,
            accept_match_lesser=False,
        )

        aligned = self._align(resolver, enable_fuzzy_alignment=override)

        self.assertEqual(aligned[0].alignment_status, expected)
        self.assertEqual(
            self._align(resolver)[0].alignment_status, stored_expected
        )

  def test_align_lesser_override_does_not_change_instance_setting(self):
    for stored, override, expected, stored_expected in [
        (True, False, None, data.AlignmentStatus.MATCH_LESSER),
        (False, True, data.AlignmentStatus.MATCH_LESSER, None),
    ]:
      with self.subTest(stored=stored, override=override):
        resolver = resolver_lib.Resolver(
            enable_fuzzy_alignment=False, accept_match_lesser=stored
        )
        # Three of four tokens form a contiguous partial exact match.
        for call_kwargs, expected_status in [
            ({"accept_match_lesser": override}, expected),
            ({}, stored_expected),
        ]:
          aligned = list(
              resolver.align(
                  [
                      data.Extraction(
                          extraction_class="condition",
                          extraction_text="severe heart problems complications",
                      )
                  ],
                  "Patient has severe heart problems today.",
                  token_offset=0,
                  **call_kwargs,
              )
          )
          self.assertEqual(aligned[0].alignment_status, expected_status)

  def test_resolve_false_override_does_not_change_instance_setting(self):
    resolver = resolver_lib.Resolver(suppress_parse_errors=True)
    malformed_input = "```json\n{not valid json}\n```"

    self.assertEqual(resolver.resolve(malformed_input), [])
    with self.assertRaises(resolver_lib.ResolverParsingError):
      resolver.resolve(malformed_input, suppress_parse_errors=False)
    self.assertEqual(resolver.resolve(malformed_input), [])

  def test_resolve_true_override_does_not_change_instance_setting(self):
    resolver = resolver_lib.Resolver(suppress_parse_errors=False)
    malformed_input = "```json\n{not valid json}\n```"

    with self.assertRaises(resolver_lib.ResolverParsingError):
      resolver.resolve(malformed_input)
    self.assertEqual(
        resolver.resolve(malformed_input, suppress_parse_errors=True), []
    )
    with self.assertRaises(resolver_lib.ResolverParsingError):
      resolver.resolve(malformed_input)


if __name__ == "__main__":
  unittest.main()
