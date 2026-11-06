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

"""Tests for require_grounding parameter in extraction.py."""

import unittest

from langextract.core import data
from langextract.extraction import _filter_ungrounded_extractions


class FilterUngroundedExtractionsTest(unittest.TestCase):
  """Tests for _filter_ungrounded_extractions function."""

  def test_empty_list(self):
    """Returns empty list for empty input."""
    result = _filter_ungrounded_extractions([])
    self.assertEqual(result, [])

  def test_none_input(self):
    """Returns empty list for None input."""
    result = _filter_ungrounded_extractions(None)
    self.assertEqual(result, [])

  def test_filters_none_char_interval(self):
    """Filters out extractions with None char_interval."""
    extractions = [
        data.Extraction(
            extraction_class="test",
            extraction_text="grounded text",
            char_interval=data.CharInterval(start_pos=0, end_pos=13),
        ),
        data.Extraction(
            extraction_class="test",
            extraction_text="ungrounded text",
            char_interval=None,
        ),
    ]
    result = _filter_ungrounded_extractions(extractions)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0].extraction_text, "grounded text")

  def test_filters_none_start_pos(self):
    """Filters out extractions with None start_pos."""
    extractions = [
        data.Extraction(
            extraction_class="test",
            extraction_text="grounded text",
            char_interval=data.CharInterval(start_pos=0, end_pos=13),
        ),
        data.Extraction(
            extraction_class="test",
            extraction_text="ungrounded text",
            char_interval=data.CharInterval(start_pos=None, end_pos=15),
        ),
    ]
    result = _filter_ungrounded_extractions(extractions)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0].extraction_text, "grounded text")

  def test_filters_none_end_pos(self):
    """Filters out extractions with None end_pos."""
    extractions = [
        data.Extraction(
            extraction_class="test",
            extraction_text="grounded text",
            char_interval=data.CharInterval(start_pos=0, end_pos=13),
        ),
        data.Extraction(
            extraction_class="test",
            extraction_text="ungrounded text",
            char_interval=data.CharInterval(start_pos=0, end_pos=None),
        ),
    ]
    result = _filter_ungrounded_extractions(extractions)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0].extraction_text, "grounded text")

  def test_keeps_all_grounded(self):
    """Keeps all extractions when all are grounded."""
    extractions = [
        data.Extraction(
            extraction_class="test",
            extraction_text="first",
            char_interval=data.CharInterval(start_pos=0, end_pos=5),
        ),
        data.Extraction(
            extraction_class="test",
            extraction_text="second",
            char_interval=data.CharInterval(start_pos=10, end_pos=16),
        ),
    ]
    result = _filter_ungrounded_extractions(extractions)
    self.assertEqual(len(result), 2)

  def test_filters_all_ungrounded(self):
    """Returns empty list when all extractions are ungrounded."""
    extractions = [
        data.Extraction(
            extraction_class="test",
            extraction_text="ungrounded1",
            char_interval=None,
        ),
        data.Extraction(
            extraction_class="test",
            extraction_text="ungrounded2",
            char_interval=data.CharInterval(start_pos=None, end_pos=None),
        ),
    ]
    result = _filter_ungrounded_extractions(extractions)
    self.assertEqual(len(result), 0)

  def test_preserves_extraction_attributes(self):
    """Preserves all attributes of grounded extractions."""
    extractions = [
        data.Extraction(
            extraction_class="medication",
            extraction_text="aspirin",
            char_interval=data.CharInterval(start_pos=10, end_pos=17),
            alignment_status=data.AlignmentStatus.MATCH_EXACT,
            extraction_index=1,
            group_index=0,
            description="A medication",
            attributes={"dosage": "100mg"},
        ),
    ]
    result = _filter_ungrounded_extractions(extractions)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0].extraction_class, "medication")
    self.assertEqual(result[0].extraction_text, "aspirin")
    self.assertEqual(
        result[0].alignment_status, data.AlignmentStatus.MATCH_EXACT
    )
    self.assertEqual(result[0].extraction_index, 1)
    self.assertEqual(result[0].group_index, 0)
    self.assertEqual(result[0].description, "A medication")
    self.assertEqual(result[0].attributes, {"dosage": "100mg"})


if __name__ == "__main__":
  unittest.main()
