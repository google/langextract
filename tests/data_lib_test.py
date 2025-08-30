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

import json

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from langextract import data_lib
from langextract import io
from langextract.core import data
from langextract.core import tokenizer


class DataLibToDictParameterizedTest(parameterized.TestCase):
  """Tests conversion of AnnotatedDocument objects to JSON dicts.

  Verifies that `annotated_document_to_dict` correctly serializes documents by:
  - Excluding private fields (e.g., token_interval).
  - Converting all expected extraction attributes properly.
  - Handling int64 values for extraction indexes.
  """

  @parameterized.named_parameters(
      dict(
          testcase_name="single_extraction_no_token_interval",
          annotated_doc=data.AnnotatedDocument(
              document_id="docA",
              text="Just a short sentence.",
              extractions=[
                  data.Extraction(
                      extraction_class="note",
                      extraction_text="short sentence",
                      extraction_index=1,
                      group_index=0,
                  ),
              ],
          ),
          expected_dict={
              "document_id": "docA",
              "extractions": [
                  {
                      "extraction_class": "note",
                      "extraction_text": "short sentence",
                      "char_interval": None,
                      "alignment_status": None,
                      "extraction_index": 1,
                      "group_index": 0,
                      "description": None,
                      "attributes": None,
                  },
              ],
              "text": "Just a short sentence.",
          },
      ),
      dict(
          testcase_name="multiple_extractions_with_token_interval",
          annotated_doc=data.AnnotatedDocument(
              document_id="docB",
              text="Patient Jane reported a headache.",
              extractions=[
                  data.Extraction(
                      extraction_class="patient",
                      extraction_text="Jane",
                      extraction_index=1,
                      group_index=0,
                  ),
                  data.Extraction(
                      extraction_class="symptom",
                      extraction_text="headache",
                      extraction_index=2,
                      group_index=0,
                      char_interval=data.CharInterval(start_pos=24, end_pos=32),
                      token_interval=tokenizer.TokenInterval(
                          start_index=4, end_index=5
                      ),  # should be ignored
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  ),
              ],
          ),
          expected_dict={
              "document_id": "docB",
              "extractions": [
                  {
                      "extraction_class": "patient",
                      "extraction_text": "Jane",
                      "char_interval": None,
                      "alignment_status": None,
                      "extraction_index": 1,
                      "group_index": 0,
                      "description": None,
                      "attributes": None,
                  },
                  {
                      "extraction_class": "symptom",
                      "extraction_text": "headache",
                      "char_interval": {"start_pos": 24, "end_pos": 32},
                      "alignment_status": "match_exact",
                      "extraction_index": 2,
                      "group_index": 0,
                      "description": None,
                      "attributes": None,
                  },
              ],
              "text": "Patient Jane reported a headache.",
          },
      ),
      dict(
          testcase_name="extraction_with_attributes_and_token_interval",
          annotated_doc=data.AnnotatedDocument(
              document_id="docC",
              text="He has mild chest pain and a cough.",
              extractions=[
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="chest pain",
                      extraction_index=2,
                      group_index=1,
                      attributes={
                          "severity": "mild",
                          "persistence": "persistent",
                      },
                      char_interval=data.CharInterval(start_pos=12, end_pos=22),
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=5
                      ),  # should be ignored
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  ),
                  data.Extraction(
                      extraction_class="symptom",
                      extraction_text="cough",
                      extraction_index=3,
                      group_index=1,
                  ),
              ],
          ),
          expected_dict={
              "document_id": "docC",
              "extractions": [
                  {
                      "extraction_class": "condition",
                      "extraction_text": "chest pain",
                      "char_interval": {"start_pos": 12, "end_pos": 22},
                      "alignment_status": "match_exact",
                      "extraction_index": 2,
                      "group_index": 1,
                      "description": None,
                      "attributes": {
                          "severity": "mild",
                          "persistence": "persistent",
                      },
                  },
                  {
                      "extraction_class": "symptom",
                      "extraction_text": "cough",
                      "char_interval": None,
                      "alignment_status": None,
                      "extraction_index": 3,
                      "group_index": 1,
                      "description": None,
                      "attributes": None,
                  },
              ],
              "text": "He has mild chest pain and a cough.",
          },
      ),
  )
  def test_annotated_document_to_dict(self, annotated_doc, expected_dict):
    actual_dict = data_lib.annotated_document_to_dict(annotated_doc)
    self.assertDictEqual(
        actual_dict,
        expected_dict,
        "annotated_document_to_dict() output differs from expected JSON dict.",
    )

  def test_annotated_document_to_dict_with_int64(self):
    doc = data.AnnotatedDocument(
        document_id="doc_int64",
        text="Sample text with int64 index",
        extractions=[
            data.Extraction(
                extraction_class="demo_extraction",
                extraction_text="placeholder",
                extraction_index=np.int64(42),  # pytype: disable=wrong-arg-types
            ),
        ],
    )

    doc_dict = data_lib.annotated_document_to_dict(doc)

    json_str = json.dumps(doc_dict, ensure_ascii=False)
    self.assertIn('"extraction_index": 42', json_str)


class IsUrlTest(absltest.TestCase):
  """Tests for io.is_url function validation."""

  def test_valid_urls(self):
    """Test that valid URLs are recognized."""
    self.assertTrue(io.is_url("http://example.com"))
    self.assertTrue(io.is_url("https://www.example.com"))
    self.assertTrue(io.is_url("http://localhost:8080"))
    self.assertTrue(io.is_url("http://192.168.1.1"))
    self.assertTrue(io.is_url("http://[2001:db8::1]"))  # IPv6
    self.assertTrue(io.is_url("http://[::1]:8080"))  # IPv6 localhost with port

  def test_invalid_urls_with_text(self):
    """Test that URLs with additional text are rejected."""
    # Validates fix for issue where text starting with URL was incorrectly fetched
    self.assertFalse(io.is_url("http://example.com is a website"))
    self.assertFalse(io.is_url("http://medical-journal.com published a study"))

  def test_invalid_urls_no_scheme(self):
    """Test that URLs without proper scheme are rejected."""
    self.assertFalse(io.is_url("example.com"))
    self.assertFalse(io.is_url("www.example.com"))
    self.assertFalse(io.is_url("ftp://example.com"))


if __name__ == "__main__":
  absltest.main()
