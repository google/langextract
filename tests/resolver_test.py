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

import textwrap
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized

from langextract import chunking
from langextract import data
from langextract import resolver as resolver_lib
from langextract import schema
from langextract import tokenizer


def assert_char_interval_match_source(
    test_case: absltest.TestCase,
    source_text: str,
    extractions: Sequence[data.Extraction],
):
  """Asserts that the char_interval of matched extractions matches the source text.

  Args:
    test_case: The TestCase instance.
    source_text: The original source text.
    extractions: A sequence of extractions to check.
  """
  for extraction in extractions:
    if extraction.alignment_status == data.AlignmentStatus.MATCH_EXACT:
      assert (
          extraction.char_interval is not None
      ), "char_interval should not be None for AlignmentStatus.MATCH_EXACT"

      char_int = extraction.char_interval
      start = char_int.start_pos
      end = char_int.end_pos
      test_case.assertIsNotNone(start, "start_pos should not be None")
      test_case.assertIsNotNone(end, "end_pos should not be None")
      extracted = source_text[start:end]
      test_case.assertEqual(
          extracted.lower(),
          extraction.extraction_text.lower(),
          f"Extraction '{extraction.extraction_text}' does not match extracted"
          f" '{extracted}' using char_interval {char_int}",
      )


class ParserTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="json_invalid_input",
          resolver=resolver_lib.Resolver(
              format_type=data.FormatType.JSON,
              fence_output=True,
          ),
          input_text="invalid input",
          expected_exception=ValueError,
          expected_regex=".*valid markers.*",
      ),
      dict(
          testcase_name="json_missing_markers",
          resolver=resolver_lib.Resolver(
              format_type=data.FormatType.JSON,
              fence_output=True,
          ),
          input_text='[{"key": "value"}]',
          expected_exception=ValueError,
          expected_regex=".*valid markers.*",
      ),
      dict(
          testcase_name="json_empty_string",
          resolver=resolver_lib.Resolver(
              format_type=data.FormatType.JSON,
              fence_output=True,
          ),
          input_text="",
          expected_exception=ValueError,
          expected_regex=".*must be a non-empty string.*",
      ),
      dict(
          testcase_name="json_partial_markers",
          resolver=resolver_lib.Resolver(
              format_type=data.FormatType.JSON,
              fence_output=True,
          ),
          input_text='```json\n{"key": "value"',
          expected_exception=ValueError,
          expected_regex=".*valid markers.*",
      ),
      dict(
          testcase_name="yaml_invalid_input",
          resolver=resolver_lib.Resolver(
              format_type=data.FormatType.YAML,
              fence_output=True,
          ),
          input_text="invalid input",
          expected_exception=ValueError,
          expected_regex=".*valid markers.*",
      ),
      dict(
          testcase_name="yaml_missing_markers",
          resolver=resolver_lib.Resolver(
              format_type=data.FormatType.YAML,
              fence_output=True,
          ),
          input_text='[{"key": "value"}]',
          expected_exception=ValueError,
          expected_regex=".*valid markers.*",
      ),
      dict(
          testcase_name="yaml_empty_content",
          resolver=resolver_lib.Resolver(
              format_type=data.FormatType.YAML,
              fence_output=True,
          ),
          input_text="```yaml\n```",
          expected_exception=resolver_lib.ResolverParsingError,
          expected_regex=(
              ".*Content must be a mapping with an"
              f" '{schema.EXTRACTIONS_KEY}' key.*"
          ),
      ),
  )
  def test_parser_error_cases(
      self, resolver, input_text, expected_exception, expected_regex
  ):
    with self.assertRaisesRegex(expected_exception, expected_regex):
      resolver.string_to_extraction_data(input_text)


class ExtractOrderedEntitiesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="valid_input",
          test_input=[
              {
                  "medication": "Naprosyn",
                  "medication_index": 4,
                  "frequency": "as needed",
                  "frequency_index": 5,
                  "reason": "pain",
                  "reason_index": 8,
              },
              {
                  "medication": "prednisone",
                  "medication_index": 5,
                  "frequency": "daily",
                  "frequency_index": 1,
              },
          ],
          expected_output=[
              data.Extraction(
                  extraction_class="frequency",
                  extraction_text="daily",
                  extraction_index=1,
                  group_index=1,
              ),
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Naprosyn",
                  extraction_index=4,
                  group_index=0,
              ),
              data.Extraction(
                  extraction_class="frequency",
                  extraction_text="as needed",
                  extraction_index=5,
                  group_index=0,
              ),
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="prednisone",
                  extraction_index=5,
                  group_index=1,
              ),
              data.Extraction(
                  extraction_class="reason",
                  extraction_text="pain",
                  extraction_index=8,
                  group_index=0,
              ),
          ],
      ),
      dict(
          testcase_name="empty_input",
          test_input=[],
          expected_output=[],
      ),
      dict(
          testcase_name="mixed_index_order",
          test_input=[
              {
                  "medication": "Ibuprofen",
                  "medication_index": 2,
                  "dosage": "400mg",
                  "dosage_index": 1,
              },
              {
                  "medication": "Acetaminophen",
                  "medication_index": 1,
                  "duration": "7 days",
                  "duration_index": 2,
              },
          ],
          expected_output=[
              data.Extraction(
                  extraction_class="dosage",
                  extraction_text="400mg",
                  extraction_index=1,
                  group_index=0,
              ),
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Acetaminophen",
                  extraction_index=1,
                  group_index=1,
              ),
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Ibuprofen",
                  extraction_index=2,
                  group_index=0,
              ),
              data.Extraction(
                  extraction_class="duration",
                  extraction_text="7 days",
                  extraction_index=2,
                  group_index=1,
              ),
          ],
      ),
      dict(
          testcase_name="missing_index_key",
          test_input=[{
              "medication": "Aspirin",
              "dosage": "325mg",
              "dosage_index": 1,
          }],
          expected_output=[
              data.Extraction(
                  extraction_class="dosage",
                  extraction_text="325mg",
                  extraction_index=1,
                  group_index=0,
              ),
          ],
      ),
      dict(
          testcase_name="all_indices_missing",
          test_input=[
              {"medication": "Aspirin", "dosage": "325mg"},
              {"medication": "Ibuprofen", "dosage": "400mg"},
          ],
          expected_output=[],
      ),
      dict(
          testcase_name="single_element_dictionaries",
          test_input=[
              {"medication": "Aspirin", "medication_index": 1},
              {"medication": "Ibuprofen", "medication_index": 2},
          ],
          expected_output=[
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Aspirin",
                  extraction_index=1,
                  group_index=0,
              ),
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Ibuprofen",
                  extraction_index=2,
                  group_index=1,
              ),
          ],
      ),
      dict(
          testcase_name="duplicate_indices_unchanged",
          test_input=[{
              "medication": "Aspirin",
              "medication_index": 1,
              "dosage": "325mg",
              "dosage_index": 1,
              "form": "tablet",
              "form_index": 1,
          }],
          expected_output=[
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Aspirin",
                  extraction_index=1,
                  group_index=0,
              ),
              data.Extraction(
                  extraction_class="dosage",
                  extraction_text="325mg",
                  extraction_index=1,
                  group_index=0,
              ),
              data.Extraction(
                  extraction_class="form",
                  extraction_text="tablet",
                  extraction_index=1,
                  group_index=0,
              ),
          ],
      ),
      dict(
          testcase_name="negative_indices",
          test_input=[{
              "medication": "Aspirin",
              "medication_index": -1,
              "dosage": "325mg",
              "dosage_index": -2,
          }],
          expected_output=[
              data.Extraction(
                  extraction_class="dosage",
                  extraction_text="325mg",
                  extraction_index=-2,
                  group_index=0,
              ),
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Aspirin",
                  extraction_index=-1,
                  group_index=0,
              ),
          ],
      ),
      dict(
          testcase_name="index_without_data_key_ignored",
          test_input=[{
              "medication_index": 1,
              "dosage": "325mg",
              "dosage_index": 2,
          }],
          expected_output=[
              data.Extraction(
                  extraction_class="dosage",
                  extraction_text="325mg",
                  extraction_index=2,
                  group_index=0,
              ),
          ],
      ),
      dict(
          testcase_name="no_index_suffix",
          resolver=resolver_lib.Resolver(
              extraction_index_suffix=None,
              format_type=data.FormatType.JSON,
          ),
          test_input=[
              {"medication": "Aspirin"},
              {"medication": "Ibuprofen"},
              {"dosage": "325mg"},
              {"dosage": "400mg"},
          ],
          expected_output=[
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Aspirin",
                  extraction_index=1,
                  group_index=0,
              ),
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Ibuprofen",
                  extraction_index=2,
                  group_index=1,
              ),
              data.Extraction(
                  extraction_class="dosage",
                  extraction_text="325mg",
                  extraction_index=3,
                  group_index=2,
              ),
              data.Extraction(
                  extraction_class="dosage",
                  extraction_text="400mg",
                  extraction_index=4,
                  group_index=3,
              ),
          ],
      ),
      dict(
          testcase_name="attributes_suffix",
          resolver=resolver_lib.Resolver(
              extraction_index_suffix=None,
              format_type=data.FormatType.JSON,
          ),
          test_input=[
              {
                  "patient": "Jane Doe",
                  "patient_attributes": {
                      "PERSON": "True",
                      "IDENTIFIABLE": "True",
                  },
              },
              {
                  "medication": "Lisinopril",
                  "medication_attributes": {
                      "THERAPEUTIC": "True",
                      "CLINICAL": "True",
                  },
              },
          ],
          expected_output=[
              data.Extraction(
                  extraction_class="patient",
                  extraction_text="Jane Doe",
                  extraction_index=1,
                  group_index=0,
                  attributes={
                      "PERSON": "True",
                      "IDENTIFIABLE": "True",
                  },
              ),
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Lisinopril",
                  extraction_index=2,
                  group_index=1,
                  attributes={
                      "THERAPEUTIC": "True",
                      "CLINICAL": "True",
                  },
              ),
          ],
      ),
      dict(
          testcase_name="indices_and_attributes",
          test_input=[
              {
                  "patient": "John Doe",
                  "patient_index": 2,
                  "patient_attributes": {
                      "IDENTIFIABLE": "True",
                  },
                  "condition": "hypertension",
                  "condition_index": 1,
                  "condition_attributes": {
                      "CHRONIC_CONDITION": "True",
                      "REQUIRES_MANAGEMENT": "True",
                  },
              },
              {
                  "medication": "Lisinopril",
                  "medication_index": 3,
                  "medication_attributes": {
                      "ANTIHYPERTENSIVE_MEDICATION": "True",
                      "DAILY_USE": "True",
                  },
                  "dosage": "10mg",
                  "dosage_index": 4,
                  "dosage_attributes": {
                      "STANDARD_DAILY_DOSE": "True",
                  },
              },
          ],
          expected_output=[
              data.Extraction(
                  extraction_class="condition",
                  extraction_text="hypertension",
                  extraction_index=1,
                  group_index=0,
                  attributes={
                      "CHRONIC_CONDITION": "True",
                      "REQUIRES_MANAGEMENT": "True",
                  },
              ),
              data.Extraction(
                  extraction_class="patient",
                  extraction_text="John Doe",
                  extraction_index=2,
                  group_index=0,
                  attributes={
                      "IDENTIFIABLE": "True",
                  },
              ),
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Lisinopril",
                  extraction_index=3,
                  group_index=1,
                  attributes={
                      "ANTIHYPERTENSIVE_MEDICATION": "True",
                      "DAILY_USE": "True",
                  },
              ),
              data.Extraction(
                  extraction_class="dosage",
                  extraction_text="10mg",
                  extraction_index=4,
                  group_index=1,
                  attributes={
                      "STANDARD_DAILY_DOSE": "True",
                  },
              ),
          ],
      ),
  )
  def test_extract_ordered_extractions_success(
      self,
      test_input,
      resolver=resolver_lib.Resolver(),
      expected_output=None,
  ):
    actual_output = resolver.extract_ordered_extractions(test_input)
    self.assertEqual(actual_output, expected_output)

  @parameterized.named_parameters(
      dict(
          testcase_name="non_integer_indices",
          resolver=resolver_lib.Resolver(
              format_type=data.FormatType.JSON,
          ),
          test_input=[{
              "medication": "Aspirin",
              "medication_index": "first",
              "dosage": "325mg",
              "dosage_index": "second",
          }],
          expected_exception=ValueError,
          expected_regex=".*string or integer.*",
      ),
      dict(
          testcase_name="float_indices",
          resolver=resolver_lib.Resolver(
              format_type=data.FormatType.JSON,
          ),
          test_input=[{"medication": "Aspirin", "medication_index": 1.0}],
          expected_exception=ValueError,
          expected_regex=".*string or integer.*",
      ),
  )
  def test_extract_ordered_extractions_exceptions(
      self, resolver, test_input, expected_exception, expected_regex
  ):
    with self.assertRaisesRegex(expected_exception, expected_regex):
      resolver.extract_ordered_extractions(test_input)


class AlignEntitiesTest(parameterized.TestCase):
  _SOURCE_TEXT_TWO_MEDS = (
      "Patient is prescribed Naprosyn and prednisone for treatment."
  )
  _SOURCE_TEXT_THREE_CONDITIONS_AND_MEDS = (
      "Patient with arthritis, fever, and inflammation is prescribed"
      " Naprosyn, prednisone, and ibuprofen."
  )
  _SOURCE_TEXT_MULTI_WORD_EXTRACTIONS = (
      "Pt was prescribed Naprosyn as needed for pain and prednisone for"
      " one month."
  )

  def setUp(self):
    super().setUp()
    self.aligner = resolver_lib.WordAligner()
    self.maxDiff = 10000

  @parameterized.named_parameters(
      (
          "basic_alignment",
          [
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Naprosyn"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                  )
              ],
          ],
          _SOURCE_TEXT_TWO_MEDS,
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=4
                      ),
                      char_interval=data.CharInterval(start_pos=22, end_pos=30),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                      token_interval=tokenizer.TokenInterval(
                          start_index=5, end_index=6
                      ),
                      char_interval=data.CharInterval(start_pos=35, end_pos=45),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "shuffled_order_of_last_two_extractions",
          [
              [
                  data.Extraction(
                      extraction_class="condition", extraction_text="arthritis"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition", extraction_text="fever"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="inflammation",
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Naprosyn"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="ibuprofen"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                  )
              ],
          ],
          _SOURCE_TEXT_THREE_CONDITIONS_AND_MEDS,
          # Indexes Aligned with Tokens
          # --------------------------------------------------------------------
          # Index    | 0        1      2         3      4      5     6
          # Token    | Patient  with   arthritis ,     fever   ,     and
          # --------------------------------------------------------------------
          # Index    | 7              8        9
          # Token    | inflammation  is       prescribed
          # --------------------------------------------------------------------
          # Index    | 10       11        12         13   14      15
          # Token    | Naprosyn ,         prednisone ,    and     ibuprofen
          # --------------------------------------------------------------------
          # Index    | 16
          # Token    | .
          [
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="arthritis",
                      token_interval=tokenizer.TokenInterval(
                          start_index=2, end_index=3
                      ),
                      char_interval=data.CharInterval(start_pos=13, end_pos=22),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="fever",
                      token_interval=tokenizer.TokenInterval(
                          start_index=4, end_index=5
                      ),
                      char_interval=data.CharInterval(start_pos=24, end_pos=29),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="inflammation",
                      token_interval=tokenizer.TokenInterval(
                          start_index=7, end_index=8
                      ),
                      char_interval=data.CharInterval(start_pos=35, end_pos=47),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=10, end_index=11
                      ),
                      char_interval=data.CharInterval(start_pos=62, end_pos=70),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="ibuprofen",
                      token_interval=None,
                      char_interval=None,
                      alignment_status=None,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                      token_interval=tokenizer.TokenInterval(
                          start_index=12, end_index=13
                      ),
                      char_interval=data.CharInterval(start_pos=72, end_pos=82),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "extraction_not_found",
          [[
              data.Extraction(
                  extraction_class="medication", extraction_text="aspirin"
              )
          ]],
          _SOURCE_TEXT_TWO_MEDS,
          [[
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="aspirin",
                  char_interval=None,
              )
          ]],
      ),
      (
          "multiple_word_extraction_partially_matched",
          [[
              data.Extraction(
                  extraction_class="condition",
                  extraction_text="high blood pressure",
              )
          ]],
          "Patient is prescribed high glucose.",
          [[
              data.Extraction(
                  extraction_class="condition",
                  extraction_text="high blood pressure",
                  token_interval=tokenizer.TokenInterval(
                      start_index=3, end_index=4
                  ),
                  alignment_status=data.AlignmentStatus.MATCH_LESSER,
                  char_interval=data.CharInterval(start_pos=22, end_pos=26),
              )
          ]],
      ),
      (
          "optimize_multiword_extractions_at_back",
          [
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Naprosyn"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn and prednisone",
                  )
              ],
          ],
          _SOURCE_TEXT_TWO_MEDS,
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      token_interval=None,
                      char_interval=None,
                      alignment_status=None,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn and prednisone",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=6
                      ),
                      char_interval=data.CharInterval(start_pos=22, end_pos=45),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "optimize_multiword_extractions_at_front",
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn and prednisone",
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Naprosyn"
                  )
              ],
          ],
          _SOURCE_TEXT_TWO_MEDS,
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn and prednisone",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=6
                      ),
                      char_interval=data.CharInterval(start_pos=22, end_pos=45),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      char_interval=None,
                  )
              ],
          ],
      ),
      (
          "test_en_dash_unicode_handling",
          [
              [
                  data.Extraction(
                      extraction_class="word", extraction_text="Separated"
                  )
              ],
              [data.Extraction(extraction_class="word", extraction_text="by")],
              [
                  data.Extraction(
                      extraction_class="word", extraction_text="en–dashes"
                  )
              ],
          ],
          "Separated–by–en–dashes.",
          [
              [
                  data.Extraction(
                      extraction_class="word",
                      extraction_text="Separated",
                      token_interval=tokenizer.TokenInterval(
                          start_index=0, end_index=1
                      ),
                      char_interval=data.CharInterval(start_pos=0, end_pos=9),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="word",
                      extraction_text="by",
                      token_interval=tokenizer.TokenInterval(
                          start_index=2, end_index=3
                      ),
                      char_interval=data.CharInterval(start_pos=10, end_pos=12),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="word",
                      extraction_text="en–dashes",
                      token_interval=tokenizer.TokenInterval(
                          start_index=4, end_index=7
                      ),
                      char_interval=data.CharInterval(start_pos=13, end_pos=22),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "empty_source_text",
          [[
              data.Extraction(
                  extraction_class="medication", extraction_text="Naprosyn"
              )
          ]],
          "",
          ValueError,
      ),
      (
          "special_characters_in_extractions",
          [[
              data.Extraction(
                  extraction_class="medication", extraction_text="Napro-syn"
              )
          ]],
          "Patient is prescribed Napro-syn.",
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Napro-syn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=6
                      ),
                      char_interval=data.CharInterval(start_pos=22, end_pos=31),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "test_extraction_with_substring_of_another_not_matched",
          [[
              data.Extraction(
                  extraction_class="medication", extraction_text="Napro"
              )
          ]],
          _SOURCE_TEXT_TWO_MEDS,
          [[
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Napro",
                  char_interval=None,
              )
          ]],
      ),
      (
          "test_empty_extractions_list",
          [],
          _SOURCE_TEXT_TWO_MEDS,
          [],
      ),
      (
          "test_extractions_with_similar_words",
          [
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Naprosyn"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Napro"
                  )
              ],
          ],
          _SOURCE_TEXT_TWO_MEDS,
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=4
                      ),
                      char_interval=data.CharInterval(start_pos=22, end_pos=30),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Napro",
                      char_interval=None,
                  )
              ],
          ],
      ),
      (
          "test_source_text_with_repeated_extractions",
          [[
              data.Extraction(
                  extraction_class="medication", extraction_text="Naprosyn"
              )
          ]],
          "Patient is prescribed Naprosyn and Naprosyn.",
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=4
                      ),
                      char_interval=data.CharInterval(start_pos=22, end_pos=30),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "test_interleaved_extractions",
          [
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Naprosyn"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition", extraction_text="arthritis"
                  )
              ],
          ],
          "Patient with arthritis is prescribed Naprosyn.",
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      char_interval=None,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="arthritis",
                      token_interval=tokenizer.TokenInterval(
                          start_index=2, end_index=3
                      ),
                      char_interval=data.CharInterval(start_pos=13, end_pos=22),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "overlapping_extractions_different_types",
          [
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Naprosyn"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="Naprosyn allergy",
                  )
              ],
          ],
          _SOURCE_TEXT_TWO_MEDS,
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=4
                      ),
                      char_interval=data.CharInterval(start_pos=22, end_pos=30),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="Naprosyn allergy",
                      char_interval=None,
                  )
              ],
          ],
      ),
      (
          "test_overlapping_text_extractions_with_overlapping_source",
          [
              [
                  data.Extraction(
                      extraction_class="condition", extraction_text="high blood"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="blood pressure",
                  )
              ],
          ],
          "Patient has high blood pressure.",
          [
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="high blood",
                      token_interval=tokenizer.TokenInterval(
                          start_index=2, end_index=4
                      ),
                      char_interval=data.CharInterval(start_pos=12, end_pos=22),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="blood pressure",
                      char_interval=None,
                  )
              ],
          ],
      ),
      (
          "test_multiple_instances_same_extraction",
          [
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Naprosyn"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                  )
              ],
          ],
          "Naprosyn, prednisone, and again Naprosyn are prescribed.",
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=0, end_index=1
                      ),
                      char_interval=data.CharInterval(start_pos=0, end_pos=8),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                      token_interval=tokenizer.TokenInterval(
                          start_index=2, end_index=3
                      ),
                      char_interval=data.CharInterval(start_pos=10, end_pos=20),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "test_longer_extraction_spanning_multiple_words",
          [[
              data.Extraction(
                  extraction_class="condition",
                  extraction_text="rheumatoid arthritis",
              )
          ]],
          "Patient diagnosed with rheumatoid arthritis.",
          [
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="rheumatoid arthritis",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=5
                      ),
                      char_interval=data.CharInterval(start_pos=23, end_pos=43),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "test_case_insensitivity",
          [
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="naprosyn"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="PREDNISONE",
                  )
              ],
          ],
          _SOURCE_TEXT_TWO_MEDS.lower(),
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="naprosyn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=4
                      ),
                      char_interval=data.CharInterval(start_pos=22, end_pos=30),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="PREDNISONE",
                      token_interval=tokenizer.TokenInterval(
                          start_index=5, end_index=6
                      ),
                      char_interval=data.CharInterval(start_pos=35, end_pos=45),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "numerical_extractions",
          [[
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="Ibuprofen 600mg",
              )
          ]],
          "Patient was given Ibuprofen 600mg twice daily.",
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Ibuprofen 600mg",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=6
                      ),
                      char_interval=data.CharInterval(start_pos=18, end_pos=33),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "test_extractions_spanning_across_sentence_boundaries",
          [
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Ibuprofen"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="instruction",
                      extraction_text="take with food",
                  )
              ],
          ],
          "Take Ibuprofen. Always take with food.",
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Ibuprofen",
                      token_interval=tokenizer.TokenInterval(
                          start_index=1, end_index=2
                      ),
                      char_interval=data.CharInterval(start_pos=5, end_pos=14),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="instruction",
                      extraction_text="take with food",
                      token_interval=tokenizer.TokenInterval(
                          start_index=4, end_index=7
                      ),
                      char_interval=data.CharInterval(start_pos=23, end_pos=37),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "test_multiple_multiword_extractions_multi_group",
          [
              [
                  data.Extraction(
                      extraction_class="medication", extraction_text="Naprosyn"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="frequency", extraction_text="as needed"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="reason", extraction_text="pain"
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="duration",
                      extraction_text="for one month",
                  )
              ],
          ],
          _SOURCE_TEXT_MULTI_WORD_EXTRACTIONS,
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Naprosyn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=4
                      ),
                      char_interval=data.CharInterval(start_pos=18, end_pos=26),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="frequency",
                      extraction_text="as needed",
                      token_interval=tokenizer.TokenInterval(
                          start_index=4, end_index=6
                      ),
                      char_interval=data.CharInterval(start_pos=27, end_pos=36),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="reason",
                      extraction_text="pain",
                      token_interval=tokenizer.TokenInterval(
                          start_index=7, end_index=8
                      ),
                      char_interval=data.CharInterval(start_pos=41, end_pos=45),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                      token_interval=tokenizer.TokenInterval(
                          start_index=9, end_index=10
                      ),
                      char_interval=data.CharInterval(start_pos=50, end_pos=60),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="duration",
                      extraction_text="for one month",
                      token_interval=tokenizer.TokenInterval(
                          start_index=10, end_index=13
                      ),
                      char_interval=data.CharInterval(start_pos=61, end_pos=74),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "extraction_with_tokenizing_pipe_delimiter",
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Napro | syn",
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                  )
              ],
          ],
          "Patient is prescribed Napro | syn and prednisone.",
          [
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="Napro | syn",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=6
                      ),
                      char_interval=data.CharInterval(start_pos=22, end_pos=33),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="prednisone",
                      token_interval=tokenizer.TokenInterval(
                          start_index=7, end_index=8
                      ),
                      char_interval=data.CharInterval(start_pos=38, end_pos=48),
                      alignment_status=data.AlignmentStatus.MATCH_EXACT,
                  )
              ],
          ],
      ),
      (
          "test_only_matching_end_does_not_align",
          [
              [
                  data.Extraction(
                      extraction_class="some_class",
                      extraction_text="only matched end",
                  )
              ],
          ],
          "end",
          [[
              data.Extraction(
                  extraction_class="some_class",
                  extraction_text="only matched end",
                  char_interval=None,
                  alignment_status=None,
              )
          ]],
      ),
      dict(
          testcase_name="fuzzy_alignment_success",
          # Tests fuzzy alignment alongside exact matching. Shows different alignment statuses:
          # "heart problems" gets fuzzy match, "severe heart problems complications" gets lesser match.
          # Demonstrates both fuzzy and lesser matching working with 75% threshold.
          extractions=[
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="heart problems",
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="severe heart problems complications",
                  )
              ],
          ],
          source_text="Patient has severe heart problems today.",
          expected_output=[
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="heart problems",
                      token_interval=tokenizer.TokenInterval(
                          start_index=3, end_index=5
                      ),
                      char_interval=data.CharInterval(start_pos=19, end_pos=33),
                      alignment_status=data.AlignmentStatus.MATCH_FUZZY,
                  )
              ],
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="severe heart problems complications",
                      token_interval=tokenizer.TokenInterval(
                          start_index=2, end_index=5
                      ),
                      char_interval=data.CharInterval(start_pos=12, end_pos=33),
                      alignment_status=data.AlignmentStatus.MATCH_LESSER,
                  )
              ],
          ],
          enable_fuzzy_alignment=True,
      ),
      dict(
          testcase_name="fuzzy_alignment_below_threshold",
          # Tests fuzzy alignment failure when overlap ratio < _FUZZY_ALIGNMENT_MIN_THRESHOLD (75%).
          # No tokens overlap between "completely different medicine" and "Patient takes aspirin daily."
          extractions=[
              [
                  data.Extraction(
                      extraction_class="medication",
                      extraction_text="completely different medicine",
                  )
              ],
          ],
          source_text="Patient takes aspirin daily.",
          expected_output=[[
              data.Extraction(
                  extraction_class="medication",
                  extraction_text="completely different medicine",
                  char_interval=None,
                  alignment_status=None,
              )
          ]],
          enable_fuzzy_alignment=True,
      ),
      dict(
          testcase_name="accept_match_lesser_disabled",
          # Tests accept_match_lesser=False with fuzzy fallback.
          extractions=[
              [
                  data.Extraction(
                      extraction_class="condition",
                      extraction_text="patient heart problems today",
                  )
              ],
          ],
          source_text="Patient has heart problems today.",
          expected_output=[[
              data.Extraction(
                  extraction_class="condition",
                  extraction_text="patient heart problems today",
                  token_interval=tokenizer.TokenInterval(
                      start_index=0, end_index=5
                  ),
                  char_interval=data.CharInterval(start_pos=0, end_pos=32),
                  alignment_status=data.AlignmentStatus.MATCH_FUZZY,
              )
          ]],
          enable_fuzzy_alignment=True,
          accept_match_lesser=False,
      ),
      dict(
          testcase_name="fuzzy_alignment_subset_window",
          # Extraction is a subset of a longer source clause; ensures extra tokens do not penalise score.
          extractions=[[
              data.Extraction(
                  extraction_class="tendon",
                  extraction_text="The iliopsoas tendon is intact",
              )
          ]],
          source_text=(
              "The iliopsoas and proximal hamstring tendons are intact."
          ),
          expected_output=[[
              data.Extraction(
                  extraction_class="tendon",
                  extraction_text="The iliopsoas tendon is intact",
                  token_interval=tokenizer.TokenInterval(
                      start_index=0, end_index=8
                  ),
                  char_interval=data.CharInterval(start_pos=0, end_pos=55),
                  alignment_status=data.AlignmentStatus.MATCH_FUZZY,
              )
          ]],
          enable_fuzzy_alignment=True,
          accept_match_lesser=False,
      ),
      dict(
          testcase_name="fuzzy_alignment_with_reordered_words",
          # Tests fuzzy alignment's ability to handle reordered words in the extraction.
          extractions=[[
              data.Extraction(
                  extraction_class="condition",
                  extraction_text="problems heart",  # Reordered words
                  char_interval=data.CharInterval(start_pos=12, end_pos=33),
                  alignment_status=data.AlignmentStatus.MATCH_FUZZY,
              )
          ]],
          source_text="Patient has severe heart problems today.",
          expected_output=[[
              data.Extraction(
                  extraction_class="condition",
                  extraction_text="problems heart",
                  # The best matching window in the source is "severe heart problems"
                  token_interval=tokenizer.TokenInterval(
                      start_index=2, end_index=5
                  ),
                  char_interval=data.CharInterval(start_pos=12, end_pos=33),
                  alignment_status=data.AlignmentStatus.MATCH_FUZZY,
              )
          ]],
          enable_fuzzy_alignment=True,
      ),
      dict(
          testcase_name="fuzzy_alignment_fails_low_ratio",
          # An extraction that partially overlaps but is below the fuzzy threshold should not be aligned.
          extractions=[[
              data.Extraction(
                  extraction_class="symptom",
                  extraction_text="headache and fever",
              )
          ]],
          source_text="Patient reports back pain and a fever.",
          expected_output=[[
              data.Extraction(
                  extraction_class="symptom",
                  extraction_text="headache and fever",
                  char_interval=None,
                  alignment_status=None,
              )
          ]],
          enable_fuzzy_alignment=True,
      ),
      dict(
          testcase_name="fuzzy_alignment_partial_overlap_success",
          # An extraction where the number of matched tokens divided by total extraction tokens
          # is >= the threshold (3/4 = 0.75).
          extractions=[[
              data.Extraction(
                  extraction_class="finding",
                  extraction_text="mild degenerative disc disease",
              )
          ]],
          source_text=(
              "Findings consistent with degenerative disc disease at L5-S1."
          ),
          expected_output=[[
              data.Extraction(
                  extraction_class="finding",
                  extraction_text="mild degenerative disc disease",
                  # The best window found is "degenerative disc disease"
                  token_interval=tokenizer.TokenInterval(
                      start_index=3, end_index=6
                  ),
                  char_interval=data.CharInterval(start_pos=20, end_pos=50),
                  alignment_status=data.AlignmentStatus.MATCH_FUZZY,
              )
          ]],
          enable_fuzzy_alignment=True,
      ),
  )
  def test_extraction_alignment(
      self,
      extractions: Sequence[Sequence[data.Extraction]],
      source_text: str,
      expected_output: Sequence[Sequence[data.Extraction]] | ValueError,
      enable_fuzzy_alignment: bool = False,
      accept_match_lesser: bool = True,
  ):
    if expected_output is ValueError:
      with self.assertRaises(ValueError):
        self.aligner.align_extractions(
            extractions, source_text, enable_fuzzy_alignment=False
        )
    else:
      aligned_extraction_groups = self.aligner.align_extractions(
          extractions,
          source_text,
          enable_fuzzy_alignment=enable_fuzzy_alignment,
          accept_match_lesser=accept_match_lesser,
      )
      flattened_extractions = []
      for group in aligned_extraction_groups:
        flattened_extractions.extend(group)
      assert_char_interval_match_source(
          self, source_text, flattened_extractions
      )
      self.assertEqual(aligned_extraction_groups, expected_output)


class ResolverTest(parameterized.TestCase):
  _TWO_MEDICATIONS_JSON_UNDELIMITED = textwrap.dedent(f"""\
      {{
        "{schema.EXTRACTIONS_KEY}": [
          {{
            "medication": "Naprosyn",
            "medication_index": 4,
            "frequency": "as needed",
            "frequency_index": 5,
            "reason": "pain",
            "reason_index": 8
          }},
          {{
            "medication": "prednisone",
            "medication_index": 9,
            "duration": "for one month",
            "duration_index": 10
          }}
        ]
      }}""")

  _TWO_MEDICATIONS_YAML_UNDELIMITED = textwrap.dedent(f"""\
  {schema.EXTRACTIONS_KEY}:
    - medication: "Naprosyn"
      medication_index: 4
      frequency: "as needed"
      frequency_index: 5
      reason: "pain"
      reason_index: 8

    - medication: "prednisone"
      medication_index: 9
      duration: "for one month"
      duration_index: 10
  """)

  _EXPECTED_TWO_MEDICATIONS_ANNOTATED = [
      data.Extraction(
          extraction_class="medication",
          extraction_text="Naprosyn",
          extraction_index=4,
          group_index=0,
      ),
      data.Extraction(
          extraction_class="frequency",
          extraction_text="as needed",
          extraction_index=5,
          group_index=0,
      ),
      data.Extraction(
          extraction_class="reason",
          extraction_text="pain",
          extraction_index=8,
          group_index=0,
      ),
      data.Extraction(
          extraction_class="medication",
          extraction_text="prednisone",
          extraction_index=9,
          group_index=1,
      ),
      data.Extraction(
          extraction_class="duration",
          extraction_text="for one month",
          extraction_index=10,
          group_index=1,
      ),
  ]

  def setUp(self):
    super().setUp()
    self.default_resolver = resolver_lib.Resolver(
        format_type=data.FormatType.JSON,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="json_with_fence",
          resolver=resolver_lib.Resolver(
              fence_output=True,
              format_type=data.FormatType.JSON,
          ),
          input_text=textwrap.dedent(f"""\
            ```json
            {{
              "{schema.EXTRACTIONS_KEY}": [
                {{
                  "medication": "Naprosyn",
                  "medication_index": 4,
                  "frequency": "as needed",
                  "frequency_index": 5,
                  "reason": "pain",
                  "reason_index": 8
                }},
                {{
                  "medication": "prednisone",
                  "medication_index": 9,
                  "duration": "for one month",
                  "duration_index": 10
                }}
              ]
            }}
            ```"""),
          expected_output=_EXPECTED_TWO_MEDICATIONS_ANNOTATED,
      ),
      dict(
          testcase_name="yaml_with_fence",
          resolver=resolver_lib.Resolver(
              fence_output=True,
              format_type=data.FormatType.YAML,
          ),
          input_text=textwrap.dedent(f"""\
            ```yaml
            {schema.EXTRACTIONS_KEY}:
              - medication: "Naprosyn"
                medication_index: 4
                frequency: "as needed"
                frequency_index: 5
                reason: "pain"
                reason_index: 8

              - medication: "prednisone"
                medication_index: 9
                duration: "for one month"
                duration_index: 10
            ```"""),
          expected_output=_EXPECTED_TWO_MEDICATIONS_ANNOTATED,
      ),
      dict(
          testcase_name="json_no_fence",
          resolver=resolver_lib.Resolver(
              fence_output=False,
              format_type=data.FormatType.JSON,
          ),
          input_text=_TWO_MEDICATIONS_JSON_UNDELIMITED,
          expected_output=_EXPECTED_TWO_MEDICATIONS_ANNOTATED,
      ),
      dict(
          testcase_name="yaml_no_fence",
          resolver=resolver_lib.Resolver(
              fence_output=False,
              format_type=data.FormatType.YAML,
          ),
          input_text=_TWO_MEDICATIONS_YAML_UNDELIMITED,
          expected_output=_EXPECTED_TWO_MEDICATIONS_ANNOTATED,
      ),
  )
  def test_resolve_valid_inputs(self, resolver, input_text, expected_output):
    actual_extractions = resolver.resolve(input_text)
    self.assertCountEqual(expected_output, actual_extractions)
    assert_char_interval_match_source(self, input_text, actual_extractions)

  def test_handle_integer_extraction(self):
    test_input = textwrap.dedent(f"""\
    ```json
    {{
      "{schema.EXTRACTIONS_KEY}": [
        {{
          "year": 2006,
          "year_index": 6
        }}
      ]
    }}
    ```""")
    expected_extractions = [
        data.Extraction(
            extraction_class="year",
            extraction_text="2006",
            extraction_index=6,
            group_index=0,
        )
    ]

    actual_extractions = self.default_resolver.resolve(test_input)
    self.assertEqual(expected_extractions, list(actual_extractions))

  def test_resolve_empty_yaml(self):
    test_input = "```json\n```"
    actual = self.default_resolver.resolve(
        test_input, suppress_parse_errors=True
    )
    self.assertEmpty(actual)

  def test_resolve_empty_yaml_without_suppress_parse_errors(self):
    test_input = "```json\n```"
    with self.assertRaises(resolver_lib.ResolverParsingError):
      self.default_resolver.resolve(test_input, suppress_parse_errors=False)

  def test_align_with_valid_chunk(self):
    text = "This is a sample text with some extractions."
    tokenized_text = tokenizer.tokenize(text)

    chunk = tokenizer.TokenInterval(start_index=0, end_index=8)
    annotated_extractions = [
        data.Extraction(
            extraction_class="medication", extraction_text="sample"
        ),
        data.Extraction(
            extraction_class="condition", extraction_text="extractions"
        ),
    ]
    expected_extractions = [
        data.Extraction(
            extraction_class="medication",
            extraction_text="sample",
            token_interval=tokenizer.TokenInterval(start_index=3, end_index=4),
            char_interval=data.CharInterval(start_pos=10, end_pos=16),
            alignment_status=data.AlignmentStatus.MATCH_EXACT,
        ),
        data.Extraction(
            extraction_class="condition",
            extraction_text="extractions",
            token_interval=tokenizer.TokenInterval(start_index=7, end_index=8),
            char_interval=data.CharInterval(start_pos=32, end_pos=43),
            alignment_status=data.AlignmentStatus.MATCH_EXACT,
        ),
    ]

    chunk_text = chunking.get_token_interval_text(tokenized_text, chunk)
    token_offset = chunk.start_index
    aligned_extractions = list(
        self.default_resolver.align(
            extractions=annotated_extractions,
            source_text=chunk_text,
            token_offset=token_offset,
            char_offset=0,
            enable_fuzzy_alignment=False,
        )
    )

    self.assertEqual(len(aligned_extractions), len(expected_extractions))
    for expected, actual in zip(expected_extractions, aligned_extractions):
      self.assertDataclassEqual(expected, actual)
    assert_char_interval_match_source(self, text, aligned_extractions)

  def test_align_with_chunk_starting_in_middle(self):
    text = "This is a sample text with some extractions."
    tokenized_text = tokenizer.tokenize(text)

    chunk = tokenizer.TokenInterval(start_index=3, end_index=8)
    annotated_extractions = [
        data.Extraction(
            extraction_class="medication", extraction_text="sample"
        ),
        data.Extraction(
            extraction_class="condition", extraction_text="extractions"
        ),
    ]
    expected_extractions = [
        data.Extraction(
            extraction_class="medication",
            extraction_text="sample",
            token_interval=tokenizer.TokenInterval(start_index=3, end_index=4),
            char_interval=data.CharInterval(start_pos=10, end_pos=16),
            alignment_status=data.AlignmentStatus.MATCH_EXACT,
        ),
        data.Extraction(
            extraction_class="condition",
            extraction_text="extractions",
            token_interval=tokenizer.TokenInterval(start_index=7, end_index=8),
            char_interval=data.CharInterval(start_pos=32, end_pos=43),
            alignment_status=data.AlignmentStatus.MATCH_EXACT,
        ),
    ]

    chunk_text = chunking.get_token_interval_text(tokenized_text, chunk)
    token_offset = chunk.start_index
    # Compute global char offset from the token at chunk.start_index.
    char_offset = tokenized_text.tokens[
        chunk.start_index
    ].char_interval.start_pos
    aligned_extractions = list(
        self.default_resolver.align(
            extractions=annotated_extractions,
            source_text=chunk_text,
            token_offset=token_offset,
            char_offset=char_offset,
            enable_fuzzy_alignment=False,
        )
    )

    self.assertEqual(len(aligned_extractions), len(expected_extractions))
    for expected, actual in zip(expected_extractions, aligned_extractions):
      self.assertDataclassEqual(expected, actual)

    assert_char_interval_match_source(self, text, aligned_extractions)

  def test_align_with_no_extractions_in_chunk(self):
    tokenized_text = tokenizer.tokenize("No extractions here.")

    # Define a chunk that includes the entire text.
    chunk = tokenizer.TokenInterval()
    chunk.start_index = 0
    chunk.end_index = 3
    annotated_extractions = []

    chunk_text = chunking.get_token_interval_text(tokenized_text, chunk)
    token_offset = chunk.start_index
    aligned_extractions = list(
        self.default_resolver.align(
            extractions=annotated_extractions,
            source_text=chunk_text,
            token_offset=token_offset,
            char_offset=0,
            enable_fuzzy_alignment=False,
        )
    )

    self.assertEmpty(aligned_extractions)

  def test_align_successful(self):
    tokenized_text = tokenizer.TokenizedText(
        text="zero one two",
        tokens=[
            tokenizer.Token(
                token_type=tokenizer.TokenType.WORD,
                char_interval=tokenizer.CharInterval(start_pos=0, end_pos=4),
                index=0,
            ),
            tokenizer.Token(
                token_type=tokenizer.TokenType.WORD,
                char_interval=tokenizer.CharInterval(start_pos=5, end_pos=8),
                index=1,
            ),
            tokenizer.Token(
                token_type=tokenizer.TokenType.WORD,
                char_interval=tokenizer.CharInterval(start_pos=9, end_pos=12),
                index=2,
            ),
        ],
    )

    # Define a chunk that includes the entire text.
    chunk = tokenizer.TokenInterval(start_index=0, end_index=3)
    annotated_extractions = [
        data.Extraction(extraction_class="foo", extraction_text="zero"),
        data.Extraction(extraction_class="foo", extraction_text="one"),
    ]

    chunk_text = chunking.get_token_interval_text(tokenized_text, chunk)
    token_offset = chunk.start_index
    aligned_extractions = list(
        self.default_resolver.align(
            extractions=annotated_extractions,
            source_text=chunk_text,
            token_offset=token_offset,
            char_offset=0,
            enable_fuzzy_alignment=False,
        )
    )

    self.assertLen(aligned_extractions, 2)
    assert_char_interval_match_source(
        self, tokenized_text.text, aligned_extractions
    )

  def test_align_with_discontinuous_tokenized_text(self):
    tokenized_text = tokenizer.TokenizedText(
        text="zero one five",
        tokens=[
            tokenizer.Token(
                token_type=tokenizer.TokenType.WORD,
                char_interval=tokenizer.CharInterval(start_pos=0, end_pos=4),
                index=0,
            ),
            tokenizer.Token(
                token_type=tokenizer.TokenType.WORD,
                char_interval=tokenizer.CharInterval(start_pos=5, end_pos=8),
                index=1,
            ),
            tokenizer.Token(
                token_type=tokenizer.TokenType.WORD,
                char_interval=tokenizer.CharInterval(start_pos=9, end_pos=14),
                index=5,
            ),
        ],
    )

    # Define a chunk that includes too many tokens.
    chunk = tokenizer.TokenInterval(start_index=0, end_index=6)
    annotated_extractions = [
        data.Extraction(extraction_class="foo", extraction_text="zero"),
        data.Extraction(extraction_class="foo", extraction_text="one"),
    ]

    with self.assertRaises(tokenizer.InvalidTokenIntervalError):
      chunk_text = chunking.get_token_interval_text(tokenized_text, chunk)
      token_offset = chunk.start_index
      list(
          self.default_resolver.align(
              annotated_extractions,
              chunk_text,
              token_offset,
              enable_fuzzy_alignment=False,
          )
      )

  def test_align_with_discontinuous_tokenized_text_but_right_chunk(self):
    tokenized_text = tokenizer.TokenizedText(
        text="zero one five",
        tokens=[
            tokenizer.Token(
                token_type=tokenizer.TokenType.WORD,
                char_interval=tokenizer.CharInterval(start_pos=0, end_pos=4),
                index=0,
            ),
            tokenizer.Token(
                token_type=tokenizer.TokenType.WORD,
                char_interval=tokenizer.CharInterval(start_pos=5, end_pos=8),
                index=1,
            ),
            tokenizer.Token(
                token_type=tokenizer.TokenType.WORD,
                char_interval=tokenizer.CharInterval(start_pos=9, end_pos=14),
                index=5,
            ),
        ],
    )

    # Define a correct chunk.
    chunk = tokenizer.TokenInterval(start_index=0, end_index=3)
    annotated_extractions = [
        data.Extraction(extraction_class="foo", extraction_text="zero"),
        data.Extraction(extraction_class="foo", extraction_text="one"),
    ]

    chunk_text = chunking.get_token_interval_text(tokenized_text, chunk)
    token_offset = chunk.start_index
    aligned_extractions = list(
        self.default_resolver.align(
            extractions=annotated_extractions,
            source_text=chunk_text,
            token_offset=token_offset,
            char_offset=0,
            enable_fuzzy_alignment=False,
        )
    )
    self.assertLen(aligned_extractions, 2)
    assert_char_interval_match_source(
        self, tokenized_text.text, aligned_extractions
    )

  def test_align_with_empty_annotated_extractions(self):
    """Test align method with empty annotated_extractions sequence."""
    tokenized_text = tokenizer.tokenize("No extractions here.")

    # Define a chunk that includes the entire text.
    chunk = tokenizer.TokenInterval()
    chunk.start_index = 0
    chunk.end_index = 3
    annotated_extractions = []  # Empty sequence representing no extractions

    chunk_text = chunking.get_token_interval_text(tokenized_text, chunk)
    token_offset = chunk.start_index
    aligned_extractions = list(
        self.default_resolver.align(
            extractions=annotated_extractions,
            source_text=chunk_text,
            token_offset=token_offset,
            char_offset=0,
            enable_fuzzy_alignment=False,
        )
    )

    self.assertEmpty(aligned_extractions)


if __name__ == "__main__":
  absltest.main()
