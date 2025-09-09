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

from collections.abc import Sequence
import dataclasses
import textwrap
from typing import Type
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract import annotation
from langextract import prompting
from langextract import resolver as resolver_lib
from langextract import schema
from langextract.core import data
from langextract.core import tokenizer
from langextract.core import types
from langextract.providers import gemini


class AnnotatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_language_model = self.enter_context(
        mock.patch.object(gemini, "GeminiLanguageModel", autospec=True)
    )
    self.annotator = annotation.Annotator(
        language_model=self.mock_language_model,
        prompt_template=prompting.PromptTemplateStructured(description=""),
    )

  def assert_char_interval_match_source(
      self, source_text: str, extractions: Sequence[data.Extraction]
  ):
    """Case-insensitive assertion that char_interval matches source text.

    For each extraction, this function extracts the substring from the source
    text using the extraction's char_interval and asserts that it matches the
    extraction's text. Note the Alignment process between tokens is also
    case-insensitive.

    Args:
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
        self.assertIsNotNone(start, "start_pos should not be None")
        self.assertIsNotNone(end, "end_pos should not be None")
        extracted = source_text[start:end]
        self.assertEqual(
            extracted.lower(),
            extraction.extraction_text.lower(),
            f"Extraction '{extraction.extraction_text}' does not match"
            f" extracted '{extracted}' using char_interval {char_int}",
        )

  def test_annotate_text_single_chunk(self):
    text = (
        "Patient Jane Doe, ID 67890, received 10mg of Lisinopril daily for"
        " hypertension diagnosed on 2023-03-15."
    )
    self.mock_language_model.infer.return_value = [[
        types.ScoredOutput(
            score=1.0,
            output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - patient: "Jane Doe"
                patient_index: 1
                patient_id: "67890"
                patient_id_index: 4
                dosage: "10mg"
                dosage_index: 6
                medication: "Lisinopril"
                medication_index: 8
                frequency: "daily"
                frequency_index: 9
                condition: "hypertension"
                condition_index: 11
                diagnosis_date: "2023-03-15"
                diagnosis_date_index: 13
              ```"""),
        )
    ]]
    resolver = resolver_lib.Resolver(format_type=data.FormatType.YAML)
    expected_annotated_text = data.AnnotatedDocument(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class="patient",
                extraction_index=1,
                extraction_text="Jane Doe",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=1, end_index=3
                ),
                char_interval=data.CharInterval(start_pos=8, end_pos=16),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="patient_id",
                extraction_index=4,
                extraction_text="67890",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=5, end_index=6
                ),
                char_interval=data.CharInterval(start_pos=21, end_pos=26),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="dosage",
                extraction_index=6,
                extraction_text="10mg",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=8, end_index=10
                ),
                char_interval=data.CharInterval(start_pos=37, end_pos=41),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="medication",
                extraction_index=8,
                extraction_text="Lisinopril",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=11, end_index=12
                ),
                char_interval=data.CharInterval(start_pos=45, end_pos=55),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="frequency",
                extraction_index=9,
                extraction_text="daily",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=12, end_index=13
                ),
                char_interval=data.CharInterval(start_pos=56, end_pos=61),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="condition",
                extraction_index=11,
                extraction_text="hypertension",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=14, end_index=15
                ),
                char_interval=data.CharInterval(start_pos=66, end_pos=78),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="diagnosis_date",
                extraction_index=13,
                extraction_text="2023-03-15",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=17, end_index=22
                ),
                char_interval=data.CharInterval(start_pos=92, end_pos=102),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
        ],
    )

    actual_annotated_text = self.annotator.annotate_text(
        text, resolver=resolver
    )
    self.assertDataclassEqual(expected_annotated_text, actual_annotated_text)
    self.assert_char_interval_match_source(
        text, actual_annotated_text.extractions
    )
    self.mock_language_model.infer.assert_called_once_with(
        batch_prompts=[f"\n\nQ: {text}\nA: "],
    )

  def test_annotate_text_without_index_suffix(self):
    text = (
        "Patient Jane Doe, ID 67890, received 10mg of Lisinopril daily for"
        " hypertension diagnosed on 2023-03-15."
    )
    self.mock_language_model.infer.return_value = [[
        types.ScoredOutput(
            score=1.0,
            output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - patient: "Jane Doe"
                patient_id: "67890"
                dosage: "10mg"
                medication: "Lisinopril"
                frequency: "daily"
                condition: "hypertension"
                diagnosis_date: "2023-03-15"
              ```"""),
        )
    ]]
    resolver = resolver_lib.Resolver(
        format_type=data.FormatType.YAML,
        extraction_index_suffix=None,
    )
    expected_annotated_text = data.AnnotatedDocument(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class="patient",
                extraction_index=1,
                extraction_text="Jane Doe",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=1, end_index=3
                ),
                char_interval=data.CharInterval(start_pos=8, end_pos=16),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="patient_id",
                extraction_index=2,
                extraction_text="67890",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=5, end_index=6
                ),
                char_interval=data.CharInterval(start_pos=21, end_pos=26),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="dosage",
                extraction_index=3,
                extraction_text="10mg",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=8, end_index=10
                ),
                char_interval=data.CharInterval(start_pos=37, end_pos=41),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="medication",
                extraction_index=4,
                extraction_text="Lisinopril",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=11, end_index=12
                ),
                char_interval=data.CharInterval(start_pos=45, end_pos=55),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="frequency",
                extraction_index=5,
                extraction_text="daily",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=12, end_index=13
                ),
                char_interval=data.CharInterval(start_pos=56, end_pos=61),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="condition",
                extraction_index=6,
                extraction_text="hypertension",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=14, end_index=15
                ),
                char_interval=data.CharInterval(start_pos=66, end_pos=78),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="diagnosis_date",
                extraction_index=7,
                extraction_text="2023-03-15",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=17, end_index=22
                ),
                char_interval=data.CharInterval(start_pos=92, end_pos=102),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
        ],
    )

    actual_annotated_text = self.annotator.annotate_text(
        text, resolver=resolver
    )
    self.assertDataclassEqual(expected_annotated_text, actual_annotated_text)
    self.assert_char_interval_match_source(
        text, actual_annotated_text.extractions
    )
    self.mock_language_model.infer.assert_called_once_with(
        batch_prompts=[f"\n\nQ: {text}\nA: "],
    )

  def test_annotate_text_with_attributes_suffix(self):
    text = (
        "Patient Jane Doe, ID 67890, received 10mg of Lisinopril daily for"
        " hypertension diagnosed on 2023-03-15."
    )
    self.mock_language_model.infer.return_value = [[
        types.ScoredOutput(
            score=1.0,
            output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - patient: "Jane Doe"
                patient_attributes:
                  status: "IDENTIFIABLE"
                patient_id: "67890"
                patient_id_attributes:
                  type: "UNIQUE_IDENTIFIER"
                dosage: "10mg"
                dosage_attributes:
                  frequency: "DAILY"
                medication: "Lisinopril"
                medication_attributes:
                  class: "ANTIHYPERTENSIVE"
                frequency: "daily"
                frequency_attributes:
                  time: "DAILY"
                condition: "hypertension"
                condition_attributes:
                  type: "CHRONIC"
                diagnosis_date: "2023-03-15"
                diagnosis_date_attributes:
                  status: "RELEVANT"
              ```"""),
        )
    ]]
    resolver = resolver_lib.Resolver(
        format_type=data.FormatType.YAML,
        extraction_index_suffix=None,
        extraction_attributes_suffix="_attributes",
    )
    expected_annotated_text = data.AnnotatedDocument(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class="patient",
                extraction_index=1,
                extraction_text="Jane Doe",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=1, end_index=3
                ),
                char_interval=data.CharInterval(start_pos=8, end_pos=16),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                attributes={
                    "status": "IDENTIFIABLE",
                },
            ),
            data.Extraction(
                extraction_class="patient_id",
                extraction_index=2,
                extraction_text="67890",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=5, end_index=6
                ),
                char_interval=data.CharInterval(start_pos=21, end_pos=26),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                attributes={"type": "UNIQUE_IDENTIFIER"},
            ),
            data.Extraction(
                extraction_class="dosage",
                extraction_index=3,
                extraction_text="10mg",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=8, end_index=10
                ),
                char_interval=data.CharInterval(start_pos=37, end_pos=41),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                attributes={"frequency": "DAILY"},
            ),
            data.Extraction(
                extraction_class="medication",
                extraction_index=4,
                extraction_text="Lisinopril",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=11, end_index=12
                ),
                char_interval=data.CharInterval(start_pos=45, end_pos=55),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                attributes={"class": "ANTIHYPERTENSIVE"},
            ),
            data.Extraction(
                extraction_class="frequency",
                extraction_index=5,
                extraction_text="daily",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=12, end_index=13
                ),
                char_interval=data.CharInterval(start_pos=56, end_pos=61),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                attributes={"time": "DAILY"},
            ),
            data.Extraction(
                extraction_class="condition",
                extraction_index=6,
                extraction_text="hypertension",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=14, end_index=15
                ),
                char_interval=data.CharInterval(start_pos=66, end_pos=78),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                attributes={"type": "CHRONIC"},
            ),
            data.Extraction(
                extraction_class="diagnosis_date",
                extraction_index=7,
                extraction_text="2023-03-15",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=17, end_index=22
                ),
                char_interval=data.CharInterval(start_pos=92, end_pos=102),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                attributes={"status": "RELEVANT"},
            ),
        ],
    )

    actual_annotated_text = self.annotator.annotate_text(
        text,
        resolver=resolver,
    )
    self.assertDataclassEqual(expected_annotated_text, actual_annotated_text)
    self.assert_char_interval_match_source(
        text, actual_annotated_text.extractions
    )
    self.mock_language_model.infer.assert_called_once_with(
        batch_prompts=[f"\n\nQ: {text}\nA: "],
    )

  def test_annotate_text_multiple_chunks(self):
    self.mock_language_model.infer.side_effect = [
        [[
            types.ScoredOutput(
                score=1.0,
                output=textwrap.dedent(f"""\
                  ```yaml
                  {schema.EXTRACTIONS_KEY}:
                  - medication: "Aspirin"
                    medication_index: 4
                    reason: "headache"
                    reason_index: 8
                  ```"""),
            )
        ]],
        [[
            types.ScoredOutput(
                score=1.0,
                output=textwrap.dedent(f"""\
                  ```yaml
                  {schema.EXTRACTIONS_KEY}:
                  - condition: "fever"
                    condition_index: 2
                  ```"""),
            )
        ]],
    ]

    # Simulating tokenization for text broken into two chunks:
    # Chunk 1: 'Patient takes one Aspirin for headaches.'
    # Chunk 2: 'Pt has fever.'
    text = "Patient takes one Aspirin for headaches. Pt has fever."

    # Indexes Aligned with Tokens
    # -------------------------------------------------------------------------
    # Index | 0        1     2    3        4    5         6  7    8    9     10
    # Token | Patient  takes one  Aspirin  for  headaches .  Pt   has  fever  .

    resolver = resolver_lib.Resolver(
        format_type=data.FormatType.YAML,
    )
    expected_annotated_text = data.AnnotatedDocument(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class="medication",
                extraction_index=4,
                extraction_text="Aspirin",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=3, end_index=4
                ),
                char_interval=data.CharInterval(start_pos=18, end_pos=25),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
            data.Extraction(
                extraction_class="reason",
                extraction_index=8,
                extraction_text="headache",
                group_index=0,
            ),
            data.Extraction(
                extraction_class="condition",
                extraction_index=2,
                extraction_text="fever",
                group_index=0,
                token_interval=tokenizer.TokenInterval(
                    start_index=9, end_index=10
                ),
                char_interval=data.CharInterval(start_pos=48, end_pos=53),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
            ),
        ],
    )

    actual_annotated_text = self.annotator.annotate_text(
        text,
        max_char_buffer=40,
        batch_length=1,
        resolver=resolver,
        enable_fuzzy_alignment=False,
    )
    self.assertDataclassEqual(expected_annotated_text, actual_annotated_text)
    self.assert_char_interval_match_source(
        text, actual_annotated_text.extractions
    )
    self.mock_language_model.infer.assert_has_calls([
        mock.call(
            batch_prompts=[
                "\n\nQ: Patient takes one Aspirin for headaches.\nA: "
            ],
            enable_fuzzy_alignment=False,
        ),
        mock.call(
            batch_prompts=["\n\nQ: Pt has fever.\nA: "],
            enable_fuzzy_alignment=False,
        ),
    ])

  def test_annotate_text_no_extractions(self):
    text = "Text without extractions."
    self.mock_language_model.infer.return_value = [[
        types.ScoredOutput(
            score=1.0,
            output=textwrap.dedent(f"""\
            ```yaml
            {schema.EXTRACTIONS_KEY}: []
            ```"""),
        )
    ]]
    resolver = resolver_lib.Resolver(
        format_type=data.FormatType.YAML,
    )
    expected_annotated_text = data.AnnotatedDocument(text=text, extractions=[])

    actual_annotated_text = self.annotator.annotate_text(
        text, resolver=resolver
    )
    self.assertDataclassEqual(expected_annotated_text, actual_annotated_text)
    self.mock_language_model.infer.assert_called_once_with(
        batch_prompts=[f"\n\nQ: {text}\nA: "],
    )


class AnnotatorMultipleDocumentTest(parameterized.TestCase):

  _FIXED_DOCUMENT_CONTENT = "Patient reports migraine."

  _LLM_INFERENCE = textwrap.dedent(f"""\
    ```yaml
    {schema.EXTRACTIONS_KEY}:
    - PATIENT: "Patient"
      PATIENT_index: 0
    - SYMPTOM: "migraine"
      SYMPTOM_index: 2
    ```""")

  _ANNOTATED_DOCUMENT = data.AnnotatedDocument(
      document_id="",
      extractions=[
          data.Extraction(
              extraction_class="PATIENT",
              extraction_text="Patient",
              token_interval=tokenizer.TokenInterval(
                  start_index=0, end_index=1
              ),
              char_interval=data.CharInterval(start_pos=0, end_pos=7),
              alignment_status=data.AlignmentStatus.MATCH_EXACT,
              extraction_index=0,
              group_index=0,
          ),
          data.Extraction(
              extraction_class="SYMPTOM",
              extraction_text="migraine",
              token_interval=tokenizer.TokenInterval(
                  start_index=2, end_index=3
              ),
              char_interval=data.CharInterval(start_pos=16, end_pos=24),
              alignment_status=data.AlignmentStatus.MATCH_EXACT,
              extraction_index=2,
              group_index=1,
          ),
      ],
      text="Patient reports migraine.",
  )

  @parameterized.named_parameters(
      dict(
          testcase_name="single_document",
          documents=[
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc1"},
          ],
          expected_result=[
              dataclasses.replace(
                  _ANNOTATED_DOCUMENT,
                  document_id="doc1",
              ),
          ],
      ),
      dict(
          testcase_name="multiple_documents",
          documents=[
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc1"},
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc2"},
          ],
          expected_result=[
              dataclasses.replace(
                  _ANNOTATED_DOCUMENT,
                  document_id="doc1",
              ),
              dataclasses.replace(
                  _ANNOTATED_DOCUMENT,
                  document_id="doc2",
              ),
          ],
      ),
      dict(
          testcase_name="zero_documents",
          documents=[],
          expected_result=[],
      ),
      dict(
          testcase_name="multiple_documents_same_batch",
          documents=[
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc1"},
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc2"},
          ],
          expected_result=[
              dataclasses.replace(
                  _ANNOTATED_DOCUMENT,
                  document_id="doc1",
              ),
              dataclasses.replace(
                  _ANNOTATED_DOCUMENT,
                  document_id="doc2",
              ),
          ],
          batch_length=10,
      ),
  )
  def test_annotate_documents(
      self,
      documents: Sequence[dict[str, str]],
      expected_result: Sequence[data.AnnotatedDocument],
      batch_length: int = 1,
  ):
    mock_language_model = self.enter_context(
        mock.patch.object(gemini, "GeminiLanguageModel", autospec=True)
    )

    # Define a side effect function so return length based on batch length.
    def mock_infer_side_effect(batch_prompts, **kwargs):
      for _ in batch_prompts:
        yield [
            types.ScoredOutput(
                score=1.0,
                output=self._LLM_INFERENCE,
            )
        ]

    mock_language_model.infer.side_effect = mock_infer_side_effect

    annotator = annotation.Annotator(
        language_model=mock_language_model,
        prompt_template=prompting.PromptTemplateStructured(description=""),
    )

    document_objects = [
        data.Document(
            text=doc["text"],
            document_id=doc["document_id"],
        )
        for doc in documents
    ]
    actual_annotations = list(
        annotator.annotate_documents(
            document_objects,
            resolver=resolver_lib.Resolver(
                fence_output=True, format_type=data.FormatType.YAML
            ),
            max_char_buffer=200,
            batch_length=batch_length,
            debug=False,
        )
    )

    self.assertLen(actual_annotations, len(expected_result))
    for actual_annotation, expected_annotation in zip(
        actual_annotations, expected_result
    ):
      self.assertDataclassEqual(expected_annotation, actual_annotation)

    self.assertGreaterEqual(mock_language_model.infer.call_count, 0)

  @parameterized.named_parameters(
      dict(
          testcase_name="same_document_id_contiguous",
          documents=[
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc1"},
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc1"},
          ],
          expected_exception=annotation.DocumentRepeatError,
      ),
      dict(
          testcase_name="same_document_id_separated",
          documents=[
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc1"},
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc2"},
              {"text": _FIXED_DOCUMENT_CONTENT, "document_id": "doc1"},
          ],
          expected_exception=annotation.DocumentRepeatError,
      ),
  )
  def test_annotate_documents_exceptions(
      self,
      documents: Sequence[dict[str, str]],
      expected_exception: Type[annotation.DocumentRepeatError],
      batch_length: int = 1,
  ):
    mock_language_model = self.enter_context(
        mock.patch.object(gemini, "GeminiLanguageModel", autospec=True)
    )
    mock_language_model.infer.return_value = [
        [
            types.ScoredOutput(
                score=1.0,
                output=self._LLM_INFERENCE,
            )
        ]
    ]
    annotator = annotation.Annotator(
        language_model=mock_language_model,
        prompt_template=prompting.PromptTemplateStructured(description=""),
    )

    document_objects = [
        data.Document(text=doc["text"], document_id=doc["document_id"])
        for doc in documents
    ]

    with self.assertRaises(expected_exception):
      list(
          annotator.annotate_documents(
              document_objects,
              max_char_buffer=200,
              batch_length=batch_length,
              debug=False,
          )
      )


class AnnotatorMultiPassTest(absltest.TestCase):
  """Tests for multi-pass extraction functionality."""

  def setUp(self):
    super().setUp()
    self.mock_language_model = self.enter_context(
        mock.patch.object(gemini, "GeminiLanguageModel", autospec=True)
    )
    self.annotator = annotation.Annotator(
        language_model=self.mock_language_model,
        prompt_template=prompting.PromptTemplateStructured(description=""),
    )

  def test_multipass_extraction_non_overlapping(self):
    """Test multi-pass extraction with non-overlapping extractions."""
    text = "Patient John Smith has diabetes and takes insulin daily."

    self.mock_language_model.infer.side_effect = [
        [[
            types.ScoredOutput(
                score=1.0,
                output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - patient: "John Smith"
                patient_index: 1
              - condition: "diabetes"
                condition_index: 4
              ```"""),
            )
        ]],
        [[
            types.ScoredOutput(
                score=1.0,
                output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - medication: "insulin"
                medication_index: 7
              - frequency: "daily"
                frequency_index: 8
              ```"""),
            )
        ]],
    ]

    resolver = resolver_lib.Resolver(format_type=data.FormatType.YAML)

    result = self.annotator.annotate_text(
        text, resolver=resolver, extraction_passes=2, debug=False
    )

    self.assertLen(result.extractions, 4)
    extraction_classes = [e.extraction_class for e in result.extractions]
    self.assertCountEqual(
        extraction_classes, ["patient", "condition", "medication", "frequency"]
    )

    self.assertEqual(self.mock_language_model.infer.call_count, 2)

  def test_multipass_extraction_overlapping(self):
    """Test multi-pass extraction with overlapping extractions (first pass wins)."""
    text = "Dr. Smith prescribed aspirin."

    # Mock overlapping extractions - both passes find "Smith" but differently
    self.mock_language_model.infer.side_effect = [
        [[
            types.ScoredOutput(
                score=1.0,
                output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - doctor: "Dr. Smith"
                doctor_index: 0
              ```"""),
            )
        ]],
        [[
            types.ScoredOutput(
                score=1.0,
                output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - patient: "Smith"
                patient_index: 1
              - medication: "aspirin"
                medication_index: 2
              ```"""),
            )
        ]],
    ]

    resolver = resolver_lib.Resolver(format_type=data.FormatType.YAML)

    result = self.annotator.annotate_text(
        text, resolver=resolver, extraction_passes=2, debug=False
    )

    self.assertLen(result.extractions, 2)
    extraction_classes = [e.extraction_class for e in result.extractions]
    self.assertCountEqual(extraction_classes, ["doctor", "medication"])

    # Verify "Dr. Smith" from first pass is kept, not "Smith" from second pass
    doctor_extraction = next(
        e for e in result.extractions if e.extraction_class == "doctor"
    )
    self.assertEqual(doctor_extraction.extraction_text, "Dr. Smith")

  def test_multipass_extraction_single_pass(self):
    """Test that extraction_passes=1 behaves like normal single-pass extraction."""
    text = "Patient has fever."

    self.mock_language_model.infer.return_value = [[
        types.ScoredOutput(
            score=1.0,
            output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - patient: "Patient"
                patient_index: 0
              - condition: "fever"
                condition_index: 2
              ```"""),
        )
    ]]

    resolver = resolver_lib.Resolver(format_type=data.FormatType.YAML)

    result = self.annotator.annotate_text(
        text, resolver=resolver, extraction_passes=1, debug=False  # Single pass
    )

    self.assertLen(result.extractions, 2)
    self.assertEqual(self.mock_language_model.infer.call_count, 1)

  def test_multipass_extraction_empty_passes(self):
    """Test multi-pass extraction when some passes return no extractions."""
    text = "Test text."

    self.mock_language_model.infer.side_effect = [
        [[
            types.ScoredOutput(
                score=1.0,
                output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - test: "Test"
                test_index: 0
              ```"""),
            )
        ]],
        [[
            types.ScoredOutput(
                score=1.0,
                output=textwrap.dedent(f"""\
              ```yaml
              {schema.EXTRACTIONS_KEY}: []
              ```"""),
            )
        ]],
    ]

    resolver = resolver_lib.Resolver(format_type=data.FormatType.YAML)

    result = self.annotator.annotate_text(
        text, resolver=resolver, extraction_passes=2, debug=False
    )

    self.assertLen(result.extractions, 1)
    self.assertEqual(result.extractions[0].extraction_class, "test")


class MultiPassHelperFunctionsTest(parameterized.TestCase):
  """Tests for multi-pass helper functions."""

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_list",
          all_extractions=[],
          expected_count=0,
          expected_classes=[],
      ),
      dict(
          testcase_name="single_pass",
          all_extractions=[[
              data.Extraction(
                  "class1", "text1", char_interval=data.CharInterval(0, 5)
              ),
              data.Extraction(
                  "class2", "text2", char_interval=data.CharInterval(10, 15)
              ),
          ]],
          expected_count=2,
          expected_classes=["class1", "class2"],
      ),
      dict(
          testcase_name="non_overlapping_passes",
          all_extractions=[
              [
                  data.Extraction(
                      "class1", "text1", char_interval=data.CharInterval(0, 5)
                  )
              ],
              [
                  data.Extraction(
                      "class2", "text2", char_interval=data.CharInterval(10, 15)
                  )
              ],
          ],
          expected_count=2,
          expected_classes=["class1", "class2"],
      ),
      dict(
          testcase_name="overlapping_passes_first_wins",
          all_extractions=[
              [
                  data.Extraction(
                      "class1", "text1", char_interval=data.CharInterval(0, 10)
                  )
              ],
              [
                  data.Extraction(
                      "class2", "text2", char_interval=data.CharInterval(5, 15)
                  ),  # Overlaps
                  data.Extraction(
                      "class3", "text3", char_interval=data.CharInterval(20, 25)
                  ),  # No overlap
              ],
          ],
          expected_count=2,
          expected_classes=[
              "class1",
              "class3",
          ],  # class2 excluded due to overlap
      ),
  )
  def test_merge_non_overlapping_extractions(
      self, all_extractions, expected_count, expected_classes
  ):
    """Test merging extractions from multiple passes."""
    result = annotation._merge_non_overlapping_extractions(all_extractions)

    self.assertLen(result, expected_count)
    if expected_classes:
      extraction_classes = [e.extraction_class for e in result]
      self.assertCountEqual(extraction_classes, expected_classes)

  @parameterized.named_parameters(
      dict(
          testcase_name="overlapping_intervals",
          ext1=data.Extraction(
              "class1", "text1", char_interval=data.CharInterval(0, 10)
          ),
          ext2=data.Extraction(
              "class2", "text2", char_interval=data.CharInterval(5, 15)
          ),
          expected=True,
      ),
      dict(
          testcase_name="non_overlapping_intervals",
          ext1=data.Extraction(
              "class1", "text1", char_interval=data.CharInterval(0, 5)
          ),
          ext2=data.Extraction(
              "class2", "text2", char_interval=data.CharInterval(10, 15)
          ),
          expected=False,
      ),
      dict(
          testcase_name="adjacent_intervals",
          ext1=data.Extraction(
              "class1", "text1", char_interval=data.CharInterval(0, 5)
          ),
          ext2=data.Extraction(
              "class2", "text2", char_interval=data.CharInterval(5, 10)
          ),
          expected=False,
      ),
      dict(
          testcase_name="none_interval_first",
          ext1=data.Extraction("class1", "text1", char_interval=None),
          ext2=data.Extraction(
              "class2", "text2", char_interval=data.CharInterval(5, 15)
          ),
          expected=False,
      ),
      dict(
          testcase_name="none_interval_second",
          ext1=data.Extraction(
              "class1", "text1", char_interval=data.CharInterval(0, 5)
          ),
          ext2=data.Extraction("class2", "text2", char_interval=None),
          expected=False,
      ),
      dict(
          testcase_name="both_none_intervals",
          ext1=data.Extraction("class1", "text1", char_interval=None),
          ext2=data.Extraction("class2", "text2", char_interval=None),
          expected=False,
      ),
  )
  def test_extractions_overlap(self, ext1, ext2, expected):
    """Test overlap detection between extractions."""
    result = annotation._extractions_overlap(ext1, ext2)
    self.assertEqual(result, expected)

  def test_batch_source_attribution_different_content(self):
    """Test batch processing correctly attributes unique extractions to source documents.

    This test verifies the fix for the source attribution bug where all
    extractions were incorrectly attributed to the first document in batch processing.
    Unlike existing tests that use identical content, this test uses different
    document content to ensure proper source attribution.
    """
    mock_language_model = self.enter_context(
        mock.patch.object(gemini, "GeminiLanguageModel", autospec=True)
    )

    # Define different mock responses for different documents
    def mock_infer_side_effect(batch_prompts, **kwargs):
      del kwargs  # Unused but required for signature
      responses = []
      for prompt in batch_prompts:
        # Create different responses based on document content
        if "patient John" in prompt:
          # First document gets entity_one extraction
          response_text = textwrap.dedent(f"""\
            ```yaml
            {schema.EXTRACTIONS_KEY}:
            - entity_one: "John"
              entity_one_index: 1
            ```""")
        elif "patient Mary" in prompt:
          # Second document gets entity_two extraction
          response_text = textwrap.dedent(f"""\
            ```yaml
            {schema.EXTRACTIONS_KEY}:
            - entity_two: "Mary"
              entity_two_index: 1
            ```""")
        else:
          # Fallback - no extractions
          response_text = textwrap.dedent(f"""\
            ```yaml
            {schema.EXTRACTIONS_KEY}: []
            ```""")

        responses.append(
            [
                types.ScoredOutput(
                    score=1.0,
                    output=response_text,
                )
            ]
        )
      return responses

    mock_language_model.infer.side_effect = mock_infer_side_effect

    annotator = annotation.Annotator(
        language_model=mock_language_model,
        prompt_template=prompting.PromptTemplateStructured(
            description="Extract entities"
        ),
    )

    # Create documents with different content and IDs
    documents = [
        data.Document(
            text="The patient John has diabetes.",
            document_id="doc_1",
        ),
        data.Document(
            text="The patient Mary has hypertension.",
            document_id="doc_2",
        ),
    ]

    # Process documents in batch (batch_length=10 ensures all in one batch)
    annotated_docs = list(
        annotator.annotate_documents(
            documents=documents,
            resolver=resolver_lib.Resolver(
                fence_output=True, format_type=data.FormatType.YAML
            ),
            batch_length=10,  # Process all documents in same batch
            debug=False,
        )
    )

    # Verify we get 2 annotated documents
    self.assertEqual(len(annotated_docs), 2)

    # Verify each document has unique extractions (no cross-contamination)
    doc_1 = next(doc for doc in annotated_docs if doc.document_id == "doc_1")
    doc_2 = next(doc for doc in annotated_docs if doc.document_id == "doc_2")

    # Check that each document has different extraction content
    self.assertNotEqual(
        doc_1.extractions,
        doc_2.extractions,
        "Doc 1 and Doc 2 should have different extractions",
    )

    # Verify each document has the correct extraction class
    if doc_1.extractions:
      self.assertEqual(
          doc_1.extractions[0].extraction_class,
          "entity_one",
          "Doc 1 should have entity_one extraction",
      )
    if doc_2.extractions:
      self.assertEqual(
          doc_2.extractions[0].extraction_class,
          "entity_two",
          "Doc 2 should have entity_two extraction",
      )

    # Verify document content is preserved correctly
    self.assertIn("patient John", doc_1.text)
    self.assertIn("patient Mary", doc_2.text)


if __name__ == "__main__":
  absltest.main()
