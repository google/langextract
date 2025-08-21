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

"""Tests for the main package functions in __init__.py."""

import textwrap
from unittest import mock

from absl.testing import absltest

from langextract import prompting
import langextract as lx
from langextract.core import data
from langextract.core import types
from langextract.providers import schemas


class InitTest(absltest.TestCase):
  """Test cases for the main package functions."""

  @mock.patch.object(
      schemas.gemini.GeminiSchema, "from_examples", autospec=True
  )
  @mock.patch("langextract.extraction.factory.create_model")
  def test_lang_extract_as_lx_extract(
      self, mock_create_model, mock_gemini_schema
  ):

    input_text = "Patient takes Aspirin 100mg every morning."

    mock_model = mock.MagicMock()
    mock_model.infer.return_value = [[
        types.ScoredOutput(
            output=textwrap.dedent("""\
            ```json
            {
              "extractions": [
                {
                  "entity": "Aspirin",
                  "entity_attributes": {
                    "class": "medication"
                  }
                },
                {
                  "entity": "100mg",
                  "entity_attributes": {
                    "frequency": "every morning",
                    "class": "dosage"
                  }
                }
              ]
            }
            ```"""),
            score=0.9,
        )
    ]]

    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model

    mock_gemini_schema.return_value = None

    expected_result = data.AnnotatedDocument(
        document_id=None,
        extractions=[
            data.Extraction(
                extraction_class="entity",
                extraction_text="Aspirin",
                char_interval=data.CharInterval(start_pos=14, end_pos=21),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                extraction_index=1,
                group_index=0,
                description=None,
                attributes={"class": "medication"},
            ),
            data.Extraction(
                extraction_class="entity",
                extraction_text="100mg",
                char_interval=data.CharInterval(start_pos=22, end_pos=27),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                extraction_index=2,
                group_index=1,
                description=None,
                attributes={"frequency": "every morning", "class": "dosage"},
            ),
        ],
        text="Patient takes Aspirin 100mg every morning.",
    )

    mock_description = textwrap.dedent("""\
        Extract medication and dosage information in order of occurrence.
        """)

    mock_examples = [
        lx.data.ExampleData(
            text="Patient takes Tylenol 500mg daily.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Tylenol",
                    attributes={
                        "type": "analgesic",
                        "class": "medication",
                    },
                ),
            ],
        )
    ]
    mock_prompt_template = prompting.PromptTemplateStructured(
        description=mock_description, examples=mock_examples
    )

    prompt_generator = prompting.QAPromptGenerator(
        template=mock_prompt_template, format_type=lx.data.FormatType.JSON
    )

    actual_result = lx.extract(
        text_or_documents=input_text,
        prompt_description=mock_description,
        examples=mock_examples,
        api_key="some_api_key",
        fence_output=True,
        use_schema_constraints=False,
    )

    mock_gemini_schema.assert_not_called()
    mock_create_model.assert_called_once()
    mock_model.infer.assert_called_once_with(
        batch_prompts=[prompt_generator.render(input_text)],
        max_workers=10,  # Default value from extract()
    )

    self.assertDataclassEqual(expected_result, actual_result)

  @mock.patch.object(
      schemas.gemini.GeminiSchema, "from_examples", autospec=True
  )
  @mock.patch("langextract.extraction.factory.create_model")
  def test_extract_custom_params_reach_inference(
      self, mock_create_model, mock_gemini_schema
  ):
    """Sanity check that custom parameters reach the inference layer."""
    input_text = "Test text"

    mock_model = mock.MagicMock()
    mock_model.infer.return_value = [[
        types.ScoredOutput(
            output='```json\n{"extractions": []}\n```',
            score=0.9,
        )
    ]]

    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model
    mock_gemini_schema.return_value = None

    mock_examples = [
        lx.data.ExampleData(
            text="Example",
            extractions=[
                lx.data.Extraction(
                    extraction_class="test",
                    extraction_text="example",
                ),
            ],
        )
    ]

    lx.extract(
        text_or_documents=input_text,
        prompt_description="Test extraction",
        examples=mock_examples,
        api_key="test_key",
        max_workers=5,
        fence_output=True,
        use_schema_constraints=False,
    )

    mock_model.infer.assert_called_once()
    _, kwargs = mock_model.infer.call_args
    self.assertEqual(kwargs.get("max_workers"), 5)


if __name__ == "__main__":
  absltest.main()
