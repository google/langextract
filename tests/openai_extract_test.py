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

"""Tests for OpenAI integration with schema constraints in extract function."""

from unittest import mock

from absl.testing import absltest

from langextract import data
from langextract import inference
import langextract as lx


class TestOpenAIExtractIntegration(absltest.TestCase):

  @mock.patch("openai.OpenAI")
  def test_extract_with_openai_schema_constraints(self, mock_openai_class):
    # Mock the OpenAI client
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    # Mock response with properly formatted extraction
    mock_response = mock.Mock()
    mock_response.choices = [
        mock.Mock(
            message=mock.Mock(
                content=(
                    '{"extractions": [{"medication": "aspirin",'
                    ' "medication_attributes": {"dosage": "100mg", "frequency":'
                    ' "daily"}}]}'
                )
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # Prepare examples
    examples = [
        data.ExampleData(
            text="Take ibuprofen 200mg twice daily",
            extractions=[
                data.Extraction(
                    extraction_text="ibuprofen 200mg twice daily",
                    extraction_class="medication",
                    attributes={"dosage": "200mg", "frequency": "twice daily"},
                )
            ],
        )
    ]

    # Extract with OpenAI and schema constraints
    result = lx.extract(
        text_or_documents="Take aspirin 100mg daily",
        prompt_description="Extract medications with dosage and frequency",
        examples=examples,
        api_key="test-key",
        language_model_type=inference.OpenAILanguageModel,
        format_type=data.FormatType.JSON,
        use_schema_constraints=True,
        fence_output=False,
        model_id="gpt-4o-mini",
    )

    # Verify OpenAI was called with schema constraints
    call_args = mock_client.chat.completions.create.call_args[1]
    self.assertIn("response_format", call_args)
    self.assertEqual(call_args["response_format"]["type"], "json_schema")
    self.assertIn("json_schema", call_args["response_format"])

    # Verify the schema structure
    json_schema = call_args["response_format"]["json_schema"]
    self.assertEqual(json_schema["name"], "langextract_extraction")
    self.assertTrue(json_schema["strict"])

    # Check schema contains expected structure
    schema_props = json_schema["schema"]["properties"]
    self.assertIn("extractions", schema_props)

    # Verify extraction results
    self.assertIsNotNone(result)
    self.assertEqual(len(result.extractions), 1)
    self.assertEqual(result.extractions[0].extraction_class, "medication")
    self.assertEqual(result.extractions[0].extraction_text, "aspirin")
    self.assertEqual(result.extractions[0].attributes["dosage"], "100mg")
    self.assertEqual(result.extractions[0].attributes["frequency"], "daily")

  def test_extract_openai_yaml_with_schema_raises_error(self):
    """Test that YAML format with schema constraints raises an error."""
    examples = [
        data.ExampleData(
            text="Test",
            extractions=[
                data.Extraction(
                    extraction_text="test", extraction_class="entity"
                )
            ],
        )
    ]

    with self.assertRaises(ValueError) as context:
      lx.extract(
          text_or_documents="Test text",
          prompt_description="Extract entities",
          examples=examples,
          api_key="test-key",
          language_model_type=inference.OpenAILanguageModel,
          format_type=data.FormatType.YAML,
          use_schema_constraints=True,
          fence_output=False,
      )

    self.assertIn(
        "OpenAI schema constraints are only supported with FormatType.JSON",
        str(context.exception),
    )

  def test_extract_openai_fence_output_with_schema_raises_error(self):
    """Test that fence_output=True with schema constraints raises an error."""
    examples = [
        data.ExampleData(
            text="Test",
            extractions=[
                data.Extraction(
                    extraction_text="test", extraction_class="entity"
                )
            ],
        )
    ]

    with self.assertRaises(ValueError) as context:
      lx.extract(
          text_or_documents="Test text",
          prompt_description="Extract entities",
          examples=examples,
          api_key="test-key",
          language_model_type=inference.OpenAILanguageModel,
          format_type=data.FormatType.JSON,
          use_schema_constraints=True,
          fence_output=True,
      )

    self.assertIn(
        "OpenAI schema constraints cannot be used with fence_output=True",
        str(context.exception),
    )

  @mock.patch("openai.OpenAI")
  def test_extract_openai_without_schema_constraints(self, mock_openai_class):
    """Test OpenAI extraction without schema constraints works as before."""
    # Mock the OpenAI client
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    # Mock response
    mock_response = mock.Mock()
    mock_response.choices = [
        mock.Mock(
            message=mock.Mock(
                content='```json\n{"extractions": [{"entity": "test"}]}\n```'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # Prepare examples
    examples = [
        data.ExampleData(
            text="Example",
            extractions=[
                data.Extraction(
                    extraction_text="example", extraction_class="entity"
                )
            ],
        )
    ]

    # Extract without schema constraints
    result = lx.extract(
        text_or_documents="Test text",
        prompt_description="Extract entities",
        examples=examples,
        api_key="test-key",
        language_model_type=inference.OpenAILanguageModel,
        format_type=data.FormatType.JSON,
        use_schema_constraints=False,
        fence_output=True,
    )

    # Verify OpenAI was called without schema constraints
    call_args = mock_client.chat.completions.create.call_args[1]
    self.assertNotIn("response_format", call_args)

    # Verify system message is standard JSON format message
    messages = call_args["messages"]
    self.assertEqual(
        messages[0]["content"],
        "You are a helpful assistant that responds in JSON format.",
    )

    # Verify result is properly returned
    self.assertIsNotNone(result)
    self.assertEqual(len(result.extractions), 1)
    self.assertEqual(result.extractions[0].extraction_class, "entity")
    self.assertEqual(result.extractions[0].extraction_text, "test")


if __name__ == "__main__":
  absltest.main()
