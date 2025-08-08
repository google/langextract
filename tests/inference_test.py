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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract import data
from langextract import inference
from langextract import schema


class TestOllamaLanguageModel(absltest.TestCase):

  @mock.patch.object(inference.OllamaLanguageModel, "_ollama_query")
  def test_ollama_infer(self, mock_ollama_query):

    # Actuall full gemma2 response using Ollama.
    gemma_response = {
        "model": "gemma2:latest",
        "created_at": "2025-01-23T22:37:08.579440841Z",
        "response": "{'bus' : '**autóbusz**'} \n\n\n  \n",
        "done": True,
        "done_reason": "stop",
        "context": [
            106,
            1645,
            108,
            1841,
            603,
            1986,
            575,
            59672,
            235336,
            107,
            108,
            106,
            2516,
            108,
            9766,
            6710,
            235281,
            865,
            664,
            688,
            7958,
            235360,
            6710,
            235306,
            688,
            12990,
            235248,
            110,
            139,
            108,
        ],
        "total_duration": 24038204381,
        "load_duration": 21551375738,
        "prompt_eval_count": 15,
        "prompt_eval_duration": 633000000,
        "eval_count": 17,
        "eval_duration": 1848000000,
    }
    mock_ollama_query.return_value = gemma_response
    model = inference.OllamaLanguageModel(
        model_id="gemma2:latest",
        model_url="http://localhost:11434",
        structured_output_format="json",
    )
    batch_prompts = ["What is bus in Hungarian?"]
    results = list(model.infer(batch_prompts))

    mock_ollama_query.assert_called_once_with(
        prompt="What is bus in Hungarian?",
        model="gemma2:latest",
        structured_output_format="json",
        model_url="http://localhost:11434",
    )
    expected_results = [[
        inference.ScoredOutput(
            score=1.0, output="{'bus' : '**autóbusz**'} \n\n\n  \n"
        )
    ]]
    self.assertEqual(results, expected_results)


class TestOpenAILanguageModelInference(parameterized.TestCase):

  @parameterized.named_parameters(
      ("without", "test-api-key", None, "gpt-4o-mini", 0.5),
      ("with", "test-api-key", "http://127.0.0.1:9001/v1", "gpt-4o-mini", 0.5),
  )
  @mock.patch("openai.OpenAI")
  def test_openai_infer_with_parameters(
      self, api_key, base_url, model_id, temperature, mock_openai_class
  ):
    # Mock the OpenAI client and chat completion response
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    # Mock response structure for v1.x API
    mock_response = mock.Mock()
    mock_response.choices = [
        mock.Mock(message=mock.Mock(content='{"name": "John", "age": 30}'))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # Create model instance
    model = inference.OpenAILanguageModel(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

    # Test inference
    batch_prompts = ["Extract name and age from: John is 30 years old"]
    results = list(model.infer(batch_prompts))

    # Verify API was called correctly
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that responds in JSON format."
                ),
            },
            {
                "role": "user",
                "content": "Extract name and age from: John is 30 years old",
            },
        ],
        temperature=temperature,
        max_tokens=None,
        top_p=None,
        n=1,
    )

    # Check results
    expected_results = [[
        inference.ScoredOutput(score=1.0, output='{"name": "John", "age": 30}')
    ]]
    self.assertEqual(results, expected_results)


class TestOpenAILanguageModel(absltest.TestCase):

  def test_openai_parse_output_json(self):
    model = inference.OpenAILanguageModel(
        api_key="test-key", format_type=data.FormatType.JSON
    )

    # Test valid JSON parsing
    output = '{"key": "value", "number": 42}'
    parsed = model.parse_output(output)
    self.assertEqual(parsed, {"key": "value", "number": 42})

    # Test invalid JSON
    with self.assertRaises(ValueError) as context:
      model.parse_output("invalid json")
    self.assertIn("Failed to parse output as JSON", str(context.exception))

  def test_openai_parse_output_yaml(self):
    model = inference.OpenAILanguageModel(
        api_key="test-key", format_type=data.FormatType.YAML
    )

    # Test valid YAML parsing
    output = "key: value\nnumber: 42"
    parsed = model.parse_output(output)
    self.assertEqual(parsed, {"key": "value", "number": 42})

    # Test invalid YAML
    with self.assertRaises(ValueError) as context:
      model.parse_output("invalid: yaml: bad")
    self.assertIn("Failed to parse output as YAML", str(context.exception))

  def test_openai_no_api_key_raises_error(self):
    with self.assertRaises(ValueError) as context:
      inference.OpenAILanguageModel(api_key=None)
    self.assertEqual(str(context.exception), "API key not provided.")

  @mock.patch("openai.OpenAI")
  def test_openai_temperature_zero(self, mock_openai_class):
    # Test that temperature=0.0 is properly passed through
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    mock_response = mock.Mock()
    mock_response.choices = [
        mock.Mock(message=mock.Mock(content='{"result": "test"}'))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    model = inference.OpenAILanguageModel(
        api_key="test-key", temperature=0.0  # Testing zero temperature
    )

    list(model.infer(["test prompt"]))

    # Verify temperature=0.0 was passed to the API
    mock_client.chat.completions.create.assert_called_with(
        model="gpt-4o-mini",
        messages=mock.ANY,
        temperature=0.0,
        n=1,
    )

  def test_openai_schema_constraints_json(self):
    # Test that schema constraints are only allowed with JSON format
    test_schema = schema.OpenAISchema._schema_dict = {
        "type": "object",
        "properties": {"test": {"type": "string"}},
    }
    openai_schema = schema.OpenAISchema(test_schema)

    # This should work - JSON format with schema
    model = inference.OpenAILanguageModel(
        api_key="test-key",
        format_type=data.FormatType.JSON,
        openai_schema=openai_schema,
    )
    self.assertEqual(model.openai_schema, openai_schema)

  def test_openai_schema_constraints_yaml_raises_error(self):
    # Test that schema constraints with YAML format raise an error
    test_schema = schema.OpenAISchema._schema_dict = {
        "type": "object",
        "properties": {"test": {"type": "string"}},
    }
    openai_schema = schema.OpenAISchema(test_schema)

    with self.assertRaises(ValueError) as context:
      inference.OpenAILanguageModel(
          api_key="test-key",
          format_type=data.FormatType.YAML,
          openai_schema=openai_schema,
      )

    self.assertIn(
        "OpenAI schema constraints are only supported with FormatType.JSON",
        str(context.exception),
    )

  @mock.patch("openai.OpenAI")
  def test_openai_with_schema_constraints(self, mock_openai_class):
    # Mock the OpenAI client and chat completion response
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    # Mock response structure
    mock_response = mock.Mock()
    mock_response.choices = [
        mock.Mock(
            message=mock.Mock(
                content=(
                    '{"extractions": [{"name": "John", "age_attributes":'
                    ' {"value": "30"}}]}'
                )
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # Create OpenAI schema
    examples = [
        data.ExampleData(
            text="Alice is 25 years old",
            extractions=[
                data.Extraction(
                    extraction_class="name",
                    extraction_text="Alice",
                    attributes={"type": "String"},
                ),
                data.Extraction(
                    extraction_class="age",
                    extraction_text="25",
                    attributes={"type": "Integer"},
                ),
            ],
        )
    ]
    openai_schema = schema.OpenAISchema.from_examples(examples)

    # Create model instance with schema
    model = inference.OpenAILanguageModel(
        model_id="gpt-4o-mini",
        api_key="test-api-key",
        openai_schema=openai_schema,
        temperature=0.5,
    )

    # Test inference
    batch_prompts = ["Extract name and age from: John is 30 years old"]
    results = list(model.infer(batch_prompts))

    # Verify we got results
    self.assertEqual(len(results), 1)
    self.assertEqual(len(results[0]), 1)

    # Verify API was called with schema constraints
    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(call_args[1]["model"], "gpt-4o-mini")
    self.assertEqual(call_args[1]["temperature"], 0.5)
    self.assertIn("response_format", call_args[1])
    self.assertEqual(call_args[1]["response_format"]["type"], "json_schema")
    self.assertIn("json_schema", call_args[1]["response_format"])
    self.assertEqual(
        call_args[1]["response_format"]["json_schema"]["name"],
        "langextract_extraction",
    )
    self.assertTrue(call_args[1]["response_format"]["json_schema"]["strict"])

    # Verify system message for structured output
    messages = call_args[1]["messages"]
    self.assertEqual(
        messages[0]["content"],
        "You are a helpful assistant that extracts information in the specified"
        " JSON format.",
    )


if __name__ == "__main__":
  absltest.main()
