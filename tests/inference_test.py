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

from langextract import data
from langextract import inference


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
        model="gemma2:latest",
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


class TestOpenAILanguageModel(absltest.TestCase):

  @mock.patch("openai.OpenAI")
  def test_openai_infer(self, mock_openai_class):
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
        model_id="gpt-4o-mini", api_key="test-api-key", temperature=0.5
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
        temperature=0.5,
        max_tokens=None,
        top_p=None,
        n=1,
    )

    # Check results
    expected_results = [[
        inference.ScoredOutput(score=1.0, output='{"name": "John", "age": 30}')
    ]]
    self.assertEqual(results, expected_results)

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
        max_tokens=None,
        top_p=None,
        n=1,
    )


if __name__ == "__main__":
  absltest.main()
