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

"""Tests for llama.cpp provider."""

from unittest import mock

from absl.testing import absltest

from langextract import inference
from langextract.core import data
from langextract.providers import llama_cpp as llama_cpp_provider


class TestLlamaCppLanguageModel(absltest.TestCase):

  @mock.patch("openai.OpenAI")
  def test_llama_cpp_init_client(self, mock_openai_class):
    """Verify client is initialized with /v1 model_url and optional api_key."""
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    model = llama_cpp_provider.LlamaCppLanguageModel(
        model_id="llama-cpp:MyModel",
        model_url="http://localhost:8080",
        api_key=None,  # optional
    )

    mock_openai_class.assert_called_once()
    call_args = mock_openai_class.call_args
    # api_key should not be passed when None
    self.assertNotIn("api_key", call_args.kwargs)
    # model_url should be normalized to include /v1 suffix
    self.assertEqual(call_args.kwargs["model_url"], "http://localhost:8080/v1")

    # Also ensure model_id prefix was stripped
    self.assertEqual(model.model_id, "MyModel")

  @mock.patch("openai.OpenAI")
  def test_llama_cpp_infer_with_parameters(self, mock_openai_class):
    """Ensure parameters and messages are forwarded to the API."""
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    mock_response = mock.Mock()
    mock_response.choices = [
        mock.Mock(message=mock.Mock(content='{"name": "John", "age": 30}'))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    model = llama_cpp_provider.LlamaCppLanguageModel(
        model_id="llama.cpp:Qwen2.5-1.5B-Instruct",
        model_url="http://127.0.0.1:8080",
        temperature=0.4,
        format_type=data.FormatType.JSON,
    )

    batch_prompts = [
        "Extract name and age from: John is 30 years old",
    ]
    results = list(model.infer(batch_prompts))

    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(call_args.kwargs["model"], "Qwen2.5-1.5B-Instruct")
    self.assertEqual(call_args.kwargs.get("temperature"), 0.4)
    self.assertEqual(call_args.kwargs["n"], 1)
    self.assertEqual(
        call_args.kwargs["response_format"], {"type": "json_object"}
    )
    self.assertEqual(len(call_args.kwargs["messages"]), 2)
    self.assertEqual(call_args.kwargs["messages"][0]["role"], "system")
    self.assertEqual(call_args.kwargs["messages"][1]["role"], "user")

    expected = [[
        inference.ScoredOutput(
            score=1.0, output='{"name": "John", "age": 30}'
        )
    ]]
    self.assertEqual(results, expected)

  @mock.patch("openai.OpenAI")
  def test_llama_cpp_temperature_none_not_sent(self, mock_openai_class):
    """Verify that temperature=None is not forwarded to API."""
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    mock_response = mock.Mock()
    mock_response.choices = [mock.Mock(message=mock.Mock(content="{}"))]
    mock_client.chat.completions.create.return_value = mock_response

    model = llama_cpp_provider.LlamaCppLanguageModel(
        model_id="MyModel",
        model_url="http://localhost:8080",
        temperature=None,
    )

    list(model.infer(["test"]))
    call_args = mock_client.chat.completions.create.call_args
    self.assertNotIn("temperature", call_args.kwargs)

  def test_requires_fence_output_false_for_json(self):
    """JSON mode should not require fences for parsing."""
    with mock.patch("openai.OpenAI") as mock_openai_class:
      mock_openai_class.return_value = mock.Mock()
      model = llama_cpp_provider.LlamaCppLanguageModel(
          model_id="MyModel",
          model_url="http://localhost:8080",
          format_type=data.FormatType.JSON,
      )
    self.assertFalse(model.requires_fence_output)


if __name__ == "__main__":
  absltest.main()
