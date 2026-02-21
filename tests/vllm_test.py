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

"""Tests for vLLM provider."""

from unittest import mock

from absl.testing import absltest

from langextract.core import data
from langextract.core import exceptions
from langextract.providers import vllm


class VLLMProviderTest(absltest.TestCase):

  @mock.patch("langextract.providers.vllm.requests.post")
  def test_infer_success_parses_message_content(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"ok": 1}'}}],
    }
    mock_post.return_value = mock_response

    model = vllm.VLLMLanguageModel(model_id="vllm:llama3")
    results = list(model.infer(["prompt"]))

    self.assertLen(results, 1)
    self.assertEqual(results[0][0].output, '{"ok": 1}')

    self.assertEqual(
        mock_post.call_args.args[0],
        "http://localhost:8000/v1/chat/completions",
    )

  @mock.patch("langextract.providers.vllm.requests.post")
  def test_base_url_without_v1_is_normalized(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
    }
    mock_post.return_value = mock_response

    model = vllm.VLLMLanguageModel(model_id="vllm:llama3", base_url="http://h")
    list(model.infer(["prompt"]))

    self.assertEqual(
        mock_post.call_args.args[0], "http://h/v1/chat/completions"
    )

  @mock.patch("langextract.providers.vllm.requests.post")
  def test_error_status_raises_inference_runtime_error(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 500
    mock_response.text = "boom"
    mock_response.json.return_value = {"error": {"message": "boom"}}
    mock_post.return_value = mock_response

    model = vllm.VLLMLanguageModel(model_id="vllm:llama3")
    with self.assertRaisesRegex(exceptions.InferenceRuntimeError, "vLLM API error"):
      list(model.infer(["prompt"]))

  @mock.patch("langextract.providers.vllm.requests.post")
  def test_invalid_response_shape_raises(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}
    mock_post.return_value = mock_response

    model = vllm.VLLMLanguageModel(model_id="vllm:llama3")
    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, "Unexpected vLLM response shape"
    ):
      list(model.infer(["prompt"]))

  @mock.patch.object(vllm.VLLMLanguageModel, "_process_single_prompt")
  def test_parallel_infer_preserves_order(self, mock_process):
    mock_process.side_effect = lambda prompt, cfg: mock.Mock(output=prompt)
    model = vllm.VLLMLanguageModel(model_id="vllm:llama3", max_workers=4)
    results = list(model.infer(["a", "b", "c"]))
    self.assertEqual([r[0].output for r in results], ["a", "b", "c"])

  def test_builds_system_message_from_format_type(self):
    model = vllm.VLLMLanguageModel(model_id="vllm:llama3", format_type=data.FormatType.YAML)
    messages = model._build_messages("p")  # pylint: disable=protected-access
    self.assertEqual(messages[0]["role"], "system")
    self.assertIn("YAML", messages[0]["content"])


if __name__ == "__main__":
  absltest.main()

