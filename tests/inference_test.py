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
import langfun as lf

from langextract import inference


class TestLangFunLanguageModel(absltest.TestCase):

  @mock.patch.object(
      inference.lf.core.language_model, "LanguageModel", autospec=True
  )
  def test_langfun_infer(self, mock_lf_model):
    mock_client_instance = mock_lf_model.return_value
    metadata = {
        "score": -0.004259720362824737,
        "logprobs": None,
        "is_cached": False,
    }
    source = lf.UserMessage(
        text="What's heart in Italian?.",
        sender="User",
        metadata={"formatted_text": "What's heart in Italian?."},
        tags=["lm-input"],
    )
    sample = lf.LMSample(
        response=lf.AIMessage(
            text="Cuore",
            sender="AI",
            metadata=metadata,
            source=source,
            tags=["lm-response"],
        ),
        score=-0.004259720362824737,
    )
    actual_response = lf.LMSamplingResult(
        samples=[sample],
    )

    # Mock the sample response.
    mock_client_instance.sample.return_value = [actual_response]
    model = inference.LangFunLanguageModel(language_model=mock_client_instance)

    batch_prompts = ["What's heart in Italian?"]

    expected_results = [
        [inference.ScoredOutput(score=-0.004259720362824737, output="Cuore")]
    ]

    results = list(model.infer(batch_prompts))

    mock_client_instance.sample.assert_called_once_with(prompts=batch_prompts)
    self.assertEqual(results, expected_results)


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


if __name__ == "__main__":
  absltest.main()
