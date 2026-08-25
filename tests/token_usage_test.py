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

"""Tests for token usage and API call tracking."""

from unittest import mock

from absl.testing import absltest

from langextract import annotation
from langextract.core import data
from langextract.core import format_handler as fh
from langextract.core import types
from langextract.providers import gemini
from langextract.providers import ollama
from langextract.providers import openai


class TestTokenUsageTracking(absltest.TestCase):

  @mock.patch("google.genai.Client")
  def test_gemini_token_usage_extraction(self, mock_client_class):
    """Test that GeminiLanguageModel extracts token usage metadata."""
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    # Simulate response carrying usage metadata
    mock_response = mock.Mock()
    mock_response.text = '{"extractions": []}'
    mock_usage = mock.Mock()
    mock_usage.prompt_token_count = 100
    mock_usage.candidates_token_count = 50
    mock_usage.total_token_count = 150
    mock_response.usage_metadata = mock_usage

    mock_client.models.generate_content.return_value = mock_response

    model = gemini.GeminiLanguageModel(api_key="test-key")
    results = list(model.infer(["Test prompt"]))

    self.assertLen(results, 1)
    scored_output = results[0][0]
    self.assertIsNotNone(scored_output.token_usage)
    self.assertEqual(scored_output.token_usage.prompt_tokens, 100)
    self.assertEqual(scored_output.token_usage.completion_tokens, 50)
    self.assertEqual(scored_output.token_usage.total_tokens, 150)

  @mock.patch("openai.OpenAI")
  def test_openai_token_usage_extraction(self, mock_openai_class):
    """Test that OpenAILanguageModel extracts token usage metadata."""
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    # Simulate OpenAI ChatCompletion response carrying usage and id
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_choice.message.content = '{"extractions": []}'
    mock_response.choices = [mock_choice]
    mock_response.id = "chatcmpl-test-id"

    mock_usage = mock.Mock()
    mock_usage.prompt_tokens = 80
    mock_usage.completion_tokens = 40
    mock_usage.total_tokens = 120
    mock_response.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_response

    model = openai.OpenAILanguageModel(api_key="test-key")
    results = list(model.infer(["Test prompt"]))

    self.assertLen(results, 1)
    scored_output = results[0][0]
    self.assertIsNotNone(scored_output.token_usage)
    self.assertEqual(scored_output.token_usage.prompt_tokens, 80)
    self.assertEqual(scored_output.token_usage.completion_tokens, 40)
    self.assertEqual(scored_output.token_usage.total_tokens, 120)
    self.assertEqual(scored_output.request_id, "chatcmpl-test-id")

  @mock.patch("requests.post")
  def test_ollama_token_usage_extraction(self, mock_post):
    """Test that OllamaLanguageModel extracts token usage metadata."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": '{"extractions": []}',
        "prompt_eval_count": 60,
        "eval_count": 30,
    }
    mock_post.return_value = mock_response

    model = ollama.OllamaLanguageModel(model_id="gemma")
    results = list(model.infer(["Test prompt"]))

    self.assertLen(results, 1)
    scored_output = results[0][0]
    self.assertIsNotNone(scored_output.token_usage)
    self.assertEqual(scored_output.token_usage.prompt_tokens, 60)
    self.assertEqual(scored_output.token_usage.completion_tokens, 30)
    self.assertEqual(scored_output.token_usage.total_tokens, 90)

  def test_annotation_pipeline_aggregates_usage(self):
    """Test that annotation aggregates usage and api_calls in single-pass."""
    mock_lm = mock.Mock(spec=gemini.GeminiLanguageModel)
    mock_lm.requires_fence_output = False

    # We will simulate 3 chunks of inference (text of length 41 with buffer 20)
    mock_lm.infer.side_effect = [
        [[
            types.ScoredOutput(
                score=1.0,
                output='{"extractions": [{"a": "b"}]}',
                token_usage=types.TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
                request_id="req-1",
            )
        ]],
        [[
            types.ScoredOutput(
                score=1.0,
                output='{"extractions": [{"c": "d"}]}',
                token_usage=types.TokenUsage(
                    prompt_tokens=20, completion_tokens=10, total_tokens=30
                ),
                request_id="req-2",
            )
        ]],
        [[
            types.ScoredOutput(
                score=1.0,
                output='{"extractions": []}',
                token_usage=types.TokenUsage(
                    prompt_tokens=5, completion_tokens=2, total_tokens=7
                ),
                request_id="req-3",
            )
        ]],
    ]

    mock_template = mock.Mock()
    mock_template.description = "Test description"
    mock_template.examples = []

    format_handler = fh.FormatHandler()
    annotator = annotation.Annotator(
        language_model=mock_lm,
        prompt_template=mock_template,
        format_handler=format_handler,
    )

    resolver = mock.Mock()
    # Return simple dummy extractions
    resolver.resolve.side_effect = [
        [data.Extraction(extraction_class="value", extraction_text="b")],
        [data.Extraction(extraction_class="value", extraction_text="d")],
        [],
    ]
    resolver.align.side_effect = lambda ex, *args, **kwargs: ex

    doc = data.Document(
        text="This is a test document with some chunks.", document_id="doc-1"
    )

    results = list(
        annotator.annotate_documents(
            documents=[doc],
            resolver=resolver,
            max_char_buffer=20,  # Small buffer to force 3 chunks
            batch_length=1,
            track_api_call_details=True,
        )
    )

    self.assertLen(results, 1)
    annotated_doc = results[0]
    self.assertEqual(annotated_doc.document_id, "doc-1")
    self.assertIsNotNone(annotated_doc.metadata)
    self.assertEqual(annotated_doc.metadata["api_calls"], 3)

    usage = annotated_doc.metadata["token_usage"]
    self.assertEqual(usage["prompt_tokens"], 35)
    self.assertEqual(usage["completion_tokens"], 17)
    self.assertEqual(usage["total_tokens"], 52)

    details = annotated_doc.metadata["api_call_details"]
    self.assertLen(details, 3)
    self.assertEqual(details[0]["chunk_index"], 0)
    self.assertEqual(details[0]["request_id"], "req-1")
    self.assertEqual(details[0]["token_usage"]["total_tokens"], 15)
    self.assertEqual(details[1]["chunk_index"], 1)
    self.assertEqual(details[1]["request_id"], "req-2")
    self.assertEqual(details[1]["token_usage"]["total_tokens"], 30)
    self.assertEqual(details[2]["chunk_index"], 2)
    self.assertEqual(details[2]["request_id"], "req-3")
    self.assertEqual(details[2]["token_usage"]["total_tokens"], 7)

  def test_annotation_pipeline_aggregates_usage_sequential_passes(self):
    """Test that annotation aggregates usage and api_calls in sequential passes."""
    mock_lm = mock.Mock(spec=gemini.GeminiLanguageModel)
    mock_lm.requires_fence_output = False

    # 2 passes, 1 chunk each
    mock_lm.infer.side_effect = [
        [[
            types.ScoredOutput(
                score=1.0,
                output='{"extractions": []}',
                token_usage=types.TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
                request_id="req-pass-0",
            )
        ]],
        [[
            types.ScoredOutput(
                score=1.0,
                output='{"extractions": []}',
                token_usage=types.TokenUsage(
                    prompt_tokens=20, completion_tokens=10, total_tokens=30
                ),
                request_id="req-pass-1",
            )
        ]],
    ]

    mock_template = mock.Mock()
    mock_template.description = "Test description"
    mock_template.examples = []

    format_handler = fh.FormatHandler()
    annotator = annotation.Annotator(
        language_model=mock_lm,
        prompt_template=mock_template,
        format_handler=format_handler,
    )

    resolver = mock.Mock()
    resolver.resolve.return_value = []
    resolver.align.return_value = []

    doc = data.Document(text="Single chunk document.", document_id="doc-1")

    results = list(
        annotator.annotate_documents(
            documents=[doc],
            resolver=resolver,
            max_char_buffer=500,
            batch_length=1,
            extraction_passes=2,
            track_api_call_details=True,
        )
    )

    self.assertLen(results, 1)
    annotated_doc = results[0]
    self.assertEqual(annotated_doc.metadata["api_calls"], 2)

    usage = annotated_doc.metadata["token_usage"]
    self.assertEqual(usage["prompt_tokens"], 30)
    self.assertEqual(usage["completion_tokens"], 15)
    self.assertEqual(usage["total_tokens"], 45)

    details = annotated_doc.metadata["api_call_details"]
    self.assertLen(details, 2)
    self.assertEqual(details[0]["pass_index"], 0)
    self.assertEqual(details[0]["request_id"], "req-pass-0")
    self.assertEqual(details[1]["pass_index"], 1)
    self.assertEqual(details[1]["request_id"], "req-pass-1")


if __name__ == "__main__":
  absltest.main()
