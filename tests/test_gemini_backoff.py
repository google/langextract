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

"""Tests for Gemini provider exponential backoff."""

from unittest import mock
from absl.testing import absltest
from langextract.core import exceptions
from langextract.providers import gemini

class TestGeminiBackoff(absltest.TestCase):

  @mock.patch("google.genai.Client")
  @mock.patch("time.sleep") # Mock sleep to speed up tests
  def test_gemini_retry_on_429(self, mock_sleep, mock_client_class):
    """Test that Gemini retries on 429 errors and eventually succeeds."""
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    # Simulate one 429 error followed by a success
    mock_response = mock.Mock()
    mock_response.text = '{"result": "success"}'

    mock_client.models.generate_content.side_effect = [
        Exception("429 RESOURCE_EXHAUSTED"),
        mock_response
    ]

    model = gemini.GeminiLanguageModel(
        api_key="test-key",
        max_retries=3
    )

    results = list(model.infer(["Test prompt"]))

    self.assertEqual(len(results), 1)
    self.assertEqual(results[0][0].output, '{"result": "success"}')
    self.assertEqual(mock_client.models.generate_content.call_count, 2)
    mock_sleep.assert_called_once()

  @mock.patch("google.genai.Client")
  @mock.patch("time.sleep")
  def test_gemini_max_retries_exceeded(self, mock_sleep, mock_client_class):
    """Test that Gemini fails after exceeding max retries."""
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    # Simulate continuous 429 errors
    mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")

    model = gemini.GeminiLanguageModel(
        api_key="test-key",
        max_retries=2
    )

    with self.assertRaises(exceptions.InferenceRuntimeError) as cm:
      list(model.infer(["Test prompt"]))

    self.assertIn("Gemini API error", str(cm.exception))
    self.assertIn("429", str(cm.exception))
    # 1 initial call + 2 retries = 3 calls
    self.assertEqual(mock_client.models.generate_content.call_count, 3)
    self.assertEqual(mock_sleep.call_count, 2)

  @mock.patch("google.genai.Client")
  @mock.patch("time.sleep")
  def test_gemini_no_retry_on_other_errors(self, mock_sleep, mock_client_class):
    """Test that Gemini does not retry on non-429 errors."""
    mock_client = mock.Mock()
    mock_client_class.return_value = mock_client

    # Simulate a non-429 error
    mock_client.models.generate_content.side_effect = Exception("500 Internal Server Error")

    model = gemini.GeminiLanguageModel(
        api_key="test-key",
        max_retries=3
    )

    with self.assertRaises(exceptions.InferenceRuntimeError) as cm:
      list(model.infer(["Test prompt"]))

    self.assertIn("500", str(cm.exception))
    self.assertEqual(mock_client.models.generate_content.call_count, 1)
    mock_sleep.assert_not_called()

if __name__ == "__main__":
  absltest.main()
