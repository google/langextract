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

"""Tests for Gemini provider retry logic on transient errors (503, 429)."""

import time
from unittest import mock

from absl.testing import absltest
from google import genai

from langextract.core import exceptions
from langextract.providers import gemini


class TestGeminiRetryableErrors(absltest.TestCase):
  """Test _is_retryable_error method."""

  def setUp(self):
    super().setUp()
    # Patch genai.Client to avoid actual API calls during test setup
    self.mock_genai_client_patcher = mock.patch.object(
        genai, 'Client', autospec=True
    )
    self.mock_genai_client_cls = self.mock_genai_client_patcher.start()
    self.model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
    )

  def tearDown(self):
    self.mock_genai_client_patcher.stop()
    super().tearDown()

  def test_503_error_is_retryable(self):
    """503 Service Unavailable (model overloaded) should be retryable."""
    error = Exception('503 The model is overloaded. Please try again later.')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_overloaded_message_is_retryable(self):
    """Errors mentioning 'overloaded' should be retryable."""
    error = Exception('The model is overloaded')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_429_error_is_retryable(self):
    """429 Too Many Requests should be retryable."""
    error = Exception('429 Resource has been exhausted. Too many requests.')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_rate_limit_error_is_retryable(self):
    """Rate limit errors should be retryable."""
    error = Exception('Rate limit exceeded for this API key')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_quota_error_is_retryable(self):
    """Quota errors should be retryable."""
    error = Exception('Quota exceeded for the day')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_500_internal_error_is_retryable(self):
    """500 Internal Server Error should be retryable."""
    error = Exception('500 Internal server error')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_connection_error_is_retryable(self):
    """ConnectionError should be retryable."""
    error = ConnectionError('Connection refused')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_timeout_error_is_retryable(self):
    """TimeoutError should be retryable."""
    error = TimeoutError('Connection timed out')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_os_error_is_retryable(self):
    """OSError (network issues) should be retryable."""
    error = OSError('Network unreachable')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_unavailable_message_is_retryable(self):
    """Errors mentioning 'unavailable' should be retryable."""
    error = Exception('Service temporarily unavailable')
    self.assertTrue(self.model._is_retryable_error(error))

  def test_400_error_is_not_retryable(self):
    """400 Bad Request should NOT be retryable."""
    error = Exception('400 Invalid JSON in request body')
    self.assertFalse(self.model._is_retryable_error(error))

  def test_401_error_is_not_retryable(self):
    """401 Unauthorized should NOT be retryable."""
    error = Exception('401 Invalid API key provided')
    self.assertFalse(self.model._is_retryable_error(error))

  def test_403_error_is_not_retryable(self):
    """403 Forbidden should NOT be retryable."""
    error = Exception('403 Permission denied for this resource')
    self.assertFalse(self.model._is_retryable_error(error))

  def test_404_error_is_not_retryable(self):
    """404 Not Found should NOT be retryable."""
    error = Exception('404 Model not found')
    self.assertFalse(self.model._is_retryable_error(error))

  def test_generic_value_error_is_not_retryable(self):
    """Generic ValueError should NOT be retryable."""
    error = ValueError('Invalid parameter value')
    self.assertFalse(self.model._is_retryable_error(error))


class TestGeminiRetryLogic(absltest.TestCase):
  """Test retry logic in _process_single_prompt."""

  def setUp(self):
    super().setUp()
    self.mock_genai_client_patcher = mock.patch.object(
        genai, 'Client', autospec=True
    )
    self.mock_genai_client_cls = self.mock_genai_client_patcher.start()
    self.mock_client = self.mock_genai_client_cls.return_value

  def tearDown(self):
    self.mock_genai_client_patcher.stop()
    super().tearDown()

  @mock.patch.object(time, 'sleep')
  def test_retry_on_503_error_success(self, mock_sleep):
    """Test that 503 errors are retried and succeed on subsequent attempt."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=3,
        retry_delay=1.0,
    )

    # First call fails with 503, second call succeeds
    mock_response = mock.create_autospec(
        genai.types.GenerateContentResponse, instance=True
    )
    mock_response.text = '{"result": "success"}'

    self.mock_client.models.generate_content.side_effect = [
        Exception('503 The model is overloaded'),
        mock_response,
    ]

    result = model._process_single_prompt('test prompt', {'temperature': 0.0})

    self.assertEqual(result.output, '{"result": "success"}')
    self.assertEqual(self.mock_client.models.generate_content.call_count, 2)
    mock_sleep.assert_called_once()

  @mock.patch.object(time, 'sleep')
  def test_retry_on_429_error_success(self, mock_sleep):
    """Test that 429 errors are retried and succeed on subsequent attempt."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=3,
        retry_delay=1.0,
    )

    mock_response = mock.create_autospec(
        genai.types.GenerateContentResponse, instance=True
    )
    mock_response.text = '{"result": "success"}'

    self.mock_client.models.generate_content.side_effect = [
        Exception('429 Rate limit exceeded'),
        mock_response,
    ]

    result = model._process_single_prompt('test prompt', {'temperature': 0.0})

    self.assertEqual(result.output, '{"result": "success"}')
    self.assertEqual(self.mock_client.models.generate_content.call_count, 2)
    mock_sleep.assert_called_once()

  @mock.patch.object(time, 'sleep')
  def test_retry_multiple_times_before_success(self, mock_sleep):
    """Test that multiple retries happen before eventual success."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=3,
        retry_delay=1.0,
        max_retry_delay=16.0,
    )

    mock_response = mock.create_autospec(
        genai.types.GenerateContentResponse, instance=True
    )
    mock_response.text = '{"result": "success"}'

    # Fail 3 times, succeed on 4th attempt
    self.mock_client.models.generate_content.side_effect = [
        Exception('503 The model is overloaded'),
        Exception('503 The model is overloaded'),
        Exception('503 The model is overloaded'),
        mock_response,
    ]

    result = model._process_single_prompt('test prompt', {'temperature': 0.0})

    self.assertEqual(result.output, '{"result": "success"}')
    self.assertEqual(self.mock_client.models.generate_content.call_count, 4)
    self.assertEqual(mock_sleep.call_count, 3)

  @mock.patch.object(time, 'sleep')
  def test_exponential_backoff(self, mock_sleep):
    """Test that retry delays increase exponentially."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=3,
        retry_delay=1.0,
        max_retry_delay=16.0,
    )

    mock_response = mock.create_autospec(
        genai.types.GenerateContentResponse, instance=True
    )
    mock_response.text = '{"result": "success"}'

    # Fail 3 times, succeed on 4th attempt
    self.mock_client.models.generate_content.side_effect = [
        Exception('503 The model is overloaded'),
        Exception('503 The model is overloaded'),
        Exception('503 The model is overloaded'),
        mock_response,
    ]

    model._process_single_prompt('test prompt', {'temperature': 0.0})

    # Check exponential backoff: 1.0, 2.0, 4.0
    sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
    self.assertEqual(sleep_calls, [1.0, 2.0, 4.0])

  @mock.patch.object(time, 'sleep')
  def test_max_retry_delay_cap(self, mock_sleep):
    """Test that retry delay is capped at max_retry_delay."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=5,
        retry_delay=4.0,
        max_retry_delay=8.0,
    )

    mock_response = mock.create_autospec(
        genai.types.GenerateContentResponse, instance=True
    )
    mock_response.text = '{"result": "success"}'

    # Fail 5 times, succeed on 6th attempt
    self.mock_client.models.generate_content.side_effect = [
        Exception('503 The model is overloaded'),
        Exception('503 The model is overloaded'),
        Exception('503 The model is overloaded'),
        Exception('503 The model is overloaded'),
        Exception('503 The model is overloaded'),
        mock_response,
    ]

    model._process_single_prompt('test prompt', {'temperature': 0.0})

    # Check that delay is capped: 4.0, 8.0, 8.0, 8.0, 8.0
    sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
    self.assertEqual(sleep_calls, [4.0, 8.0, 8.0, 8.0, 8.0])

  @mock.patch.object(time, 'sleep')
  def test_max_retries_exhausted(self, mock_sleep):
    """Test that error is raised after max retries are exhausted."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=2,
        retry_delay=1.0,
    )

    # All attempts fail
    self.mock_client.models.generate_content.side_effect = [
        RuntimeError('503 The model is overloaded'),
        RuntimeError('503 The model is overloaded'),
        RuntimeError('503 The model is overloaded'),  # 3rd attempt
    ]

    with self.assertRaises(exceptions.InferenceRuntimeError) as ctx:
      model._process_single_prompt('test prompt', {'temperature': 0.0})

    self.assertIn('503', str(ctx.exception))
    self.assertEqual(self.mock_client.models.generate_content.call_count, 3)
    self.assertEqual(mock_sleep.call_count, 2)

  def test_no_retry_on_400_error(self):
    """Test that 400 errors are not retried."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=3,
        retry_delay=1.0,
    )

    self.mock_client.models.generate_content.side_effect = Exception(
        '400 Invalid request'
    )

    with self.assertRaises(exceptions.InferenceRuntimeError) as ctx:
      model._process_single_prompt('test prompt', {'temperature': 0.0})

    self.assertIn('400', str(ctx.exception))
    # Should only be called once - no retries
    self.assertEqual(self.mock_client.models.generate_content.call_count, 1)

  def test_no_retry_on_401_error(self):
    """Test that 401 errors are not retried."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=3,
    )

    self.mock_client.models.generate_content.side_effect = Exception(
        '401 Invalid API key'
    )

    with self.assertRaises(exceptions.InferenceRuntimeError):
      model._process_single_prompt('test prompt', {'temperature': 0.0})

    self.assertEqual(self.mock_client.models.generate_content.call_count, 1)

  def test_retry_disabled_with_zero_max_retries(self):
    """Test that retries are disabled when max_retries=0."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=0,
    )

    self.mock_client.models.generate_content.side_effect = Exception(
        '503 The model is overloaded'
    )

    with self.assertRaises(exceptions.InferenceRuntimeError):
      model._process_single_prompt('test prompt', {'temperature': 0.0})

    # Should only be called once - no retries
    self.assertEqual(self.mock_client.models.generate_content.call_count, 1)


class TestGeminiParallelRetry(absltest.TestCase):
  """Test retry behavior in parallel inference scenarios."""

  def setUp(self):
    super().setUp()
    self.mock_genai_client_patcher = mock.patch.object(
        genai, 'Client', autospec=True
    )
    self.mock_genai_client_cls = self.mock_genai_client_patcher.start()
    self.mock_client = self.mock_genai_client_cls.return_value

  def tearDown(self):
    self.mock_genai_client_patcher.stop()
    super().tearDown()

  @mock.patch.object(time, 'sleep')
  def test_parallel_inference_with_one_chunk_503_retry(self, mock_sleep):
    """Test parallel inference where one chunk encounters 503 and retries.

    This tests the main user scenario: processing multiple chunks in parallel
    where one chunk gets a 503 error. The chunk should retry and eventually
    succeed without failing the entire batch.
    """
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_workers=2,
        max_retries=2,
        retry_delay=0.1,
    )

    # Track which prompt fails to ensure proper retry behavior
    call_count_per_prompt = {}

    def mock_generate_content(model, contents, config):
      prompt = contents
      if prompt not in call_count_per_prompt:
        call_count_per_prompt[prompt] = 0
      call_count_per_prompt[prompt] += 1

      # First prompt succeeds immediately
      if prompt == 'prompt_0':
        response = mock.create_autospec(
            genai.types.GenerateContentResponse, instance=True
        )
        response.text = '{"chunk": 0}'
        return response

      # Second prompt fails twice then succeeds
      if prompt == 'prompt_1':
        if call_count_per_prompt[prompt] <= 2:
          raise RuntimeError('503 The model is overloaded')
        response = mock.create_autospec(
            genai.types.GenerateContentResponse, instance=True
        )
        response.text = '{"chunk": 1}'
        return response

      # Third prompt succeeds immediately
      response = mock.create_autospec(
          genai.types.GenerateContentResponse, instance=True
      )
      response.text = f'{{"chunk": "{prompt}"}}'
      return response

    self.mock_client.models.generate_content.side_effect = mock_generate_content

    prompts = ['prompt_0', 'prompt_1', 'prompt_2']
    results = list(model.infer(prompts))

    # All 3 chunks should succeed
    self.assertLen(results, 3)
    self.assertEqual(results[0][0].output, '{"chunk": 0}')
    self.assertEqual(results[1][0].output, '{"chunk": 1}')
    self.assertEqual(results[2][0].output, '{"chunk": "prompt_2"}')

    # prompt_1 should have been called 3 times (2 failures + 1 success)
    self.assertEqual(call_count_per_prompt.get('prompt_1', 0), 3)

  @mock.patch.object(time, 'sleep')
  def test_parallel_inference_all_succeed_no_retry(self, mock_sleep):
    """Test parallel inference where all chunks succeed immediately."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_workers=4,
        max_retries=3,
    )

    def mock_generate_content(model, contents, config):
      response = mock.create_autospec(
          genai.types.GenerateContentResponse, instance=True
      )
      response.text = f'{{"prompt": "{contents}"}}'
      return response

    self.mock_client.models.generate_content.side_effect = mock_generate_content

    prompts = ['p1', 'p2', 'p3', 'p4']
    results = list(model.infer(prompts))

    self.assertLen(results, 4)
    # No sleep should be called since no retries needed
    mock_sleep.assert_not_called()

  @mock.patch.object(time, 'sleep')
  def test_parallel_inference_permanent_error_still_fails(self, mock_sleep):
    """Test that parallel inference fails on a permanent error (400)."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_workers=2,
        max_retries=2,
    )

    def mock_generate_content(model, contents, config):
      if contents == 'bad_prompt':
        raise RuntimeError('400 Invalid request format')
      response = mock.create_autospec(
          genai.types.GenerateContentResponse, instance=True
      )
      response.text = f'{{"prompt": "{contents}"}}'
      return response

    self.mock_client.models.generate_content.side_effect = mock_generate_content

    prompts = ['good_prompt', 'bad_prompt']

    with self.assertRaises(exceptions.InferenceRuntimeError) as ctx:
      list(model.infer(prompts))

    self.assertIn('400', str(ctx.exception))


class TestGeminiRetryConfiguration(absltest.TestCase):
  """Test retry configuration parameters."""

  @mock.patch.object(genai, 'Client', autospec=True)
  def test_default_retry_parameters(self, mock_client_cls):
    """Test that default retry parameters are set correctly."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
    )

    self.assertEqual(model.max_retries, 3)
    self.assertEqual(model.retry_delay, 1.0)
    self.assertEqual(model.max_retry_delay, 16.0)

  @mock.patch.object(genai, 'Client', autospec=True)
  def test_custom_retry_parameters(self, mock_client_cls):
    """Test that custom retry parameters are accepted."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        api_key='test-api-key',
        max_retries=5,
        retry_delay=2.0,
        max_retry_delay=32.0,
    )

    self.assertEqual(model.max_retries, 5)
    self.assertEqual(model.retry_delay, 2.0)
    self.assertEqual(model.max_retry_delay, 32.0)

  @mock.patch.object(genai, 'Client', autospec=True)
  def test_vertex_ai_with_retry_parameters(self, mock_client_cls):
    """Test that retry parameters work with Vertex AI configuration."""
    model = gemini.GeminiLanguageModel(
        model_id='gemini-2.5-flash',
        vertexai=True,
        project='test-project',
        location='us-central1',
        max_retries=4,
        retry_delay=0.5,
    )

    self.assertEqual(model.max_retries, 4)
    self.assertEqual(model.retry_delay, 0.5)


if __name__ == '__main__':
  absltest.main()
