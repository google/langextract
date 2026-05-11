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

"""Tests for Gemini provider retry logic on transient errors."""

import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google import genai
from google.genai import errors as genai_errors
import httpx

from langextract.core import exceptions
from langextract.providers import gemini


def _make_response(text: str):
  """Build a mock GenerateContentResponse carrying `text`."""
  response = mock.create_autospec(
      genai.types.GenerateContentResponse, instance=True
  )
  response.text = text
  return response


def _build_model(**overrides):
  """Build a GeminiLanguageModel with sensible test defaults."""
  kwargs = {'model_id': 'gemini-2.5-flash', 'api_key': 'test-api-key'}
  kwargs.update(overrides)
  return gemini.GeminiLanguageModel(**kwargs)


class _MockClientTest(parameterized.TestCase):
  """Base class that patches genai.Client for the duration of each test."""

  def setUp(self):
    super().setUp()
    patcher = mock.patch.object(genai, 'Client', autospec=True)
    self.mock_client_cls = patcher.start()
    self.addCleanup(patcher.stop)
    self.mock_client = self.mock_client_cls.return_value


class TestGeminiRetryableErrors(_MockClientTest):
  """Classification of errors as retryable vs terminal."""

  def setUp(self):
    super().setUp()
    self.model = _build_model()

  @parameterized.named_parameters(
      dict(
          testcase_name='503_overloaded',
          message='503 The model is overloaded. Try again later.',
      ),
      dict(
          testcase_name='bare_overloaded',
          message='The model is overloaded',
      ),
      dict(
          testcase_name='429_rate_limit',
          message='429 Resource has been exhausted.',
      ),
      dict(
          testcase_name='rate_limit_phrase',
          message='Rate limit exceeded for this API key',
      ),
      dict(
          testcase_name='quota_exceeded',
          message='Quota exceeded for the day',
      ),
      dict(
          testcase_name='500_internal',
          message='500 Internal server error',
      ),
      dict(
          testcase_name='temporarily_unavailable',
          message='Service temporarily unavailable',
      ),
  )
  def test_retryable_by_message(self, message):
    self.assertTrue(self.model._is_retryable_error(Exception(message)))

  @parameterized.named_parameters(
      dict(testcase_name='400', message='400 Invalid JSON in request body'),
      dict(testcase_name='401', message='401 Invalid API key'),
      dict(testcase_name='403', message='403 Permission denied'),
      dict(testcase_name='404', message='404 Model not found'),
      # Narrowed heuristics: bare "quota"/"unavailable" can be permanent.
      dict(
          testcase_name='bare_quota',
          message='Quota denied for this region',
      ),
      dict(
          testcase_name='bare_unavailable',
          message='This model is unavailable in your region',
      ),
  )
  def test_not_retryable_by_message(self, message):
    self.assertFalse(self.model._is_retryable_error(Exception(message)))

  def test_generic_value_error_is_not_retryable(self):
    self.assertFalse(self.model._is_retryable_error(ValueError('oops')))

  @parameterized.named_parameters(
      dict(
          testcase_name='connection_error',
          error=ConnectionError('refused'),
      ),
      dict(
          testcase_name='timeout_error',
          error=TimeoutError('timed out'),
      ),
  )
  def test_stdlib_network_errors_are_retryable(self, error):
    self.assertTrue(self.model._is_retryable_error(error))

  def test_bare_os_error_is_not_retryable(self):
    """Bare OSError (file/permission) is excluded."""
    self.assertFalse(self.model._is_retryable_error(OSError('file not found')))

  @parameterized.named_parameters(
      dict(testcase_name='408', code=408),
      dict(testcase_name='429', code=429),
      dict(testcase_name='500', code=500),
      dict(testcase_name='502', code=502),
      dict(testcase_name='503', code=503),
      dict(testcase_name='504', code=504),
  )
  def test_typed_apierror_retryable_codes(self, code):
    error = genai_errors.APIError(code=code, response_json={})
    self.assertTrue(self.model._is_retryable_error(error))

  @parameterized.named_parameters(
      dict(testcase_name='400', code=400),
      dict(testcase_name='401', code=401),
      dict(testcase_name='403', code=403),
      dict(testcase_name='404', code=404),
  )
  def test_typed_apierror_terminal_codes(self, code):
    error = genai_errors.APIError(code=code, response_json={})
    self.assertFalse(self.model._is_retryable_error(error))

  @parameterized.named_parameters(
      dict(
          testcase_name='connect_error',
          error=httpx.ConnectError('refused'),
      ),
      dict(
          testcase_name='read_error',
          error=httpx.ReadError('read failed'),
      ),
      dict(
          testcase_name='connect_timeout',
          error=httpx.ConnectTimeout('connect timeout'),
      ),
  )
  def test_httpx_transport_errors_are_retryable(self, error):
    self.assertTrue(self.model._is_retryable_error(error))

  @parameterized.named_parameters(
      # Client/config bugs: retrying cannot fix them.
      dict(
          testcase_name='local_protocol',
          error=httpx.LocalProtocolError('bad header'),
      ),
      dict(
          testcase_name='unsupported_protocol',
          error=httpx.UnsupportedProtocol('ftp unsupported'),
      ),
  )
  def test_httpx_client_bugs_are_not_retryable(self, error):
    self.assertFalse(self.model._is_retryable_error(error))


class TestGeminiRetryLogic(_MockClientTest):
  """Behavior of the retry loop inside _process_single_prompt."""

  @parameterized.named_parameters(
      dict(testcase_name='503', message='503 The model is overloaded'),
      dict(testcase_name='429', message='429 Rate limit exceeded'),
  )
  @mock.patch.object(time, 'sleep')
  def test_retry_then_success(self, mock_sleep, message):
    model = _build_model(max_retries=3, retry_delay=1.0)
    self.mock_client.models.generate_content.side_effect = [
        Exception(message),
        _make_response('{"ok": 1}'),
    ]

    result = model._process_single_prompt('prompt', {'temperature': 0.0})

    self.assertEqual(result.output, '{"ok": 1}')
    self.assertEqual(self.mock_client.models.generate_content.call_count, 2)
    mock_sleep.assert_called_once()

  @mock.patch.object(time, 'sleep')
  def test_retry_multiple_times_before_success(self, mock_sleep):
    model = _build_model(max_retries=3, retry_delay=1.0, max_retry_delay=16.0)
    self.mock_client.models.generate_content.side_effect = [
        Exception('503'),
        Exception('503'),
        Exception('503'),
        _make_response('{"ok": 1}'),
    ]

    model._process_single_prompt('prompt', {'temperature': 0.0})

    self.assertEqual(self.mock_client.models.generate_content.call_count, 4)
    self.assertEqual(mock_sleep.call_count, 3)

  @mock.patch.object(time, 'sleep')
  def test_exponential_backoff_is_jittered(self, mock_sleep):
    """Each sleep is delay * uniform(0.5, 1.5); delay doubles each attempt."""
    model = _build_model(max_retries=3, retry_delay=1.0, max_retry_delay=16.0)
    self.mock_client.models.generate_content.side_effect = [
        Exception('503'),
        Exception('503'),
        Exception('503'),
        _make_response('{"ok": 1}'),
    ]

    model._process_single_prompt('prompt', {'temperature': 0.0})

    sleeps = [call.args[0] for call in mock_sleep.call_args_list]
    # delays 1.0, 2.0, 4.0 -> jitter windows [0.5, 1.5], [1.0, 3.0], [2.0, 6.0]
    for observed, (low, high) in zip(
        sleeps, [(0.5, 1.5), (1.0, 3.0), (2.0, 6.0)]
    ):
      self.assertGreaterEqual(observed, low)
      self.assertLessEqual(observed, high)

  @mock.patch.object(time, 'sleep')
  def test_max_retry_delay_caps_post_jitter_sleep(self, mock_sleep):
    model = _build_model(max_retries=5, retry_delay=4.0, max_retry_delay=8.0)
    self.mock_client.models.generate_content.side_effect = [
        Exception('503'),
    ] * 5 + [_make_response('{"ok": 1}')]

    model._process_single_prompt('prompt', {'temperature': 0.0})

    sleeps = [call.args[0] for call in mock_sleep.call_args_list]
    self.assertLen(sleeps, 5)
    # First: min(4*[0.5,1.5], 8) -> [2, 6]; subsequent: delay caps to 8.
    self._assert_in_range(sleeps[0], 2.0, 6.0)
    for s in sleeps[1:]:
      self._assert_in_range(s, 4.0, 8.0)

  def _assert_in_range(self, value, low, high):
    self.assertGreaterEqual(value, low)
    self.assertLessEqual(value, high)

  @mock.patch.object(time, 'sleep')
  def test_max_retries_exhausted_raises(self, mock_sleep):
    model = _build_model(max_retries=2, retry_delay=1.0)
    self.mock_client.models.generate_content.side_effect = RuntimeError(
        '503 overloaded'
    )

    with self.assertRaises(exceptions.InferenceRuntimeError) as ctx:
      model._process_single_prompt('prompt', {'temperature': 0.0})

    self.assertIn('503', str(ctx.exception))
    self.assertEqual(self.mock_client.models.generate_content.call_count, 3)
    self.assertEqual(mock_sleep.call_count, 2)

  @parameterized.named_parameters(
      dict(testcase_name='400', message='400 Invalid request'),
      dict(testcase_name='401', message='401 Invalid API key'),
  )
  def test_terminal_errors_not_retried(self, message):
    model = _build_model(max_retries=3, retry_delay=1.0)
    self.mock_client.models.generate_content.side_effect = Exception(message)

    with self.assertRaises(exceptions.InferenceRuntimeError):
      model._process_single_prompt('prompt', {'temperature': 0.0})

    self.assertEqual(self.mock_client.models.generate_content.call_count, 1)

  def test_zero_max_retries_disables_retry(self):
    model = _build_model(max_retries=0)
    self.mock_client.models.generate_content.side_effect = Exception(
        '503 overloaded'
    )

    with self.assertRaises(exceptions.InferenceRuntimeError):
      model._process_single_prompt('prompt', {'temperature': 0.0})

    self.assertEqual(self.mock_client.models.generate_content.call_count, 1)


class TestGeminiParallelRetry(_MockClientTest):
  """Retry behavior under concurrent chunk processing."""

  @mock.patch.object(time, 'sleep')
  def test_single_chunk_retries_without_failing_batch(self, _mock_sleep):
    """One flaky chunk retries while peers complete independently."""
    model = _build_model(max_workers=2, max_retries=2, retry_delay=0.1)
    calls = {}

    def side_effect(model, contents, config):  # pylint: disable=unused-argument
      calls[contents] = calls.get(contents, 0) + 1
      if contents == 'flaky' and calls[contents] <= 2:
        raise RuntimeError('503 overloaded')
      return _make_response(f'{{"p": "{contents}"}}')

    self.mock_client.models.generate_content.side_effect = side_effect

    results = list(model.infer(['ok', 'flaky', 'fine']))

    self.assertLen(results, 3)
    self.assertEqual(results[1][0].output, '{"p": "flaky"}')
    self.assertEqual(calls['flaky'], 3)

  @mock.patch.object(time, 'sleep')
  def test_all_succeed_never_sleeps(self, mock_sleep):
    model = _build_model(max_workers=4, max_retries=3)
    self.mock_client.models.generate_content.side_effect = (
        lambda model, contents, config: _make_response(f'{{"p": "{contents}"}}')
    )

    list(model.infer(['a', 'b', 'c', 'd']))

    mock_sleep.assert_not_called()

  @mock.patch.object(time, 'sleep')
  def test_permanent_error_fails_batch(self, _mock_sleep):
    model = _build_model(max_workers=2, max_retries=2)

    def side_effect(model, contents, config):  # pylint: disable=unused-argument
      if contents == 'bad':
        raise RuntimeError('400 invalid')
      return _make_response(f'{{"p": "{contents}"}}')

    self.mock_client.models.generate_content.side_effect = side_effect

    with self.assertRaises(exceptions.InferenceRuntimeError) as ctx:
      list(model.infer(['good', 'bad']))
    self.assertIn('400', str(ctx.exception))


class TestGeminiRetryConfiguration(_MockClientTest):
  """Constructor accepts and validates retry knobs."""

  def test_default_retry_parameters(self):
    model = _build_model()
    self.assertEqual(model.max_retries, 3)
    self.assertEqual(model.retry_delay, 1.0)
    self.assertEqual(model.max_retry_delay, 16.0)

  def test_custom_retry_parameters(self):
    model = _build_model(max_retries=5, retry_delay=2.0, max_retry_delay=32.0)
    self.assertEqual(model.max_retries, 5)
    self.assertEqual(model.retry_delay, 2.0)
    self.assertEqual(model.max_retry_delay, 32.0)

  def test_vertex_ai_with_retry_parameters(self):
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

  @parameterized.named_parameters(
      dict(
          testcase_name='negative_max_retries',
          overrides={'max_retries': -1},
      ),
      dict(
          testcase_name='non_integral_max_retries',
          overrides={'max_retries': 1.5},
      ),
      dict(
          testcase_name='bool_max_retries',
          overrides={'max_retries': True},
      ),
      dict(
          testcase_name='negative_retry_delay',
          overrides={'retry_delay': -0.1},
      ),
      dict(
          testcase_name='string_retry_delay',
          overrides={'retry_delay': '1.0'},
      ),
      dict(
          testcase_name='bool_retry_delay',
          overrides={'retry_delay': False},
      ),
      dict(
          testcase_name='zero_max_retry_delay',
          overrides={'max_retry_delay': 0},
      ),
      dict(
          testcase_name='negative_max_retry_delay',
          overrides={'max_retry_delay': -1.0},
      ),
      dict(
          testcase_name='string_max_retry_delay',
          overrides={'max_retry_delay': '2.0'},
      ),
      dict(
          testcase_name='bool_max_retry_delay',
          overrides={'max_retry_delay': True},
      ),
  )
  def test_invalid_knobs_raise(self, overrides):
    with self.assertRaises(exceptions.InferenceConfigError):
      _build_model(**overrides)
    self.mock_client_cls.assert_not_called()


def _http_options_object(attempts):
  """Build an HttpOptions-like object with retry_options.attempts=attempts."""
  retry_options = mock.MagicMock(spec=['attempts'])
  retry_options.attempts = attempts
  return mock.MagicMock(spec=['retry_options'], retry_options=retry_options)


class TestGeminiHttpOptionsRetryGuard(_MockClientTest):
  """SDK retry stacking guard.

  google-genai's HttpRetryOptions.attempts is "total attempts including the
  first"; 0 or 1 is normalized to no retries. Only guard when SDK retries
  would actually execute (attempts is None or > 1).
  """

  @parameterized.named_parameters(
      dict(
          testcase_name='object_attempts_default',
          http_options=_http_options_object(None),
      ),
      dict(
          testcase_name='object_attempts_5',
          http_options=_http_options_object(5),
      ),
      dict(
          testcase_name='dict_snake_attempts_default',
          http_options={'retry_options': {}},
      ),
      dict(
          testcase_name='dict_snake_attempts_5',
          http_options={'retry_options': {'attempts': 5}},
      ),
      dict(
          testcase_name='dict_camel_attempts_default',
          http_options={'retryOptions': {}},
      ),
      dict(
          testcase_name='dict_camel_attempts_5',
          http_options={'retryOptions': {'attempts': 5}},
      ),
  )
  def test_raises_when_sdk_retries_effective(self, http_options):
    with self.assertRaises(exceptions.InferenceConfigError):
      _build_model(http_options=http_options, max_retries=3)

  @parameterized.named_parameters(
      # attempts <= 1 means SDK retries do not execute; no stacking risk.
      dict(
          testcase_name='object_attempts_0',
          http_options=_http_options_object(0),
      ),
      dict(
          testcase_name='object_attempts_1',
          http_options=_http_options_object(1),
      ),
      dict(
          testcase_name='dict_snake_attempts_0',
          http_options={'retry_options': {'attempts': 0}},
      ),
      dict(
          testcase_name='dict_snake_attempts_1',
          http_options={'retry_options': {'attempts': 1}},
      ),
      dict(
          testcase_name='dict_camel_attempts_0',
          http_options={'retryOptions': {'attempts': 0}},
      ),
      dict(
          testcase_name='dict_camel_attempts_1',
          http_options={'retryOptions': {'attempts': 1}},
      ),
  )
  def test_allowed_when_sdk_retries_noop(self, http_options):
    model = _build_model(http_options=http_options, max_retries=3)
    self.assertEqual(model.max_retries, 3)

  def test_allowed_when_sdk_retries_disabled_via_max_retries(self):
    http_options = _http_options_object(5)
    model = _build_model(http_options=http_options, max_retries=0)
    self.assertEqual(model.max_retries, 0)

  def test_allowed_when_http_options_has_no_retry(self):
    http_options = mock.MagicMock(spec=['retry_options'], retry_options=None)
    model = _build_model(http_options=http_options, max_retries=3)
    self.assertEqual(model.max_retries, 3)


if __name__ == '__main__':
  absltest.main()
