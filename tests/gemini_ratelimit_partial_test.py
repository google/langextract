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

"""Tests for Gemini client-side rate limiting and partial-result preservation."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google import genai

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
  kwargs = {'model_id': 'gemini-3.5-flash', 'api_key': 'test-api-key'}
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
    self.generate = self.mock_client.models.generate_content


class TestRateLimiter(absltest.TestCase):
  """The leaky-bucket throttle spaces request starts by 60 / max_rpm."""

  def test_min_interval_derived_from_rpm(self):
    self.assertAlmostEqual(gemini._RateLimiter(120.0)._min_interval, 0.5)

  def test_spaces_calls_by_the_min_interval(self):
    # Freeze the clock so we assert the *scheduled* spacing, no real sleeping.
    limiter = gemini._RateLimiter(60.0)  # one per second
    sleeps = []
    fake_time = mock.Mock()
    fake_time.monotonic.return_value = 100.0
    fake_time.sleep.side_effect = sleeps.append
    with mock.patch.object(gemini, 'time', fake_time):
      limiter.acquire()  # first slot == now, no wait
      limiter.acquire()  # +1s
      limiter.acquire()  # +2s
    self.assertEqual(sleeps, [1.0, 2.0])


class TestMaxRpmConfig(_MockClientTest):
  """max_rpm wiring on the provider."""

  def test_throttle_disabled_by_default(self):
    self.assertIsNone(_build_model()._rate_limiter)

  def test_throttle_enabled_when_positive(self):
    model = _build_model(max_rpm=30)
    self.assertIsNotNone(model._rate_limiter)
    self.assertAlmostEqual(model._rate_limiter._min_interval, 2.0)

  @parameterized.named_parameters(
      dict(testcase_name='negative', value=-1),
      dict(testcase_name='bool', value=True),
  )
  def test_invalid_max_rpm_rejected(self, value):
    with self.assertRaises(exceptions.InferenceConfigError):
      _build_model(max_rpm=value)

  def test_acquire_runs_before_every_request(self):
    model = _build_model(max_rpm=600, max_workers=4)
    self.generate.side_effect = lambda **k: _make_response('{}')
    with mock.patch.object(model._rate_limiter, 'acquire') as acquire:
      list(model.infer(['p1', 'p2', 'p3']))
    self.assertEqual(acquire.call_count, 3)


class TestPartialResultPreservation(_MockClientTest):
  """A failed chunk must not discard the chunks that already succeeded."""

  def _side_effect(self, failing):
    def fake(**kwargs):
      contents = kwargs['contents']
      if contents in failing:
        raise ValueError(f'boom on {contents}')
      return _make_response(f'out-{contents}')

    return fake

  def test_all_success_yields_one_output_per_prompt_in_order(self):
    model = _build_model(max_workers=4)
    self.generate.side_effect = self._side_effect(failing=())
    outputs = list(model.infer(['a', 'b', 'c']))
    self.assertEqual(
        [o[0].output for o in outputs], ['out-a', 'out-b', 'out-c']
    )

  def test_failure_preserves_completed_chunks_on_the_exception(self):
    model = _build_model(max_retries=0, max_workers=4)
    self.generate.side_effect = self._side_effect(failing={'p2'})

    with self.assertRaises(exceptions.InferenceRuntimeError) as ctx:
      list(model.infer(['p1', 'p2', 'p3']))

    err = ctx.exception
    self.assertEqual(err.failed_indices, [1])
    self.assertIn('preserved', str(err))
    # Completed work is aligned to the prompts, None where the chunk failed.
    self.assertIsNotNone(err.partial_results)
    self.assertEqual(err.partial_results[0][0].output, 'out-p1')
    self.assertIsNone(err.partial_results[1])
    self.assertEqual(err.partial_results[2][0].output, 'out-p3')

  def test_reports_every_failed_index(self):
    model = _build_model(max_retries=0, max_workers=4)
    self.generate.side_effect = self._side_effect(failing={'p1', 'p3'})

    with self.assertRaises(exceptions.InferenceRuntimeError) as ctx:
      list(model.infer(['p1', 'p2', 'p3', 'p4']))

    err = ctx.exception
    self.assertEqual(err.failed_indices, [0, 2])
    self.assertEqual(err.partial_results[1][0].output, 'out-p2')
    self.assertEqual(err.partial_results[3][0].output, 'out-p4')


if __name__ == '__main__':
  absltest.main()
