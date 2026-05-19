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

"""Tests for OpenAI Batch API helper."""

# pylint: disable=too-few-public-methods

from __future__ import annotations

import json
import types as py_types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract.core import exceptions
from langextract.providers import openai_batch


class _FakeFiles:

  def __init__(self):
    self.created = []
    self.deleted = []
    self._content_by_id = {}

  def create(self, *, file, purpose):
    self.created.append({'file': file, 'purpose': purpose})
    return py_types.SimpleNamespace(id=f'file-{len(self.created)}')

  def content(self, file_id):
    return py_types.SimpleNamespace(text=self._content_by_id[file_id])

  def delete(self, file_id):
    self.deleted.append(file_id)

  def set_content(self, file_id: str, text: str) -> None:
    self._content_by_id[file_id] = text


class _FakeBatches:

  def __init__(self):
    self.created = []
    self.cancelled = []
    self._retrieve_queue = []

  def create(self, **kwargs):
    self.created.append(kwargs)
    return py_types.SimpleNamespace(id=f'batch-{len(self.created)}')

  def retrieve(self, _batch_id):
    if not self._retrieve_queue:
      raise RuntimeError('retrieve queue empty')
    return self._retrieve_queue.pop(0)

  def cancel(self, batch_id):
    self.cancelled.append(batch_id)

  def push_retrieve(self, obj):
    self._retrieve_queue.append(obj)


class _FakeClient:

  def __init__(self):
    self.files = _FakeFiles()
    self.batches = _FakeBatches()


class _StatusError(Exception):

  def __init__(self, status_code: int):
    super().__init__(f'Error code: {status_code}')
    self.status_code = status_code


def _make_output_line(custom_id: str, content: str) -> str:
  obj = {
      'custom_id': custom_id,
      'response': {
          'status_code': 200,
          'body': {
              'choices': [
                  {'message': {'content': content}},
              ]
          },
      },
      'error': None,
  }
  return json.dumps(obj)


def _make_error_line(
    custom_id: str, message: str, status_code: int = 400
) -> str:
  obj = {
      'custom_id': custom_id,
      'response': {
          'status_code': status_code,
          'body': {'error': {'message': message}},
      },
      'error': {'message': message},
  }
  return json.dumps(obj)


def _make_refusal_line(custom_id: str, refusal: str) -> str:
  obj = {
      'custom_id': custom_id,
      'response': {
          'status_code': 200,
          'body': {
              'choices': [
                  {'message': {'content': None, 'refusal': refusal}},
              ]
          },
      },
      'error': None,
  }
  return json.dumps(obj)


class OpenAIBatchConfigTest(absltest.TestCase):

  def test_default_timeout_covers_completion_window(self):
    cfg = openai_batch.BatchConfig(enabled=True)

    self.assertGreater(cfg.timeout, 24 * 60 * 60)

  def test_from_dict_accepts_batch_config(self):
    cfg = openai_batch.BatchConfig(enabled=True, threshold=1)

    self.assertIs(openai_batch.BatchConfig.from_dict(cfg), cfg)

  def test_from_dict_accepts_boolean_shorthand(self):
    enabled = openai_batch.BatchConfig.from_dict(True)
    disabled = openai_batch.BatchConfig.from_dict(False)

    self.assertTrue(enabled.enabled)
    self.assertFalse(disabled.enabled)

  def test_from_dict_rejects_invalid_type(self):
    with self.assertRaisesRegex(TypeError, 'batch must be a mapping'):
      openai_batch.BatchConfig.from_dict([])

  def test_infer_batch_rejects_invalid_batch_size(self):
    cfg = openai_batch.BatchConfig(enabled=True, threshold=1)

    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, 'batch_size must be > 0'
    ):
      openai_batch.infer_batch(
          client=_FakeClient(),
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
          batch_size=0,
      )


class OpenAIBatchHelperTest(parameterized.TestCase):

  @mock.patch(
      'langextract.providers.openai_batch.time.sleep', return_value=None
  )
  def test_orders_results_by_custom_id(self, _mock_sleep):
    client = _FakeClient()

    client.batches.push_retrieve(py_types.SimpleNamespace(status='in_progress'))
    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )

    out = '\n'.join([
        _make_output_line('idx-000001', 'B'),
        _make_output_line('idx-000000', 'A'),
    ])
    client.files.set_content('out-1', out)

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    res = openai_batch.infer_batch(
        client=client,
        model_id='gpt-test',
        prompts=['p0', 'p1'],
        cfg=cfg,
        request_builder=lambda p: {
            'model': 'gpt-test',
            'messages': [{'role': 'user', 'content': p}],
        },
    )

    self.assertEqual(res, ['A', 'B'])

  @mock.patch(
      'langextract.providers.openai_batch.time.sleep', return_value=None
  )
  def test_splits_jobs_by_batch_size(self, _mock_sleep):
    client = _FakeClient()

    for job_idx in range(3):
      client.batches.push_retrieve(
          py_types.SimpleNamespace(status='in_progress')
      )
      client.batches.push_retrieve(
          py_types.SimpleNamespace(
              status='completed', output_file_id=f'out-{job_idx}'
          )
      )

    client.files.set_content(
        'out-0',
        '\n'.join([
            _make_output_line('idx-000000', '0'),
            _make_output_line('idx-000001', '1'),
        ]),
    )
    client.files.set_content(
        'out-1',
        '\n'.join([
            _make_output_line('idx-000002', '2'),
            _make_output_line('idx-000003', '3'),
        ]),
    )
    client.files.set_content(
        'out-2',
        _make_output_line('idx-000004', '4'),
    )

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
        max_requests_per_job=100,
    )

    prompts = ['p0', 'p1', 'p2', 'p3', 'p4']
    res = openai_batch.infer_batch(
        client=client,
        model_id='gpt-test',
        prompts=prompts,
        cfg=cfg,
        request_builder=lambda p: {
            'model': 'gpt-test',
            'messages': [{'role': 'user', 'content': p}],
        },
        batch_size=2,
    )

    self.assertEqual(res, ['0', '1', '2', '3', '4'])
    self.assertLen(client.batches.created, 3)
    self.assertLen(client.files.created, 3)

  def test_metadata_and_job_create_hook_are_used(self):
    client = _FakeClient()
    created_jobs = []

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )
    client.files.set_content('out-1', _make_output_line('idx-000000', 'ok'))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
        metadata={'purpose': 'test'},
        on_job_create=created_jobs.append,
    )

    res = openai_batch.infer_batch(
        client=client,
        model_id='gpt-test',
        prompts=['p0'],
        cfg=cfg,
        request_builder=lambda p: {
            'model': 'gpt-test',
            'messages': [{'role': 'user', 'content': p}],
        },
    )

    self.assertEqual(res, ['ok'])
    self.assertEqual(client.batches.created[0]['metadata'], {'purpose': 'test'})
    self.assertLen(created_jobs, 1)
    self.assertEqual(created_jobs[0].id, 'batch-1')

  def test_item_error_raises(self):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )

    obj = {
        'custom_id': 'idx-000000',
        'error': {'message': 'boom'},
        'response': None,
    }
    client.files.set_content('out-1', json.dumps(obj))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, 'per-item errors'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

  def test_job_create_failure_deletes_uploaded_input_file(self):
    client = _FakeClient()
    client.batches.create = mock.Mock(side_effect=RuntimeError('boom'))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, 'job create failed'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

    self.assertEqual(client.files.deleted, ['file-1'])

  def test_failed_job_reports_errors_field(self):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(
            status='failed',
            errors={'data': [{'message': 'invalid request'}]},
        )
    )

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, 'invalid request'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

  @parameterized.named_parameters(
      dict(testcase_name='failed', status='failed'),
      dict(testcase_name='expired', status='expired'),
      dict(testcase_name='cancelled', status='cancelled'),
  )
  def test_terminal_job_status_reports_status(self, status):
    client = _FakeClient()

    client.batches.push_retrieve(py_types.SimpleNamespace(status=status))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, f'status={status}'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

  def test_completed_job_missing_output_reports_custom_id(self):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )
    client.files.set_content('out-1', _make_output_line('idx-000000', 'ok'))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, 'custom_id=idx-000001'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0', 'p1'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

  def test_completed_job_without_output_or_error_file_raises(self):
    client = _FakeClient()

    client.batches.push_retrieve(py_types.SimpleNamespace(status='completed'))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, 'no output_file_id or error_file_id'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

  def test_unexpected_custom_id_logs_warning(self):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )
    client.files.set_content('out-1', _make_output_line('bad-id', 'ok'))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertLogs(level='WARNING') as logs:
      with self.assertRaisesRegex(
          exceptions.InferenceRuntimeError, 'custom_id=idx-000000'
      ):
        openai_batch.infer_batch(
            client=client,
            model_id='gpt-test',
            prompts=['p0'],
            cfg=cfg,
            request_builder=lambda p: {
                'model': 'gpt-test',
                'messages': [{'role': 'user', 'content': p}],
            },
        )

    self.assertIn('unexpected custom_id', '\n'.join(logs.output))

  @mock.patch(
      'langextract.providers.openai_batch.time.sleep', return_value=None
  )
  def test_output_download_retries_transient_forbidden(self, mock_sleep):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )
    client.files.content = mock.Mock(
        side_effect=[
            _StatusError(403),
            py_types.SimpleNamespace(
                text=_make_output_line('idx-000000', 'ok')
            ),
        ]
    )

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    res = openai_batch.infer_batch(
        client=client,
        model_id='gpt-test',
        prompts=['p0'],
        cfg=cfg,
        request_builder=lambda p: {
            'model': 'gpt-test',
            'messages': [{'role': 'user', 'content': p}],
        },
    )

    self.assertEqual(res, ['ok'])
    self.assertEqual(client.files.content.call_count, 2)
    mock_sleep.assert_called_once_with(1)

  @mock.patch(
      'langextract.providers.openai_batch.time.sleep', return_value=None
  )
  def test_output_download_forbidden_mentions_files_permission(
      self, mock_sleep
  ):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )
    client.files.content = mock.Mock(side_effect=_StatusError(403))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, 'Files Read permission'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )
    self.assertEqual(client.files.content.call_count, 3)
    self.assertEqual(mock_sleep.call_count, 2)

  def test_completed_job_reads_error_file(self):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(
            status='completed',
            output_file_id='out-1',
            error_file_id='err-1',
        )
    )
    client.files.set_content('out-1', _make_output_line('idx-000000', 'ok'))
    client.files.set_content(
        'err-1', _make_error_line('idx-000001', 'rate limited', 429)
    )

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, 'rate limited'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0', 'p1'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

  def test_non_2xx_response_raises_item_error(self):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )
    client.files.set_content(
        'out-1', _make_error_line('idx-000000', 'rate limited', 429)
    )

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, 'status_code=429'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

  def test_refusal_message_is_reported(self):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )
    client.files.set_content(
        'out-1', _make_refusal_line('idx-000000', 'cannot comply')
    )

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceRuntimeError, 'cannot comply'
    ):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

  @mock.patch('langextract.providers.openai_batch.time.time')
  def test_timeout_cancels_job(self, mock_time):
    # The second clock read crosses the timeout immediately, without sleeping.
    mock_time.side_effect = [0, 10]
    client = _FakeClient()

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window='24h',
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaisesRegex(exceptions.InferenceRuntimeError, 'timed out'):
      openai_batch.infer_batch(
          client=client,
          model_id='gpt-test',
          prompts=['p0'],
          cfg=cfg,
          request_builder=lambda p: {
              'model': 'gpt-test',
              'messages': [{'role': 'user', 'content': p}],
          },
      )

    self.assertEqual(client.batches.cancelled, ['batch-1'])

  @mock.patch(
      'langextract.providers.openai_batch.time.sleep', return_value=None
  )
  def test_default_completion_window_is_sent(self, _mock_sleep):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status='completed', output_file_id='out-1')
    )
    client.files.set_content('out-1', _make_output_line('idx-000000', 'ok'))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        poll_interval=1,
        timeout=5,
    )

    res = openai_batch.infer_batch(
        client=client,
        model_id='gpt-test',
        prompts=['p0'],
        cfg=cfg,
        request_builder=lambda p: {
            'model': 'gpt-test',
            'messages': [{'role': 'user', 'content': p}],
        },
    )

    self.assertEqual(res, ['ok'])
    self.assertLen(client.batches.created, 1)
    self.assertEqual(client.batches.created[0]['completion_window'], '24h')

  def test_unsupported_completion_window_raises(self):
    with self.assertRaisesRegex(ValueError, 'completion_window'):
      openai_batch.BatchConfig(enabled=True, completion_window='1h')


if __name__ == '__main__':
  absltest.main()
