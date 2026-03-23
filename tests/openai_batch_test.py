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

from __future__ import annotations

import json
import types as py_types
import unittest
from unittest import mock

from langextract.core import exceptions
from langextract.providers import openai_batch


class _FakeFiles:

  def __init__(self):
    self.created = []
    self._content_by_id = {}

  def create(self, *, file, purpose):
    self.created.append({"file": file, "purpose": purpose})
    return py_types.SimpleNamespace(id=f"file-{len(self.created)}")

  def content(self, file_id):
    return py_types.SimpleNamespace(text=self._content_by_id[file_id])

  def set_content(self, file_id: str, text: str) -> None:
    self._content_by_id[file_id] = text


class _FakeBatches:

  def __init__(self):
    self.created = []
    self._retrieve_queue = []

  def create(self, **kwargs):
    self.created.append(kwargs)
    return py_types.SimpleNamespace(id=f"batch-{len(self.created)}")

  def retrieve(self, batch_id):
    if not self._retrieve_queue:
      raise RuntimeError("retrieve queue empty")
    return self._retrieve_queue.pop(0)

  def push_retrieve(self, obj):
    self._retrieve_queue.append(obj)


class _FakeClient:

  def __init__(self):
    self.files = _FakeFiles()
    self.batches = _FakeBatches()


def _make_output_line(custom_id: str, content: str) -> str:
  obj = {
      "custom_id": custom_id,
      "response": {
          "body": {
              "choices": [
                  {"message": {"content": content}},
              ]
          }
      },
      "error": None,
  }
  return json.dumps(obj)


class OpenAIBatchHelperTest(unittest.TestCase):

  @mock.patch(
      "langextract.providers.openai_batch.time.sleep", return_value=None
  )
  def test_orders_results_by_custom_id(self, _mock_sleep):
    client = _FakeClient()

    # Job status: in_progress -> completed.
    client.batches.push_retrieve(py_types.SimpleNamespace(status="in_progress"))
    client.batches.push_retrieve(
        py_types.SimpleNamespace(status="completed", output_file_id="out-1")
    )

    # Output JSONL is intentionally out of order.
    out = "\n".join([
        _make_output_line("idx-000001", "B"),
        _make_output_line("idx-000000", "A"),
    ])
    client.files.set_content("out-1", out)

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window="24h",
        poll_interval=1,
        timeout=5,
    )

    res = openai_batch.infer_batch(
        client=client,
        model_id="gpt-test",
        prompts=["p0", "p1"],
        cfg=cfg,
        request_builder=lambda p: {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": p}],
        },
    )

    self.assertEqual(res, ["A", "B"])

  @mock.patch(
      "langextract.providers.openai_batch.time.sleep", return_value=None
  )
  def test_splits_jobs_by_batch_size(self, _mock_sleep):
    client = _FakeClient()

    # For 3 jobs we will do: [in_progress, completed] x 3
    for job_idx in range(3):
      client.batches.push_retrieve(
          py_types.SimpleNamespace(status="in_progress")
      )
      client.batches.push_retrieve(
          py_types.SimpleNamespace(
              status="completed", output_file_id=f"out-{job_idx}"
          )
      )

    # Each job returns exactly the lines for its indices.
    client.files.set_content(
        "out-0",
        "\n".join([
            _make_output_line("idx-000000", "0"),
            _make_output_line("idx-000001", "1"),
        ]),
    )
    client.files.set_content(
        "out-1",
        "\n".join([
            _make_output_line("idx-000002", "2"),
            _make_output_line("idx-000003", "3"),
        ]),
    )
    client.files.set_content(
        "out-2",
        _make_output_line("idx-000004", "4"),
    )

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window="24h",
        poll_interval=1,
        timeout=5,
        max_requests_per_job=100,
    )

    prompts = ["p0", "p1", "p2", "p3", "p4"]
    res = openai_batch.infer_batch(
        client=client,
        model_id="gpt-test",
        prompts=prompts,
        cfg=cfg,
        request_builder=lambda p: {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": p}],
        },
        batch_size=2,
    )

    self.assertEqual(res, ["0", "1", "2", "3", "4"])
    self.assertEqual(len(client.batches.created), 3)
    self.assertEqual(len(client.files.created), 3)

  def test_item_error_raises(self):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status="completed", output_file_id="out-1")
    )

    obj = {
        "custom_id": "idx-000000",
        "error": {"message": "boom"},
        "response": None,
    }
    client.files.set_content("out-1", json.dumps(obj))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window="24h",
        poll_interval=1,
        timeout=5,
    )

    with self.assertRaises(exceptions.InferenceRuntimeError):
      _ = openai_batch.infer_batch(
          client=client,
          model_id="gpt-test",
          prompts=["p0"],
          cfg=cfg,
          request_builder=lambda p: {
              "model": "gpt-test",
              "messages": [{"role": "user", "content": p}],
          },
      )

  @mock.patch(
      "langextract.providers.openai_batch.time.sleep", return_value=None
  )
  def test_completion_window_is_optional(self, _mock_sleep):
    client = _FakeClient()

    client.batches.push_retrieve(
        py_types.SimpleNamespace(status="completed", output_file_id="out-1")
    )
    client.files.set_content("out-1", _make_output_line("idx-000000", "ok"))

    cfg = openai_batch.BatchConfig(
        enabled=True,
        threshold=1,
        completion_window=None,
        poll_interval=1,
        timeout=5,
    )

    res = openai_batch.infer_batch(
        client=client,
        model_id="gpt-test",
        prompts=["p0"],
        cfg=cfg,
        request_builder=lambda p: {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": p}],
        },
    )

    self.assertEqual(res, ["ok"])
    self.assertEqual(len(client.batches.created), 1)
    self.assertNotIn("completion_window", client.batches.created[0])


if __name__ == "__main__":
  unittest.main()
