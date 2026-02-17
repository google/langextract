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

"""Tests for OpenAI Batch API support."""

from __future__ import annotations

import json
from unittest import mock

from absl.testing import absltest

from langextract.core import exceptions
from langextract.providers import openai_batch


def _batch_line(*, custom_id: str, content: str | None, error: dict | None):
  response = None
  if content is not None:
    response = {
        "status_code": 200,
        "request_id": "req_x",
        "body": {
            "choices": [{"message": {"role": "assistant", "content": content}}]
        },
    }
  return {
      "id": "batch_req_x",
      "custom_id": custom_id,
      "response": response,
      "error": error,
  }


class OpenAIBatchTest(absltest.TestCase):

  def test_infer_batch_preserves_order_with_out_of_order_output(self):
    client = mock.Mock()
    client.files.create.return_value = mock.Mock(id="file_1")
    client.batches.create.return_value = mock.Mock(id="batch_1")
    client.batches.retrieve.return_value = mock.Mock(
        id="batch_1",
        status="completed",
        output_file_id="out_1",
        error_file_id=None,
    )

    # Output lines are out of order: idx-1 arrives before idx-0
    output_lines = "\n".join(
        [
            json.dumps(
                _batch_line(custom_id="idx-1", content="out1", error=None)
            ),
            json.dumps(
                _batch_line(custom_id="idx-0", content="out0", error=None)
            ),
            "",
        ]
    ).encode("utf-8")

    client.files.content.return_value = mock.Mock(
        read=mock.Mock(return_value=output_lines)
    )

    cfg = openai_batch.BatchConfig(enabled=True, threshold=2)
    outputs = openai_batch.infer_batch(
        client=client,
        request_bodies=[{"model": "gpt"}, {"model": "gpt"}],
        cfg=cfg,
    )

    self.assertEqual(outputs, ["out0", "out1"])
    client.files.create.assert_called_once()
    client.batches.create.assert_called_once()

  def test_infer_batch_raises_on_item_error_by_default(self):
    client = mock.Mock()
    client.files.create.return_value = mock.Mock(id="file_1")
    client.batches.create.return_value = mock.Mock(id="batch_1")
    client.batches.retrieve.return_value = mock.Mock(
        id="batch_1",
        status="completed",
        output_file_id="out_1",
        error_file_id="err_1",
    )

    # Success for idx-0
    output_lines = (
        json.dumps(_batch_line(custom_id="idx-0", content="ok", error=None))
        + "\n"
    ).encode("utf-8")

    # Error for idx-1
    error_lines = (
        json.dumps(
            _batch_line(
                custom_id="idx-1",
                content=None,
                error={"code": "bad_request", "message": "nope"},
            )
        )
        + "\n"
    ).encode("utf-8")

    def _content(file_id: str):
      if file_id == "out_1":
        return mock.Mock(read=mock.Mock(return_value=output_lines))
      return mock.Mock(read=mock.Mock(return_value=error_lines))

    client.files.content.side_effect = _content

    cfg = openai_batch.BatchConfig(enabled=True, threshold=2)
    with self.assertRaises(exceptions.InferenceRuntimeError) as cm:
      _ = openai_batch.infer_batch(
          client=client,
          request_bodies=[{"model": "gpt"}, {"model": "gpt"}],
          cfg=cfg,
      )

    self.assertIn("custom_id=idx-1", str(cm.exception))
    self.assertIn("nope", str(cm.exception))

  def test_infer_batch_fills_fallback_on_item_error_when_configured(self):
    client = mock.Mock()
    client.files.create.return_value = mock.Mock(id="file_1")
    client.batches.create.return_value = mock.Mock(id="batch_1")
    client.batches.retrieve.return_value = mock.Mock(
        id="batch_1",
        status="completed",
        output_file_id="out_1",
        error_file_id="err_1",
    )

    output_lines = (
        json.dumps(_batch_line(custom_id="idx-0", content="ok", error=None))
        + "\n"
    ).encode("utf-8")
    error_lines = (
        json.dumps(
            _batch_line(
                custom_id="idx-1",
                content=None,
                error={"code": "bad_request", "message": "nope"},
            )
        )
        + "\n"
    ).encode("utf-8")

    def _content(file_id: str):
      if file_id == "out_1":
        return mock.Mock(read=mock.Mock(return_value=output_lines))
      return mock.Mock(read=mock.Mock(return_value=error_lines))

    client.files.content.side_effect = _content

    cfg = openai_batch.BatchConfig(
        enabled=True, threshold=2, ignore_item_errors=True
    )
    outputs = openai_batch.infer_batch(
        client=client,
        request_bodies=[{"model": "gpt"}, {"model": "gpt"}],
        cfg=cfg,
        fallback_output='{"extractions": []}',
    )

    self.assertEqual(outputs, ["ok", '{"extractions": []}'])


if __name__ == "__main__":
  absltest.main()

