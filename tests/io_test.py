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

"""Tests for langextract.io module."""

import pathlib
import tempfile
from unittest import mock

import pytest
import requests

from langextract import io
from langextract.core import data


def test_save_annotated_documents_closes_progress_bar_on_exception():
  """save_annotated_documents closes the progress bar on iteration errors."""
  progress_bar = mock.Mock()

  def annotated_documents():
    yield data.AnnotatedDocument(document_id="doc-1", text="hello")
    raise RuntimeError("boom")

  with tempfile.TemporaryDirectory() as tmpdir:
    with mock.patch.object(
        io.progress,
        "create_save_progress_bar",
        return_value=progress_bar,
    ):
      with pytest.raises(RuntimeError, match="boom"):
        io.save_annotated_documents(
            annotated_documents(),
            output_dir=pathlib.Path(tmpdir),
            show_progress=True,
        )

  progress_bar.update.assert_called_once_with(1)
  progress_bar.close.assert_called_once_with()


def test_download_text_from_url_closes_progress_bar_on_stream_exception():
  """download_text_from_url closes the progress bar on stream errors."""
  progress_bar = mock.Mock()
  response = mock.Mock()
  response.headers = {
      "Content-Type": "text/plain",
      "Content-Length": "10",
  }
  response.raise_for_status.return_value = None

  def iter_content(chunk_size):
    del chunk_size
    yield b"hello"
    raise requests.RequestException("stream failed")

  response.iter_content.side_effect = iter_content

  with mock.patch.object(io.requests, "get", return_value=response):
    with mock.patch.object(
        io.progress,
        "create_download_progress_bar",
        return_value=progress_bar,
    ):
      with pytest.raises(requests.RequestException, match="stream failed"):
        io.download_text_from_url("https://example.com/file.txt")

  progress_bar.update.assert_called_once_with(5)
  progress_bar.close.assert_called_once_with()
