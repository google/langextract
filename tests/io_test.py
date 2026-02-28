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

"""Tests for langextract.io module — progress bar resource leak fix."""

import pathlib
import unittest
from unittest import mock

import requests

from langextract import io as io_lib
from langextract.core import data


def _make_annotated_doc(doc_id, text="sample"):
  """Helper to create an AnnotatedDocument with minimal fields."""
  return data.AnnotatedDocument(
      document_id=doc_id,
      text=text,
      extractions=[
          data.Extraction("label", "entity"),
      ],
  )


def _failing_generator(docs, fail_after=1):
  """Yields docs then raises RuntimeError after fail_after items."""
  count = 0
  for doc in docs:
    if count >= fail_after:
      raise RuntimeError("generator failed mid-iteration")
    yield doc
    count += 1


class SaveAnnotatedDocumentsProgressBarTest(unittest.TestCase):
  """Verify progress_bar.close() is called even when an exception occurs."""

  @mock.patch("langextract.progress.create_save_progress_bar")
  def test_progress_bar_closed_on_exception(self, mock_create_bar):
    """progress_bar.close() must be called when the generator raises."""
    mock_bar = mock.MagicMock()
    mock_create_bar.return_value = mock_bar

    docs = [_make_annotated_doc("doc1"), _make_annotated_doc("doc2")]
    failing_iter = _failing_generator(docs, fail_after=1)

    with self.assertRaises(RuntimeError):
      io_lib.save_annotated_documents(
          annotated_documents=failing_iter,
          output_dir=pathlib.Path("/tmp/test_io_issue401"),
          show_progress=False,
      )

    # The critical assertion: close() must be called despite the exception.
    mock_bar.close.assert_called_once()

  @mock.patch("langextract.progress.create_save_progress_bar")
  def test_progress_bar_closed_on_success(self, mock_create_bar):
    """progress_bar.close() is still called on a successful run."""
    mock_bar = mock.MagicMock()
    mock_create_bar.return_value = mock_bar

    docs = [_make_annotated_doc("doc1"), _make_annotated_doc("doc2")]

    io_lib.save_annotated_documents(
        annotated_documents=iter(docs),
        output_dir=pathlib.Path("/tmp/test_io_issue401"),
        show_progress=False,
    )

    mock_bar.close.assert_called_once()


class DownloadTextFromUrlProgressBarTest(unittest.TestCase):
  """Verify progress_bar.close() is called on network errors mid-download."""

  @mock.patch("langextract.progress.create_download_progress_bar")
  @mock.patch("requests.get")
  def test_progress_bar_closed_on_network_error(
      self, mock_get, mock_create_bar
  ):
    """progress_bar.close() must be called when iter_content raises."""
    mock_bar = mock.MagicMock()
    mock_create_bar.return_value = mock_bar

    # Build a mock response whose iter_content raises mid-stream.
    mock_response = mock.MagicMock()
    mock_response.headers = {
        "Content-Type": "text/html",
        "Content-Length": "1024",
    }
    mock_response.raise_for_status = mock.MagicMock()

    def _exploding_iter(chunk_size=8192):  # pylint: disable=unused-argument
      yield b"partial data"
      raise requests.ConnectionError("connection reset")

    mock_response.iter_content = _exploding_iter
    mock_get.return_value = mock_response

    with self.assertRaises(requests.RequestException):
      io_lib.download_text_from_url(
          "https://example.com/large.txt",
          show_progress=True,
      )

    # The critical assertion: close() must be called despite the error.
    mock_bar.close.assert_called_once()

  @mock.patch("langextract.progress.create_download_progress_bar")
  @mock.patch("requests.get")
  def test_progress_bar_closed_on_success(self, mock_get, mock_create_bar):
    """progress_bar.close() is still called on a successful download."""
    mock_bar = mock.MagicMock()
    mock_create_bar.return_value = mock_bar

    mock_response = mock.MagicMock()
    mock_response.headers = {
        "Content-Type": "text/plain",
        "Content-Length": "11",
    }
    mock_response.raise_for_status = mock.MagicMock()
    mock_response.iter_content = mock.MagicMock(
        return_value=iter([b"hello world"])
    )
    mock_get.return_value = mock_response

    result = io_lib.download_text_from_url(
        "https://example.com/small.txt",
        show_progress=True,
    )

    self.assertEqual(result, "hello world")
    mock_bar.close.assert_called_once()


if __name__ == "__main__":
  unittest.main()
