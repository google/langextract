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

import csv
import pathlib
import tempfile
import unittest
from unittest import mock

import pytest
import requests

from langextract import io
from langextract.core import data


class _ClosingTracker:

  def __init__(self):
    self.closed = False
    self.updates = []

  def update(self, value):
    self.updates.append(value)

  def close(self):
    self.closed = True


class IoTest(unittest.TestCase):

  def test_save_annotated_documents_closes_progress_bar_on_generator_error(
      self,
  ):
    tracker = _ClosingTracker()

    def annotated_documents():
      yield data.AnnotatedDocument(
          document_id='doc-1', text='hello', extractions=[]
      )
      raise RuntimeError('boom')

    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch(
          'langextract.io.progress.create_save_progress_bar',
          return_value=tracker,
      ):
        with self.assertRaisesRegex(RuntimeError, 'boom'):
          io.save_annotated_documents(
              annotated_documents(),
              output_dir=pathlib.Path(tmpdir),
              show_progress=True,
          )

    self.assertTrue(tracker.closed)
    self.assertEqual(tracker.updates, [1])

  def test_download_text_from_url_closes_progress_bar_on_stream_error(self):
    tracker = _ClosingTracker()
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.headers = {
        'Content-Type': 'text/plain',
        'Content-Length': '10',
    }
    response.raise_for_status.return_value = None

    def broken_iter_content(*, chunk_size):
      del chunk_size
      yield b'hello'
      raise requests.RequestException('connection reset')

    response.iter_content.side_effect = broken_iter_content

    with mock.patch('langextract.io.requests.get', return_value=response):
      with mock.patch(
          'langextract.io.progress.create_download_progress_bar',
          return_value=tracker,
      ):
        with self.assertRaisesRegex(
            requests.RequestException, 'connection reset'
        ):
          io.download_text_from_url('https://example.com/file.txt')

    self.assertTrue(tracker.closed)
    self.assertEqual(tracker.updates, [5])


@pytest.mark.parametrize(
    'value', ['ordinary', 'NA', 'N/A', 'NULL', 'NaN', 'null', '']
)
@pytest.mark.parametrize('column', ['id', 'text'])
def test_dataset_preserves_literal_strings(tmp_path, value, column):
  path = tmp_path / 'input.csv'
  row = {'id': '0001', 'text': 'ordinary'}
  row[column] = value
  with path.open('w', encoding='utf-8', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=['id', 'text'])
    writer.writeheader()
    writer.writerow(row)
  documents = list(io.Dataset(path, id_key='id', text_key='text').load())
  assert len(documents) == 1
  assert documents[0].document_id == row['id']
  assert documents[0].text == row['text']
  assert documents[0].tokenized_text.text == row['text']


def test_dataset_preserves_custom_delimiter(tmp_path):
  path = tmp_path / 'input.csv'
  path.write_text('id|text\n0001|NULL\n', encoding='utf-8')
  document = next(
      io.Dataset(path, id_key='id', text_key='text').load(delimiter='|')
  )
  assert document.document_id == '0001'
  assert document.text == 'NULL'


if __name__ == '__main__':
  unittest.main()
