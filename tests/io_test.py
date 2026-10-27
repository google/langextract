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
import unittest
from unittest import mock

from absl.testing import parameterized
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


def _load_csv(csv_text: str, delimiter: str = ',') -> list[data.Document]:
  with tempfile.TemporaryDirectory() as tmpdir:
    path = pathlib.Path(tmpdir) / 'input.csv'
    path.write_text(csv_text, encoding='utf-8')
    return list(
        io.Dataset(path, id_key='id', text_key='text').load(delimiter=delimiter)
    )


class DatasetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='na', value='NA'),
      dict(testcase_name='n_slash_a', value='N/A'),
      dict(testcase_name='null_uppercase', value='NULL'),
      dict(testcase_name='null_lowercase', value='null'),
      dict(testcase_name='nan', value='NaN'),
  )
  def test_load_preserves_literal_strings(self, value):
    documents = _load_csv(f'id,text\n{value},{value}\n')

    self.assertLen(documents, 1)
    self.assertEqual(documents[0].document_id, value)
    self.assertEqual(documents[0].text, value)
    self.assertEqual(documents[0].tokenized_text.text, value)

  @parameterized.named_parameters(
      dict(testcase_name='empty_field', csv_text='id,text\n0001,\n'),
      dict(testcase_name='missing_field', csv_text='id,text\n0001\n'),
  )
  def test_load_reads_empty_or_missing_text_as_empty_string(self, csv_text):
    documents = _load_csv(csv_text)

    self.assertLen(documents, 1)
    self.assertEqual(documents[0].document_id, '0001')
    self.assertEqual(documents[0].text, '')
    self.assertEmpty(documents[0].tokenized_text.tokens)

  def test_load_generates_unique_document_ids_for_empty_ids(self):
    documents = _load_csv('id,text\n,first\n,second\n')

    self.assertLen(documents, 2)
    self.assertStartsWith(documents[0].document_id, 'doc_')
    self.assertStartsWith(documents[1].document_id, 'doc_')
    self.assertNotEqual(documents[0].document_id, documents[1].document_id)

  def test_load_with_custom_delimiter_preserves_literal_strings(self):
    documents = _load_csv('id|text\n0001|NULL\n', delimiter='|')

    self.assertLen(documents, 1)
    self.assertEqual(documents[0].document_id, '0001')
    self.assertEqual(documents[0].text, 'NULL')


if __name__ == '__main__':
  unittest.main()
