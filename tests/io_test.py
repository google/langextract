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

import requests

from langextract import io
import langextract as lx
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
    response.status_code = 200
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

    # allow_internal_urls=True so the test doesn't depend on DNS resolution
    # for example.com in the test environment.
    with mock.patch('langextract.io.requests.get', return_value=response):
      with mock.patch(
          'langextract.io.progress.create_download_progress_bar',
          return_value=tracker,
      ):
        with self.assertRaisesRegex(
            requests.RequestException, 'connection reset'
        ):
          io.download_text_from_url(
              'https://example.com/file.txt', allow_internal_urls=True
          )

    self.assertTrue(tracker.closed)
    self.assertEqual(tracker.updates, [5])


class SsrfValidationTest(unittest.TestCase):
  """_is_internal_hostname / _validate_url_not_internal coverage."""

  def _assert_internal(self, hostname: str):
    self.assertTrue(
        io._is_internal_hostname(hostname),
        msg=f'{hostname!r} should be flagged internal',
    )

  def _assert_not_internal(self, hostname: str):
    self.assertFalse(
        io._is_internal_hostname(hostname),
        msg=f'{hostname!r} should NOT be flagged internal',
    )

  def test_literal_internal_hostnames(self):
    for host in ['localhost', '127.0.0.1', '0.0.0.0', '[::1]']:
      self._assert_internal(host)

  def test_private_ipv4_ranges(self):
    for host in ['10.0.0.1', '172.16.0.1', '192.168.1.1']:
      self._assert_internal(host)

  def test_link_local_and_metadata(self):
    # AWS / GCP / Azure IMDS lives in link-local.
    self._assert_internal('169.254.169.254')

  def test_cgnat_range_is_blocked(self):
    # RFC 6598 CGNAT range; Alibaba exposes metadata here.
    self._assert_internal('100.64.0.1')
    self._assert_internal('100.100.100.200')

  def test_ipv4_mapped_ipv6_into_cgnat_is_blocked(self):
    # Without ipv4_mapped normalization in _ip_is_disallowed, these would
    # slip past because `ip in _CGNAT_NETWORK` is False for IPv6Address.
    self._assert_internal('::ffff:100.64.0.1')
    self._assert_internal('::ffff:100.100.100.200')

  def test_ipv4_mapped_ipv6_into_private_is_blocked(self):
    # Sanity: other mapped IPv4 privates still caught via is_private.
    self._assert_internal('::ffff:10.0.0.1')
    self._assert_internal('::ffff:127.0.0.1')

  def test_ipv6_private_is_blocked(self):
    # ULA.
    self._assert_internal('fd00::1')

  def test_suffix_based_internal_names(self):
    for host in ['server.corp', 'host.internal', 'printer.local']:
      self._assert_internal(host)

  def test_public_hostnames_pass(self):
    # These MUST NOT be flagged; 8.8.8.8 is a public DNS server.
    self._assert_not_internal('8.8.8.8')
    # 100.63.x.x and 100.128.x.x are just outside CGNAT and are public.
    self._assert_not_internal('100.63.255.255')
    self._assert_not_internal('100.128.0.1')

  def test_validate_url_rejects_internal(self):
    with self.assertRaises(io.InvalidUrlError):
      io._validate_url_not_internal('http://169.254.169.254/latest/meta-data/')
    with self.assertRaises(io.InvalidUrlError):
      io._validate_url_not_internal('http://100.100.100.200/')
    with self.assertRaises(io.InvalidUrlError):
      io._validate_url_not_internal('http://localhost:8080/')


class DownloadRedirectValidationTest(unittest.TestCase):
  """Redirect hops must be validated to close the SSRF bypass."""

  def _mock_response(self, *, status_code, headers=None, body=b''):
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.status_code = status_code
    response.headers = headers or {}
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter([body])
    return response

  def test_redirect_to_internal_blocked(self):
    # Public URL 302s to AWS IMDS; redirect hop must be validated.
    redirect = self._mock_response(
        status_code=302,
        headers={'Location': 'http://169.254.169.254/latest/meta-data/'},
    )

    with mock.patch(
        'langextract.io.requests.get', return_value=redirect
    ) as get:
      with self.assertRaises(io.InvalidUrlError):
        # allow_internal_urls=True on origin would skip origin check but we
        # want to show the redirect-hop check even works when origin passes.
        # We pass a safe public-looking origin and let the mock return a 302.
        io.download_text_from_url('https://public.example.com/redir')
      # Origin GET was issued once; loop raised on the internal Location.
      self.assertGreaterEqual(get.call_count, 1)

  def test_allow_internal_urls_skips_all_validation(self):
    ok = self._mock_response(
        status_code=200,
        headers={'Content-Type': 'text/plain', 'Content-Length': '5'},
        body=b'hello',
    )
    with mock.patch('langextract.io.requests.get', return_value=ok):
      # localhost origin + allow_internal_urls should be permitted.
      result = io.download_text_from_url(
          'http://localhost:8080/file', allow_internal_urls=True
      )
    self.assertEqual(result, 'hello')


class SaveAnnotatedDocumentsPathTest(unittest.TestCase):
  """output_name must reject directory components (no silent strip)."""

  def _single_doc(self):
    return iter([
        data.AnnotatedDocument(
            document_id='doc-1', text='hello', extractions=[]
        )
    ])

  def test_rejects_traversal(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaises(IOError):
        io.save_annotated_documents(
            self._single_doc(),
            output_dir=pathlib.Path(tmpdir),
            output_name='../escape.jsonl',
            show_progress=False,
        )

  def test_rejects_subdirectory_component(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaises(IOError):
        io.save_annotated_documents(
            self._single_doc(),
            output_dir=pathlib.Path(tmpdir),
            output_name='subdir/data.jsonl',
            show_progress=False,
        )

  def test_rejects_absolute_output_name(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaises(IOError):
        io.save_annotated_documents(
            self._single_doc(),
            output_dir=pathlib.Path(tmpdir),
            output_name='/etc/data.jsonl',
            show_progress=False,
        )

  def test_plain_filename_accepted(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      io.save_annotated_documents(
          self._single_doc(),
          output_dir=pathlib.Path(tmpdir),
          output_name='data.jsonl',
          show_progress=False,
      )
      self.assertTrue((pathlib.Path(tmpdir) / 'data.jsonl').exists())


class ExtractAllowInternalUrlsSmokeTest(unittest.TestCase):
  """Lock in that extract() forwards allow_internal_urls to the fetcher."""

  def test_allow_internal_urls_is_forwarded(self):
    sentinel = RuntimeError('short-circuit from mocked downloader')

    def fake_download(url, **kwargs):
      # Assert the kwarg we care about was threaded through; then abort so
      # we don't need a real model / examples for the rest of extract().
      self.assertTrue(kwargs.get('allow_internal_urls'))
      raise sentinel

    with mock.patch(
        'langextract.io.download_text_from_url', side_effect=fake_download
    ):
      with self.assertRaises(RuntimeError) as cm:
        lx.extract(
            text_or_documents='http://localhost:8080/doc',
            prompt_description='x',
            examples=[
                lx.data.ExampleData(
                    text='hi',
                    extractions=[
                        lx.data.Extraction(
                            extraction_class='thing', extraction_text='hi'
                        )
                    ],
                )
            ],
            model_id='gemini-2.5-flash',
            api_key='fake',
            allow_internal_urls=True,
        )
    self.assertIs(cm.exception, sentinel)


if __name__ == '__main__':
  unittest.main()
