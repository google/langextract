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

"""Supports Input and Output Operations for Data Annotations."""
from __future__ import annotations

import abc
import dataclasses
import ipaddress
import json
import os
import pathlib
import socket
from typing import Any, Iterator
from urllib import parse as urlparse

import pandas as pd
import requests

from langextract import data_lib
from langextract import progress
from langextract.core import data
from langextract.core import exceptions

DEFAULT_TIMEOUT_SECONDS = 30


class InvalidUrlError(exceptions.LangExtractError):
  """Error raised when a URL is invalid or points to a disallowed address."""


# Cloud-provider metadata endpoints that fall outside the standard private
# / loopback / link-local ranges and must be blocked explicitly.
_METADATA_HOSTS = frozenset({
    # AWS / GCP / Azure IMDS (also link-local so covered by is_link_local,
    # kept here for documentation).
    '169.254.169.254',
})

# 100.64.0.0/10 is the RFC 6598 carrier-grade NAT (CGNAT) range; Python's
# ipaddress module does NOT classify it as private, yet at least one major
# cloud (Alibaba) exposes its metadata service inside it (100.100.100.200).
_CGNAT_NETWORK = ipaddress.ip_network('100.64.0.0/10')

_INTERNAL_LITERAL_HOSTNAMES = frozenset({
    'localhost',
    '0.0.0.0',
    '[::1]',
    '[::]',
    '[::ffff:127.0.0.1]',
})

_INTERNAL_SUFFIXES = (
    '.local',
    '.localhost',
    '.internal',
    '.home',
    '.lan',
    '.corp',
    '.intra',
    '.intranet',
)


def _ip_is_disallowed(ip: ipaddress._BaseAddress) -> bool:
  """Return True if `ip` belongs to a range we refuse to fetch from."""
  # Normalize IPv4-mapped IPv6 (e.g. ::ffff:100.100.100.200) to the
  # underlying IPv4 address before running the predicate. Without this,
  # ``ip in _CGNAT_NETWORK`` is always False for an IPv6Address, which
  # would let mapped forms of the CGNAT range slip past.
  if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
    ip = ip.ipv4_mapped
  return (
      ip.is_loopback
      or ip.is_private
      or ip.is_link_local
      or ip.is_multicast
      or ip.is_unspecified
      or ip.is_reserved
      or ip in _CGNAT_NETWORK
  )


def _is_internal_hostname(  # pylint: disable=too-many-return-statements
    hostname: str,
) -> bool:
  """Check whether `hostname` resolves to an internal / reserved address.

  This is best-effort. Callers should treat a True return as authoritative
  ("do not fetch") but a False return as non-authoritative: DNS rebinding
  remains possible because the HTTP client re-resolves at request time.
  See `download_text_from_url` for the redirect-hop validation that covers
  the common bypass path.

  Args:
    hostname: The hostname extracted from a URL.

  Returns:
    True if the hostname is a known-internal literal, resolves to an
    internal IP, or matches an internal-domain suffix.
  """
  if not hostname:
    return False

  hostname_lower = hostname.lower()
  if hostname_lower in _INTERNAL_LITERAL_HOSTNAMES:
    return True

  if hostname in _METADATA_HOSTS:
    return True

  # Literal IP in the hostname.
  try:
    ip = ipaddress.ip_address(hostname)
    return _ip_is_disallowed(ip)
  except ValueError:
    pass  # not a literal IP, fall through to suffix + DNS checks

  # Suffix check runs before DNS so that obviously-internal names like
  # "server.corp" or "printer.local" don't leak to the resolver first.
  if any(hostname_lower.endswith(suffix) for suffix in _INTERNAL_SUFFIXES):
    return True

  # Partial DNS-rebinding mitigation: if the name resolves (right now) to an
  # internal address, refuse upfront. A time-of-check/time-of-use race still
  # exists because `requests` will re-resolve when making the request.
  try:
    resolved = socket.getaddrinfo(hostname, None)
  except (socket.gaierror, ValueError):
    resolved = []
  for result in resolved:
    try:
      ip_obj = ipaddress.ip_address(result[4][0])
    except ValueError:
      continue
    if _ip_is_disallowed(ip_obj):
      return True

  return False


def _validate_url_not_internal(url: str) -> None:
  """Validate that `url` does not point to an internal / reserved address.

  Args:
    url: The URL to validate.

  Raises:
    InvalidUrlError: If the URL has no hostname, fails to parse, or points
      to an address covered by `_is_internal_hostname`.
  """
  try:
    result = urlparse.urlparse(url)
    hostname = result.hostname
  except (ValueError, AttributeError) as e:
    raise InvalidUrlError(f'Invalid URL {url}: {e}') from e

  if not hostname:
    raise InvalidUrlError(f'URL has no hostname: {url}')

  if _is_internal_hostname(hostname):
    raise InvalidUrlError(
        f'URL {url} points to an internal or reserved address '
        f'({hostname}); pass allow_internal_urls=True to override.'
    )


class InvalidDatasetError(exceptions.LangExtractError):
  """Error raised when Dataset is empty or invalid."""


@dataclasses.dataclass(frozen=True)
class Dataset(abc.ABC):
  """A dataset for inputs to LLM Labeler."""

  input_path: pathlib.Path
  id_key: str
  text_key: str

  def load(self, delimiter: str = ',') -> Iterator[data.Document]:
    """Loads the dataset from a CSV file.

    Args:
      delimiter: The delimiter to use when reading the CSV file.

    Yields:
      A Document for each row in the dataset.

    Raises:
      IOError: If the file does not exist.
      InvalidDatasetError: If the dataset is empty or invalid.
      NotImplementedError: If the file type is not supported.
    """
    if not os.path.exists(self.input_path):
      raise IOError(f'File does not exist: {self.input_path}')

    if str(self.input_path).endswith('.csv'):
      try:
        csv_data = _read_csv(
            self.input_path,
            column_names=[self.text_key, self.id_key],
            delimiter=delimiter,
        )
      except InvalidDatasetError as e:
        raise InvalidDatasetError(f'Empty dataset: {self.input_path}') from e
      for row in csv_data:
        yield data.Document(
            text=row[self.text_key],
            document_id=row[self.id_key],
        )
    else:
      raise NotImplementedError(f'Unsupported file type: {self.input_path}')


def save_annotated_documents(
    annotated_documents: Iterator[data.AnnotatedDocument],
    output_dir: pathlib.Path | str | None = None,
    output_name: str = 'data.jsonl',
    show_progress: bool = True,
) -> None:
  """Saves annotated documents to a JSON Lines file.

  Args:
    annotated_documents: Iterator over AnnotatedDocument objects to save.
    output_dir: The directory to which the JSONL file should be written.
      Can be a Path object or a string. Defaults to 'test_output/' if None.
    output_name: File name for the JSONL file.
    show_progress: Whether to show a progress bar during saving.

  Raises:
    IOError: If the output directory cannot be created.
    InvalidDatasetError: If no documents are produced.
  """
  if output_dir is None:
    output_dir = pathlib.Path('test_output')
  else:
    output_dir = pathlib.Path(output_dir)

  output_dir = output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  # Reject directory components in output_name rather than silently stripping
  # them; the docstring documents output_name as a file name, so passing
  # "subdir/data.jsonl" is a caller mistake we want to surface.
  output_name_path = pathlib.PurePath(output_name)
  if output_name_path.name != output_name or output_name_path.parts[:-1]:
    raise IOError(
        'output_name must be a file name with no directory components; '
        f'got {output_name!r}'
    )
  if not output_name_path.name or output_name_path.name in {'.', '..'}:
    raise IOError(f'Invalid output_name: {output_name!r}')

  output_file = (output_dir / output_name_path.name).resolve()

  # Defense-in-depth: even after rejecting path components, a symlinked leaf
  # inside output_dir could resolve outside it. Require containment.
  if output_dir not in output_file.parents and output_file != output_dir:
    raise IOError(
        f'Path traversal detected: resolved output {output_file} is outside '
        f'output_dir {output_dir}'
    )
  has_data = False
  doc_count = 0

  # Create progress bar
  progress_bar = progress.create_save_progress_bar(
      output_path=str(output_file), disable=not show_progress
  )

  try:
    with open(output_file, 'w', encoding='utf-8') as f:
      for adoc in annotated_documents:
        if not adoc.document_id:
          continue

        doc_dict = data_lib.annotated_document_to_dict(adoc)
        f.write(json.dumps(doc_dict, ensure_ascii=False) + '\n')
        has_data = True
        doc_count += 1
        progress_bar.update(1)
  finally:
    progress_bar.close()

  if not has_data:
    raise InvalidDatasetError(f'No documents to save in: {output_file}')

  if show_progress:
    progress.print_save_complete(doc_count, str(output_file))


def load_annotated_documents_jsonl(
    jsonl_path: pathlib.Path,
    show_progress: bool = True,
) -> Iterator[data.AnnotatedDocument]:
  """Loads annotated documents from a JSON Lines file.

  Args:
    jsonl_path: The file path to the JSON Lines file.
    show_progress: Whether to show a progress bar during loading.

  Yields:
    AnnotatedDocument objects.

  Raises:
    IOError: If the file does not exist or is invalid.
  """
  if not os.path.exists(jsonl_path):
    raise IOError(f'File does not exist: {jsonl_path}')

  # Get file size for progress bar
  file_size = os.path.getsize(jsonl_path)

  # Create progress bar
  progress_bar = progress.create_load_progress_bar(
      file_path=str(jsonl_path),
      total_size=file_size if show_progress else None,
      disable=not show_progress,
  )

  doc_count = 0
  bytes_read = 0

  with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
      line_bytes = len(line.encode('utf-8'))
      bytes_read += line_bytes
      progress_bar.update(line_bytes)

      line = line.strip()
      if not line:
        continue
      doc_dict = json.loads(line)
      doc_count += 1
      yield data_lib.dict_to_annotated_document(doc_dict)

  progress_bar.close()

  if show_progress:
    progress.print_load_complete(doc_count, str(jsonl_path))


def _read_csv(
    filepath: pathlib.Path, column_names: list[str], delimiter: str = ','
) -> Iterator[dict[str, Any]]:
  """Reads a CSV file and yields rows as dicts.

  Args:
    filepath: The path to the file.
    column_names: The names of the columns to read.
    delimiter: The delimiter to use when reading the CSV file.

  Yields:
    An iterator of dicts representing each row.

  Raises:
    IOError: If the file does not exist.
    InvalidDatasetError: If the dataset is empty or invalid.
  """
  if not os.path.exists(filepath):
    raise IOError(f'File does not exist: {filepath}')

  try:
    with open(filepath, 'r', encoding='utf-8') as f:
      df = pd.read_csv(f, usecols=column_names, dtype=str, delimiter=delimiter)
      for _, row in df.iterrows():
        yield row.to_dict()
  except pd.errors.EmptyDataError as e:
    raise InvalidDatasetError(f'Empty dataset: {filepath}') from e
  except ValueError as e:
    raise InvalidDatasetError(f'Invalid dataset file: {filepath}') from e


def is_url(text: str) -> bool:
  """Check if the given text is a valid URL.

  Uses urllib.parse to validate that the text is a properly formed URL
  with http or https scheme and a valid network location.

  Args:
    text: The string to check.

  Returns:
    True if the text is a valid URL with http(s) scheme, False otherwise.
  """
  if not text or not isinstance(text, str):
    return False

  text = text.strip()

  # Reject text with whitespace (not a pure URL)
  if ' ' in text or '\n' in text or '\t' in text:
    return False

  try:
    result = urlparse.urlparse(text)
    hostname = result.hostname

    # Must have valid scheme, netloc, and hostname
    if not (result.scheme in ('http', 'https') and result.netloc and hostname):
      return False

    # Accept IPs, localhost, or domains with dots
    try:
      ipaddress.ip_address(hostname)
      return True
    except ValueError:
      return hostname == 'localhost' or '.' in hostname
  except (ValueError, AttributeError):
    return False


_MAX_REDIRECT_HOPS = 5


def _follow_redirects_with_validation(
    url: str,
    timeout: int,
    allow_internal_urls: bool,
) -> requests.Response:
  """Issue a streaming GET, following redirects manually with validation.

  Each redirect hop is validated against `_is_internal_hostname` so that a
  public URL cannot serve `302 Location: http://169.254.169.254/` and be
  transparently followed into the metadata service.

  The caller is responsible for closing the returned Response.

  Args:
    url: The starting URL (already validated by the caller if
      `allow_internal_urls` is False).
    timeout: Per-request timeout in seconds.
    allow_internal_urls: If True, redirect targets are not re-validated.

  Returns:
    The final, non-3xx Response. The caller owns the stream and must close.

  Raises:
    InvalidUrlError: If a redirect target is internal, or the chain
      exceeds `_MAX_REDIRECT_HOPS`.
    requests.RequestException: For transport failures, or a 3xx response
      with no Location header.
  """
  current_url = url
  for _ in range(_MAX_REDIRECT_HOPS + 1):
    response = requests.get(
        current_url, stream=True, timeout=timeout, allow_redirects=False
    )
    if not 300 <= response.status_code < 400:
      return response
    location = response.headers.get('Location')
    response.close()
    if not location:
      raise requests.RequestException(
          f'Redirect response with no Location header from {current_url}'
      )
    current_url = urlparse.urljoin(current_url, location)
    if not allow_internal_urls:
      _validate_url_not_internal(current_url)
  raise InvalidUrlError(
      f'Too many redirects (>{_MAX_REDIRECT_HOPS}) starting from {url}'
  )


def download_text_from_url(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    show_progress: bool = True,
    chunk_size: int = 8192,
    *,
    allow_internal_urls: bool = False,
) -> str:
  """Download text content from a URL with optional progress bar.

  By default, URLs pointing to internal, loopback, link-local, private,
  CGNAT, multicast, or reserved addresses are rejected to prevent SSRF.
  Pass `allow_internal_urls=True` to opt out when fetching from a
  known-safe internal endpoint (e.g. development servers).

  Redirects are followed manually so each hop can be re-validated. Without
  this, `requests` would follow a `302 Location: http://169.254.169.254/`
  transparently and bypass the origin-URL check. Protection against
  DNS rebinding is partial: the HTTP client re-resolves the hostname at
  request time, so an attacker-controlled nameserver with a short TTL can
  still return an internal IP post-validation. Full mitigation would
  require pinning the resolved IP into the HTTP adapter.

  Args:
    url: The URL to download from.
    timeout: Request timeout in seconds.
    show_progress: Whether to show a progress bar during download.
    chunk_size: Size of chunks to download at a time.
    allow_internal_urls: If True, skip the internal-address block and do
      not re-validate redirect hops. Default False (secure by default).

  Returns:
    The text content of the URL.

  Raises:
    InvalidUrlError: If the URL (or any redirect hop) points to an internal
      address and `allow_internal_urls` is False, or the redirect chain
      exceeds the hop limit.
    requests.RequestException: If the download fails.
    ValueError: If the content is not text-based.
  """
  if not allow_internal_urls:
    _validate_url_not_internal(url)

  try:
    # Follow redirects manually with per-hop validation. `requests` would
    # otherwise transparently follow a `302 Location: http://internal/...`
    # response and bypass the origin-URL check above.
    response = _follow_redirects_with_validation(
        url, timeout=timeout, allow_internal_urls=allow_internal_urls
    )

    try:
      response.raise_for_status()

      # Check content type
      content_type = response.headers.get('Content-Type', '').lower()
      if not any(
          ct in content_type
          for ct in ['text/', 'application/json', 'application/xml']
      ):
        # Try to proceed anyway, but warn
        print(f"Warning: Content-Type '{content_type}' may not be text-based")

      # Get content length for progress bar
      total_size = int(response.headers.get('Content-Length', 0))

      filename = url.split('/')[-1][:50]

      # Download content with progress bar
      chunks = []
      if show_progress and total_size > 0:
        progress_bar = progress.create_download_progress_bar(
            total_size=total_size, url=url
        )

        try:
          for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
              chunks.append(chunk)
              progress_bar.update(len(chunk))
        finally:
          progress_bar.close()
      else:
        # Download without progress bar
        for chunk in response.iter_content(chunk_size=chunk_size):
          if chunk:
            chunks.append(chunk)

      # Combine chunks and decode
      content = b''.join(chunks)

      # Try to decode as text
      encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
      text_content = None
      for encoding in encodings:
        try:
          text_content = content.decode(encoding)
          break
        except UnicodeDecodeError:
          continue

      if text_content is None:
        raise ValueError(f'Could not decode content from {url} as text')

      # Show content summary with clean formatting
      if show_progress:
        char_count = len(text_content)
        word_count = len(text_content.split())
        progress.print_download_complete(char_count, word_count, filename)

      return text_content
    finally:
      response.close()

  except requests.RequestException as e:
    raise requests.RequestException(
        f'Failed to download from {url}: {str(e)}'
    ) from e
