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
import re
from typing import Any, Iterator
from urllib import parse as urlparse

import pandas as pd
import requests

from langextract import data_lib
from langextract import progress
from langextract.core import data
from langextract.core import exceptions

DEFAULT_TIMEOUT_SECONDS = 30
SCRAPER_DEFAULT = 'default'
SCRAPER_FIRECRAWL = 'firecrawl'
_BLOCK_TAGS = {
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'p',
    'li',
    'tr',
    'div',
    'article',
    'section',
}
_NOISY_TAGS = {
    'script',
    'style',
    'noscript',
    'header',
    'footer',
    'nav',
    'aside',
    'form',
    'input',
    'select',
    'textarea',
    'button',
    'option',
    'label',
}


def _looks_like_html(content_type: str, text_content: str) -> bool:
  """Detects whether the downloaded text appears to be HTML."""
  normalized = text_content.lstrip().lower()
  return (
      'text/html' in content_type
      or normalized.startswith('<!doctype html')
      or normalized.startswith('<html')
  )


def _normalize_html_text(html_content: str) -> str:
  """Parses HTML and extracts readable text with paragraph boundaries."""
  try:
    from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel

    soup = BeautifulSoup(html_content, 'lxml')

    for noisy in soup.find_all(_NOISY_TAGS):
      noisy.decompose()

    for container in soup.select(
        '[class*="menu" i], [id*="menu" i], [class*="sidebar" i], '
        '[id*="sidebar" i], [class*="breadcrumb" i], [id*="breadcrumb" i], '
        '[class*="navigation" i], [id*="navigation" i]'
    ):
      container.decompose()

    content_root = None
    for selector in [
        'main',
        '[role="main"]',
        'article',
        '.article',
        '.post-content',
        '.entry-content',
        '#content',
        '.content',
        'body',
    ]:
      candidate = soup.select_one(selector)
      if candidate and candidate.get_text(strip=True):
        content_root = candidate
        break

    if content_root is None:
      content_root = soup

    for br in content_root.find_all('br'):
      br.replace_with('\n')

    for tag in content_root.find_all(_BLOCK_TAGS):
      tag.insert_before('\n\n')
      tag.insert_after('\n\n')

    text_content = content_root.get_text(separator=' ', strip=False)
  except ImportError:
    text_content = html_content
    for tag in _NOISY_TAGS:
      text_content = re.sub(
          rf'<{tag}\b[^>]*>.*?</{tag}>',
          ' ',
          text_content,
          flags=re.IGNORECASE | re.DOTALL,
      )
    text_content = re.sub(
        (
            r'<[^>]*(class|id)=(?:"|\')[^"\']*'
            r'(menu|sidebar|breadcrumb|navigation)[^"\']*(?:"|\')[^>]*>.*?</[^>]+>'
        ),
        ' ',
        text_content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text_content = re.sub(
        r'</?(h[1-6]|p|li|tr|div|article|section|main|body)\b[^>]*>',
        '\n\n',
        text_content,
        flags=re.IGNORECASE,
    )
    text_content = re.sub(r'<br\s*/?>', '\n', text_content, flags=re.IGNORECASE)
    text_content = re.sub(r'<[^>]+>', ' ', text_content)

  lines = [re.sub(r'\s+', ' ', line).strip() for line in text_content.split('\n')]
  non_empty = [line for line in lines if line]
  cleaned = '\n\n'.join(non_empty)
  return cleaned.strip()


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

  output_dir.mkdir(parents=True, exist_ok=True)

  output_file = output_dir / output_name
  has_data = False
  doc_count = 0

  # Create progress bar
  progress_bar = progress.create_save_progress_bar(
      output_path=str(output_file), disable=not show_progress
  )

  with open(output_file, 'w', encoding='utf-8') as f:
    for adoc in annotated_documents:
      if not adoc.document_id:
        continue

      doc_dict = data_lib.annotated_document_to_dict(adoc)
      f.write(json.dumps(doc_dict, ensure_ascii=False) + '\n')
      has_data = True
      doc_count += 1
      progress_bar.update(1)

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


def _download_with_firecrawl(
    url: str,
    firecrawl_api_key: str | None = None,
    show_progress: bool = True,
) -> str:
  """Download and parse a URL using Firecrawl for robust scraping.

  Firecrawl handles JavaScript-rendered pages, anti-bot protections, and
  difficult-to-scrape government/legal sites that often fail with plain
  HTTP requests.

  Args:
    url: The URL to scrape.
    firecrawl_api_key: Firecrawl API key. Falls back to the
      FIRECRAWL_API_KEY environment variable if not provided.
    show_progress: Whether to show progress information.

  Returns:
    The extracted text content of the page.

  Raises:
    ImportError: If firecrawl-py is not installed.
    RuntimeError: If scraping fails or returns empty content.
  """
  try:
    from firecrawl import FirecrawlApp  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise ImportError(
        "firecrawl-py is required for the Firecrawl scraper. "
        "Install it with: pip install firecrawl-py"
    ) from exc

  api_key = firecrawl_api_key or os.environ.get('FIRECRAWL_API_KEY', '')
  if not api_key:
    raise ValueError(
        "Firecrawl API key is required. Pass firecrawl_api_key or set "
        "the FIRECRAWL_API_KEY environment variable."
    )

  app = FirecrawlApp(api_key=api_key)

  if show_progress:
    print(f"Scraping {url} with Firecrawl...")

  result = app.scrape(url, formats=['markdown'])

  markdown = ''
  if isinstance(result, dict):
    markdown = result.get('markdown', '')
  elif hasattr(result, 'markdown'):
    markdown = result.markdown or ''
  else:
    markdown = str(result)

  if not markdown.strip():
    raise RuntimeError(f"Firecrawl returned empty content for {url}")

  text_content = markdown.strip()

  if show_progress:
    char_count = len(text_content)
    word_count = len(text_content.split())
    filename = url.split('/')[-1][:50]
    progress.print_download_complete(char_count, word_count, filename)

  return text_content


def download_text_from_url(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    show_progress: bool = True,
    chunk_size: int = 8192,
    scraper: str = SCRAPER_DEFAULT,
    firecrawl_api_key: str | None = None,
) -> str:
  """Download text content from a URL with optional progress bar.

  Args:
    url: The URL to download from.
    timeout: Request timeout in seconds.
    show_progress: Whether to show a progress bar during download.
    chunk_size: Size of chunks to download at a time.
    scraper: Scraping backend to use. 'default' uses requests +
      BeautifulSoup. 'firecrawl' uses the Firecrawl API for robust
      scraping of difficult pages (JS-rendered, anti-bot, government
      sites). Defaults to 'default'.
    firecrawl_api_key: API key for Firecrawl when scraper='firecrawl'.
      Falls back to FIRECRAWL_API_KEY env var.

  Returns:
    The text content of the URL.

  Raises:
    requests.RequestException: If the download fails (default scraper).
    ValueError: If the content is not text-based.
    ImportError: If firecrawl-py is not installed (firecrawl scraper).
    RuntimeError: If Firecrawl scraping fails.
  """
  if scraper == SCRAPER_FIRECRAWL:
    return _download_with_firecrawl(
        url=url,
        firecrawl_api_key=firecrawl_api_key,
        show_progress=show_progress,
    )

  try:
    # Make initial request to get headers.
    # In some managed environments, HTTPS proxy tunneling may be blocked for
    # specific domains. If that happens, retry once without proxies.
    try:
      response = requests.get(url, stream=True, timeout=timeout)
    except requests.exceptions.ProxyError as proxy_error:
      parsed_url = urlparse.urlparse(url)
      if parsed_url.scheme != 'https':
        raise
      try:
        response = requests.get(
            url,
            stream=True,
            timeout=timeout,
            proxies={'http': None, 'https': None},
        )
      except requests.RequestException as direct_error:
        raise requests.RequestException(
            'Failed to download from '
            f'{url}: Proxy connection failed ({proxy_error}) and '
            f'direct retry failed ({direct_error})'
        ) from direct_error

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

      for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
          chunks.append(chunk)
          progress_bar.update(len(chunk))

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

    if _looks_like_html(content_type, text_content):
      text_content = _normalize_html_text(text_content)

    # Show content summary with clean formatting
    if show_progress:
      char_count = len(text_content)
      word_count = len(text_content.split())
      progress.print_download_complete(char_count, word_count, filename)

    return text_content

  except requests.RequestException as e:
    raise requests.RequestException(
        f'Failed to download from {url}: {str(e)}'
    ) from e
