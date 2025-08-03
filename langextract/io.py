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

import abc
import dataclasses
import json
import os
import pathlib
from typing import Any, Iterator

import pandas as pd
import requests

from langextract import data
from langextract import data_lib
from langextract import exceptions
from langextract import progress

DEFAULT_TIMEOUT_SECONDS = 30


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
    output_dir: pathlib.Path | None = None,
    output_name: str = 'data.jsonl',
    show_progress: bool = True,
) -> None:
  """Saves annotated documents to a JSON Lines file.

  Args:
    annotated_documents: Iterator over AnnotatedDocument objects to save.
    output_dir: The directory to which the JSONL file should be written.
      Defaults to 'test_output/' if None.
    output_name: File name for the JSONL file.
    show_progress: Whether to show a progress bar during saving.

  Raises:
    IOError: If the output directory cannot be created.
    InvalidDatasetError: If no documents are produced.
  """
  if output_dir is None:
    output_dir = pathlib.Path('test_output')

  output_dir.mkdir(parents=True, exist_ok=True)

  output_file = output_dir / output_name
  has_data = False
  doc_count = 0

  # Create progress bar
  progress_bar = progress.create_save_progress_bar(
      output_path=str(output_file), disable=not show_progress
  )

  with open(output_file, 'w') as f:
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

  with open(jsonl_path, 'r') as f:
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
    with open(filepath, 'r') as f:
      df = pd.read_csv(f, usecols=column_names, dtype=str, delimiter=delimiter)
      for _, row in df.iterrows():
        yield row.to_dict()
  except pd.errors.EmptyDataError as e:
    raise InvalidDatasetError(f'Empty dataset: {filepath}') from e
  except ValueError as e:
    raise InvalidDatasetError(f'Invalid dataset file: {filepath}') from e


def is_url(text: str) -> bool:
  """Check if the given text is a URL.

  Args:
    text: The string to check.

  Returns:
    True if the text is a URL (starts with http:// or https://), False
    otherwise.
  """
  return text.startswith('http://') or text.startswith('https://')


def download_text_from_url(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    show_progress: bool = True,
    chunk_size: int = 8192,
) -> str:
  """Download text content from a URL with optional progress bar.

  Args:
    url: The URL to download from.
    timeout: Request timeout in seconds.
    show_progress: Whether to show a progress bar during download.
    chunk_size: Size of chunks to download at a time.

  Returns:
    The text content of the URL.

  Raises:
    requests.RequestException: If the download fails.
    ValueError: If the content is not text-based.
  """
  try:
    # Make initial request to get headers
    response = requests.get(url, stream=True, timeout=timeout)
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
