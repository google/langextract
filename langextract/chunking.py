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

"""Library for breaking documents into chunks of sentences.

When a text-to-text model (e.g. a large language model with a fixed context
size) can not accommodate a large document, this library can help us break the
document into chunks of a required maximum length that we can perform
inference on.
"""

from collections.abc import Iterable, Iterator, Sequence
import dataclasses
import re

from absl import logging
import more_itertools

from langextract import data
from langextract import exceptions
from langextract import tokenizer


class TokenUtilError(exceptions.LangExtractError):
  """Error raised when token_util returns unexpected values."""


@dataclasses.dataclass
class TextChunk:
  """Stores a text chunk with attributes to the source document.

  Attributes:
    token_interval: The token interval of the chunk in the source document.
    document: The source document.
  """

  token_interval: tokenizer.TokenInterval
  document: data.Document | None = None
  _chunk_text: str | None = dataclasses.field(
      default=None, init=False, repr=False
  )
  _sanitized_chunk_text: str | None = dataclasses.field(
      default=None, init=False, repr=False
  )
  _char_interval: data.CharInterval | None = dataclasses.field(
      default=None, init=False, repr=False
  )

  def __str__(self):
    interval_repr = (
        f"start_index: {self.token_interval.start_index}, end_index:"
        f" {self.token_interval.end_index}"
    )

    doc_id_repr = (
        f"Document ID: {self.document_id}"
        if self.document_id
        else "Document ID: None"
    )

    try:
      chunk_text_repr = f"'{self.chunk_text}'"
    except ValueError:
      chunk_text_repr = "<unavailable: document_text not set>"

    return (
        "TextChunk(\n"
        f"  interval=[{interval_repr}],\n"
        f"  {doc_id_repr},\n"
        f"  Chunk Text: {chunk_text_repr}\n"
        ")"
    )

  @property
  def document_id(self) -> str | None:
    """Gets the document ID from the source document."""
    if self.document is not None:
      return self.document.document_id
    return None

  @property
  def document_text(self) -> tokenizer.TokenizedText | None:
    """Gets the tokenized text from the source document."""
    if self.document is not None:
      return self.document.tokenized_text
    return None

  @property
  def chunk_text(self) -> str:
    """Gets the chunk text. Raises an error if `document_text` is not set."""
    if self.document_text is None:
      raise ValueError("document_text must be set to access chunk_text.")
    if self._chunk_text is None:
      self._chunk_text = get_token_interval_text(
          self.document_text, self.token_interval
      )
    return self._chunk_text

  @property
  def sanitized_chunk_text(self) -> str:
    """Gets the sanitized chunk text."""
    if self._sanitized_chunk_text is None:
      self._sanitized_chunk_text = _sanitize(self.chunk_text)
    return self._sanitized_chunk_text

  @property
  def additional_context(self) -> str | None:
    """Gets the additional context for prompting from the source document."""
    if self.document is not None:
      return self.document.additional_context
    return None

  @property
  def char_interval(self) -> data.CharInterval:
    """Gets the character interval corresponding to the token interval.

    Returns:
      data.CharInterval: The character interval for this chunk.

    Raises:
      ValueError: If document_text is not set.
    """
    if self._char_interval is None:
      if self.document_text is None:
        raise ValueError("document_text must be set to compute char_interval.")
      self._char_interval = get_char_interval(
          self.document_text, self.token_interval
      )
    return self._char_interval


def create_token_interval(
    start_index: int, end_index: int
) -> tokenizer.TokenInterval:
  """Creates a token interval.

  Args:
    start_index: first token's index (inclusive).
    end_index: last token's index + 1 (exclusive).

  Returns:
    Token interval.

  Raises:
    ValueError: If the token indices are invalid.
  """
  if start_index < 0:
    raise ValueError(f"Start index {start_index} must be positive.")
  if start_index >= end_index:
    raise ValueError(
        f"Start index {start_index} must be < end index {end_index}."
    )
  return tokenizer.TokenInterval(start_index=start_index, end_index=end_index)


def get_token_interval_text(
    tokenized_text: tokenizer.TokenizedText,
    token_interval: tokenizer.TokenInterval,
) -> str:
  """Get the text within an interval of tokens.

  Args:
    tokenized_text: Tokenized documents.
    token_interval: An interval specifying the start (inclusive) and end
      (exclusive) indices of the tokens to extract. These indices refer to the
      positions in the list of tokens within `tokenized_text.tokens`, not the
      value of the field `index` of `token_pb2.Token`. If the tokens are
      [(index:0, text:A), (index:5, text:B), (index:10, text:C)], we should use
      token_interval=[0, 2] to represent taking A and B, not [0, 6]. Please see
      details from the implementation of tokenizer.tokens_text

  Returns:
    Text within the token interval.

  Raises:
    ValueError: If the token indices are invalid.
    TokenUtilError: If tokenizer.tokens_text returns an empty
    string.
  """
  if token_interval.start_index >= token_interval.end_index:
    raise ValueError(
        f"Start index {token_interval.start_index} must be < end index "
        f"{token_interval.end_index}."
    )
  return_string = tokenizer.tokens_text(tokenized_text, token_interval)
  logging.debug(
      "Token util returns string: %s for tokenized_text: %s, token_interval:"
      " %s",
      return_string,
      tokenized_text,
      token_interval,
  )
  if tokenized_text.text and not return_string:
    raise TokenUtilError(
        "Token util returns an empty string unexpectedly. Number of tokens is"
        f" tokenized_text: {len(tokenized_text.tokens)}, token_interval is"
        f" {token_interval.start_index} to {token_interval.end_index}, which"
        " should not lead to empty string."
    )
  return return_string


def get_char_interval(
    tokenized_text: tokenizer.TokenizedText,
    token_interval: tokenizer.TokenInterval,
) -> data.CharInterval:
  """Returns the char interval corresponding to the token interval.

  Args:
    tokenized_text: Document.
    token_interval: Token interval.

  Returns:
    Char interval of the token interval of interest.

  Raises:
    ValueError: If the token_interval is invalid.
  """
  if token_interval.start_index >= token_interval.end_index:
    raise ValueError(
        f"Start index {token_interval.start_index} must be < end index "
        f"{token_interval.end_index}."
    )
  start_token = tokenized_text.tokens[token_interval.start_index]
  # Penultimate token prior to interval.end_index
  final_token = tokenized_text.tokens[token_interval.end_index - 1]
  return data.CharInterval(
      start_pos=start_token.char_interval.start_pos,
      end_pos=final_token.char_interval.end_pos,
  )


def _sanitize(text: str) -> str:
  """Converts all whitespace characters in input text to a single space.

  Args:
    text: Input to sanitize.

  Returns:
    Sanitized text with newlines and excess spaces removed.

  Raises:
    ValueError: If the sanitized text is empty.
  """

  sanitized_text = re.sub(r"\s+", " ", text.strip())
  if not sanitized_text:
    raise ValueError("Sanitized text is empty.")
  return sanitized_text


def make_batches_of_textchunk(
    chunk_iter: Iterator[TextChunk],
    batch_length: int,
) -> Iterable[Sequence[TextChunk]]:
  """Processes chunks into batches of TextChunk for inference, using itertools.batched.

  Args:
    chunk_iter: Iterator of TextChunks.
    batch_length: Number of chunks to include in each batch.

  Yields:
    Batches of TextChunks.
  """
  for batch in more_itertools.batched(chunk_iter, batch_length):
    yield list(batch)


class SentenceIterator:
  """Iterate through sentences of a tokenized text."""

  def __init__(
      self,
      tokenized_text: tokenizer.TokenizedText,
      curr_token_pos: int = 0,
  ):
    """Constructor.

    Args:
      tokenized_text: Document to iterate through.
      curr_token_pos: Iterate through sentences from this token position.

    Raises:
      IndexError: if curr_token_pos is not within the document.
    """
    self.tokenized_text = tokenized_text
    self.token_len = len(tokenized_text.tokens)
    if curr_token_pos < 0:
      raise IndexError(
          f"Current token position {curr_token_pos} can not be negative."
      )
    elif curr_token_pos > self.token_len:
      raise IndexError(
          f"Current token position {curr_token_pos} is past the length of the "
          f"document {self.token_len}."
      )
    self.curr_token_pos = curr_token_pos

  def __iter__(self) -> Iterator[tokenizer.TokenInterval]:
    return self

  def __next__(self) -> tokenizer.TokenInterval:
    """Returns next sentence's interval starting from current token position.

    Returns:
      Next sentence token interval starting from current token position.

    Raises:
      StopIteration: If end of text is reached.
    """
    assert self.curr_token_pos <= self.token_len
    if self.curr_token_pos == self.token_len:
      raise StopIteration
    # This locates the sentence which contains the current token position.
    sentence_range = tokenizer.find_sentence_range(
        self.tokenized_text.text,
        self.tokenized_text.tokens,
        self.curr_token_pos,
    )
    assert sentence_range
    # Start the sentence from the current token position.
    # If we are in the middle of a sentence, we should start from there.
    sentence_range = create_token_interval(
        self.curr_token_pos, sentence_range.end_index
    )
    self.curr_token_pos = sentence_range.end_index
    return sentence_range


class ChunkIterator:
  r"""Iterate through chunks of a tokenized text.

  Chunks may consist of sentences or sentence fragments that can fit into the
  maximum character buffer that we can run inference on.

  A)
  If a sentence length exceeds the max char buffer, then it needs to be broken
  into chunks that can fit within the max char buffer. We do this in a way that
  maximizes the chunk length while respecting newlines (if present) and token
  boundaries.
  Consider this sentence from a poem by John Donne:
  ```
  No man is an island,
  Entire of itself,
  Every man is a piece of the continent,
  A part of the main.
  ```
  With max_char_buffer=40, the chunks are:
  * "No man is an island,\nEntire of itself," len=38
  * "Every man is a piece of the continent," len=38
  * "A part of the main." len=19

  B)
  If a single token exceeds the max char buffer, it comprises the whole chunk.
  Consider the sentence:
  "This is antidisestablishmentarianism."
  With max_char_buffer=20, the chunks are:
  * "This is" len=7
  * "antidisestablishmentarianism" len=28
  * "." len(1)

  C)
  If multiple *whole* sentences can fit within the max char buffer, then they
  are used to form the chunk.
  Consider the sentences:
  "Roses are red. Violets are blue. Flowers are nice. And so are you."
  With max_char_buffer=60, the chunks are:
  * "Roses are red. Violets are blue. Flowers are nice." len=50
  * "And so are you." len=15
  """

  def __init__(
      self,
      text: str | tokenizer.TokenizedText,
      max_char_buffer: int,
      document: data.Document | None = None,
  ):
    """Constructor.

    Args:
      text: Document to chunk. Can be either a string or a tokenized text.
      max_char_buffer: Size of buffer that we can run inference on.
      document: Optional source document.
    """
    if isinstance(text, str):
      text = tokenizer.TokenizedText(text=text)
    self.tokenized_text = text
    self.max_char_buffer = max_char_buffer
    self.sentence_iter = SentenceIterator(self.tokenized_text)
    self.broken_sentence = False

    # TODO: Refactor redundancy between document and text.
    if document is None:
      self.document = data.Document(text=text.text)
    else:
      self.document = document

  def __iter__(self) -> Iterator[TextChunk]:
    return self

  def _tokens_exceed_buffer(
      self, token_interval: tokenizer.TokenInterval
  ) -> bool:
    """Check if the token interval exceeds the maximum buffer size.

    Args:
      token_interval: Token interval to check.

    Returns:
      True if the token interval exceeds the maximum buffer size.
    """
    char_interval = get_char_interval(self.tokenized_text, token_interval)
    return (
        char_interval.end_pos - char_interval.start_pos
    ) > self.max_char_buffer

  def __next__(self) -> TextChunk:
    sentence = next(self.sentence_iter)
    # If the next token is greater than the max_char_buffer, let it be the
    # entire chunk.
    curr_chunk = create_token_interval(
        sentence.start_index, sentence.start_index + 1
    )
    if self._tokens_exceed_buffer(curr_chunk):
      self.sentence_iter = SentenceIterator(
          self.tokenized_text, curr_token_pos=sentence.start_index + 1
      )
      self.broken_sentence = curr_chunk.end_index < sentence.end_index
      return TextChunk(
          token_interval=curr_chunk,
          document=self.document,
      )

    # Append tokens to the chunk up to the max_char_buffer.
    start_of_new_line = -1
    for token_index in range(curr_chunk.start_index, sentence.end_index):
      if self.tokenized_text.tokens[token_index].first_token_after_newline:
        start_of_new_line = token_index
      test_chunk = create_token_interval(
          curr_chunk.start_index, token_index + 1
      )
      if self._tokens_exceed_buffer(test_chunk):
        if start_of_new_line > 0:
          # Terminate the curr_chunk at the start of the most recent newline.
          curr_chunk = create_token_interval(
              curr_chunk.start_index, start_of_new_line
          )
        self.sentence_iter = SentenceIterator(
            self.tokenized_text, curr_token_pos=curr_chunk.end_index
        )
        self.broken_sentence = True
        return TextChunk(
            token_interval=curr_chunk,
            document=self.document,
        )
      else:
        curr_chunk = test_chunk

    if self.broken_sentence:
      self.broken_sentence = False
    else:
      for sentence in self.sentence_iter:
        test_chunk = create_token_interval(
            curr_chunk.start_index, sentence.end_index
        )
        if self._tokens_exceed_buffer(test_chunk):
          self.sentence_iter = SentenceIterator(
              self.tokenized_text, curr_token_pos=curr_chunk.end_index
          )
          return TextChunk(
              token_interval=curr_chunk,
              document=self.document,
          )
        else:
          curr_chunk = test_chunk

    return TextChunk(
        token_interval=curr_chunk,
        document=self.document,
    )
