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

"""Tokenization utilities for text.

Provides methods to split text into regex-based, word-level (and
punctuation-level) tokens. (1) Tokenization is necessary for alignment
between extracted data and the source text in resolver.py. (2) Tokenization
is also used in forming sentence boundaries for LLM information extraction for
smaller context use cases. These smaller context use cases are less necessary,
with newer larger context LLMs. This module is not used within the language
model to represent tokens during inference.
"""

from collections.abc import Sequence, Set
import dataclasses
import enum
import re

from langextract.core import debug_utils
from langextract.core import exceptions

__all__ = [
    "BaseTokenizerError",
    "InvalidTokenIntervalError",
    "SentenceRangeError",
    "CharInterval",
    "TokenInterval",
    "TokenType",
    "Token",
    "TokenizedText",
    "tokenize",
    "tokens_text",
    "find_sentence_range",
]


class BaseTokenizerError(exceptions.LangExtractError):
  """Base class for all tokenizer-related errors."""


class InvalidTokenIntervalError(BaseTokenizerError):
  """Error raised when a token interval is invalid or out of range."""


class SentenceRangeError(BaseTokenizerError):
  """Error raised when the start token index for a sentence is out of range."""


@dataclasses.dataclass
class CharInterval:
  """Represents a range of character positions in the original text.

  Attributes:
    start_pos: The starting character index (inclusive).
    end_pos: The ending character index (exclusive).
  """

  start_pos: int
  end_pos: int


@dataclasses.dataclass
class TokenInterval:
  """Represents an interval over tokens in tokenized text.

  The interval is defined by a start index (inclusive) and an end index
  (exclusive).

  Attributes:
    start_index: The index of the first token in the interval.
    end_index: The index one past the last token in the interval.
  """

  start_index: int = 0
  end_index: int = 0


class TokenType(enum.IntEnum):
  """Enumeration of token types produced during tokenization.

  Attributes:
    WORD: Represents an alphabetical word token.
    NUMBER: Represents a numeric token.
    PUNCTUATION: Represents punctuation characters.
    ACRONYM: Represents an acronym or slash-delimited abbreviation.
  """

  WORD = 0
  NUMBER = 1
  PUNCTUATION = 2
  ACRONYM = 3


@dataclasses.dataclass
class Token:
  """Represents a token extracted from text.

  Each token is assigned an index and classified into a type (word, number,
  punctuation,
  or acronym). The token also records the range of characters (its CharInterval)
  that
  correspond to the substring from the original text. Additionally, it tracks
  whether it
  follows a newline.

  Attributes:
    index: The position of the token in the sequence of tokens.
    token_type: The type of the token, as defined by TokenType.
    char_interval: The character interval within the original text that this
      token spans.
    first_token_after_newline: True if the token immediately follows a newline
      or carriage return.
  """

  index: int
  token_type: TokenType
  char_interval: CharInterval = dataclasses.field(
      default_factory=lambda: CharInterval(0, 0)
  )
  first_token_after_newline: bool = False


@dataclasses.dataclass
class TokenizedText:
  """Holds the result of tokenizing a text string.

  Attributes:
    text: The original text that was tokenized.
    tokens: A list of Token objects extracted from the text.
  """

  text: str
  tokens: list[Token] = dataclasses.field(default_factory=list)


# Regex patterns for tokenization.
_LETTERS_PATTERN = r"[A-Za-z]+"
_DIGITS_PATTERN = r"[0-9]+"
_SYMBOLS_PATTERN = r"[^A-Za-z0-9\s]+"
_END_OF_SENTENCE_PATTERN = re.compile(r"[.?!]$")
_SLASH_ABBREV_PATTERN = r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+"
_CN = r"[^\x00-\xff]"

_TOKEN_PATTERN = re.compile(
    rf"{_CN}|{_SLASH_ABBREV_PATTERN}|{_LETTERS_PATTERN}|{_DIGITS_PATTERN}|{_SYMBOLS_PATTERN}"
)
_WORD_PATTERN = re.compile(rf"(?:{_CN}|{_LETTERS_PATTERN}|{_DIGITS_PATTERN})\Z")

# Known abbreviations that should not count as sentence enders.
# TODO: This can potentially be removed given most use cases
# are larger context.
_KNOWN_ABBREVIATIONS = frozenset({"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St."})


@debug_utils.debug_log_calls
def tokenize(text: str) -> TokenizedText:
  """Splits text into tokens (words, digits, or punctuation).

  Each token is annotated with its character position and type (WORD or
  PUNCTUATION). If there is a newline or carriage return in the gap before
  a token, that token's `first_token_after_newline` is set to True.

  Args:
    text: The text to tokenize.

  Returns:
    A TokenizedText object containing all extracted tokens.
  """
  tokenized = TokenizedText(text=text)
  previous_end = 0
  for token_index, match in enumerate(_TOKEN_PATTERN.finditer(text)):
    start_pos, end_pos = match.span()
    matched_text = match.group()
    # Create a new token.
    token = Token(
        index=token_index,
        char_interval=CharInterval(start_pos=start_pos, end_pos=end_pos),
        token_type=TokenType.WORD,
        first_token_after_newline=False,
    )
    # Check if there's a newline in the gap before this token.
    if token_index > 0:
      gap = text[previous_end:start_pos]
      if "\n" in gap or "\r" in gap:
        token.first_token_after_newline = True
    # Classify token type.
    if re.fullmatch(_DIGITS_PATTERN, matched_text):
      token.token_type = TokenType.NUMBER
    elif re.fullmatch(_SLASH_ABBREV_PATTERN, matched_text):
      token.token_type = TokenType.ACRONYM
    elif _WORD_PATTERN.fullmatch(matched_text):
      token.token_type = TokenType.WORD
    else:
      token.token_type = TokenType.PUNCTUATION
    tokenized.tokens.append(token)
    previous_end = end_pos
  return tokenized


def tokens_text(
    tokenized_text: TokenizedText,
    token_interval: TokenInterval,
) -> str:
  """Reconstructs the substring of the original text spanning a given token interval.

  Args:
    tokenized_text: A TokenizedText object containing token data.
    token_interval: The interval specifying the range [start_index, end_index)
      of tokens.

  Returns:
    The exact substring of the original text corresponding to the token
    interval.

  Raises:
    InvalidTokenIntervalError: If the token_interval is invalid or out of range.
  """
  if (
      token_interval.start_index < 0
      or token_interval.end_index > len(tokenized_text.tokens)
      or token_interval.start_index >= token_interval.end_index
  ):

    raise InvalidTokenIntervalError(
        f"Invalid token interval. start_index={token_interval.start_index}, "
        f"end_index={token_interval.end_index}, "
        f"total_tokens={len(tokenized_text.tokens)}."
    )

  start_token = tokenized_text.tokens[token_interval.start_index]
  end_token = tokenized_text.tokens[token_interval.end_index - 1]
  return tokenized_text.text[
      start_token.char_interval.start_pos : end_token.char_interval.end_pos
  ]


def _is_end_of_sentence_token(
    text: str,
    tokens: Sequence[Token],
    current_idx: int,
    known_abbreviations: Set[str] = _KNOWN_ABBREVIATIONS,
) -> bool:
  """Checks if the punctuation token at `current_idx` ends a sentence.

  A token is considered a sentence terminator and is not part of a known
  abbreviation. Only searches the text corresponding to the current token.

  Args:
    text: The entire input text.
    tokens: The sequence of Token objects.
    current_idx: The current token index to check.
    known_abbreviations: Abbreviations that should not count as sentence enders
      (e.g., "Dr.").

  Returns:
    True if the token at `current_idx` ends a sentence, otherwise False.
  """
  current_token_text = text[
      tokens[current_idx]
      .char_interval.start_pos : tokens[current_idx]
      .char_interval.end_pos
  ]
  if _END_OF_SENTENCE_PATTERN.search(current_token_text):
    if current_idx > 0:
      prev_token_text = text[
          tokens[current_idx - 1]
          .char_interval.start_pos : tokens[current_idx - 1]
          .char_interval.end_pos
      ]
      if f"{prev_token_text}{current_token_text}" in known_abbreviations:
        return False
    return True
  return False


def _is_sentence_break_after_newline(
    text: str,
    tokens: Sequence[Token],
    current_idx: int,
) -> bool:
  """Checks if there's a newline before the next token and if that next token starts uppercase.

  This is a heuristic for determining sentence boundaries. It favors terminating
  a sentence prematurely over missing a sentence boundary, and will terminate a
  sentence early if the first line ends with new line and the second line begins
  with a capital letter.

  Args:
    text: The entire input text.
    tokens: The sequence of Token objects.
    current_idx: The current token index.

  Returns:
    True if a newline is found between current_idx and current_idx+1, and
    the next token (if any) begins with an uppercase character.
  """
  if current_idx + 1 >= len(tokens):
    return False

  gap_text = text[
      tokens[current_idx]
      .char_interval.end_pos : tokens[current_idx + 1]
      .char_interval.start_pos
  ]
  if "\n" not in gap_text:
    return False

  next_token_text = text[
      tokens[current_idx + 1]
      .char_interval.start_pos : tokens[current_idx + 1]
      .char_interval.end_pos
  ]
  return bool(next_token_text) and next_token_text[0].isupper()


def find_sentence_range(
    text: str,
    tokens: Sequence[Token],
    start_token_index: int,
) -> TokenInterval:
  """Finds a 'sentence' interval from a given start index.

  Sentence boundaries are defined by:
    - punctuation tokens in _END_OF_SENTENCE_PATTERN
    - newline breaks followed by an uppercase letter
    - not abbreviations in _KNOWN_ABBREVIATIONS (e.g., "Dr.")

  This favors terminating a sentence prematurely over missing a sentence
  boundary, and will terminate a sentence early if the first line ends with new
  line and the second line begins with a capital letter.

  Args:
    text: The original text.
    tokens: The tokens that make up `text`.
    start_token_index: The token index from which to begin the sentence.

  Returns:
    A TokenInterval representing the sentence range [start_token_index, end). If
    no sentence boundary is found, the end index will be the length of
    `tokens`.

  Raises:
    SentenceRangeError: If `start_token_index` is out of range.
  """
  if start_token_index < 0 or start_token_index >= len(tokens):
    raise SentenceRangeError(
        f"start_token_index={start_token_index} out of range. "
        f"Total tokens: {len(tokens)}."
    )

  i = start_token_index
  while i < len(tokens):
    if tokens[i].token_type == TokenType.PUNCTUATION:
      if _is_end_of_sentence_token(text, tokens, i, _KNOWN_ABBREVIATIONS):
        return TokenInterval(start_index=start_token_index, end_index=i + 1)
    if _is_sentence_break_after_newline(text, tokens, i):
      return TokenInterval(start_index=start_token_index, end_index=i + 1)
    i += 1

  return TokenInterval(start_index=start_token_index, end_index=len(tokens))
