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

Provides methods to split text into regex-based or Unicode-aware tokens.
Tokenization is used for alignment in `resolver.py` and for determining
sentence boundaries for smaller context use cases. This module is not used
for tokenization within the language model during inference.
"""

import abc
from collections.abc import Sequence, Set
import dataclasses
import enum
import functools
import unicodedata

import regex

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
    "Tokenizer",
    "RegexTokenizer",
    "UnicodeTokenizer",
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


@dataclasses.dataclass(slots=True)
class CharInterval:
  """Represents a range of character positions in the original text.

  Attributes:
    start_pos: The starting character index (inclusive).
    end_pos: The ending character index (exclusive).
  """

  start_pos: int
  end_pos: int


@dataclasses.dataclass(slots=True)
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
  """

  WORD = 0
  NUMBER = 1
  PUNCTUATION = 2


@dataclasses.dataclass(slots=True)
class Token:
  """Represents a token extracted from text.

  Each token is assigned an index and classified into a type (word, number,
  punctuation, or acronym). The token also records the range of characters
  (its CharInterval) that correspond to the substring from the original text.
  Additionally, it tracks whether it follows a newline.

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
    text: The text that was tokenized. For UnicodeTokenizer, this is
      NOT normalized to NFC (to preserve indices).
    tokens: A list of Token objects extracted from the text.
  """

  text: str
  tokens: list[Token] = dataclasses.field(default_factory=list)


_LETTERS_PATTERN = r"[^\W\d_]+"
_DIGITS_PATTERN = r"\d+"
# Group identical symbols (e.g. "!!") but split mixed ones.
_SYMBOLS_PATTERN = r"([^\w\s]|_)\1*"
_END_OF_SENTENCE_PATTERN = regex.compile(r"[.?!。！？\u0964][\"'”’»)\]}]*$")

_TOKEN_PATTERN = regex.compile(
    rf"{_LETTERS_PATTERN}|{_DIGITS_PATTERN}|{_SYMBOLS_PATTERN}"
)
_WORD_PATTERN = regex.compile(rf"(?:{_LETTERS_PATTERN}|{_DIGITS_PATTERN})\Z")

# Abbreviations that do not end sentences.
# TODO: Evaluate removal for large-context use cases.
_KNOWN_ABBREVIATIONS = frozenset({"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St."})
_CLOSING_PUNCTUATION = frozenset({'"', "'", "”", "’", "»", ")", "]", "}"})


class Tokenizer(abc.ABC):
  """Abstract base class for tokenizers."""

  @abc.abstractmethod
  def tokenize(self, text: str) -> TokenizedText:
    """Splits text into tokens.

    Args:
      text: The text to tokenize.

    Returns:
      A TokenizedText object.
    """


class RegexTokenizer(Tokenizer):
  """Regex-based tokenizer (default).

  The RegexTokenizer is faster than UnicodeTokenizer for English text because it
  skips involved Unicode handling.
  """

  @debug_utils.debug_log_calls
  def tokenize(self, text: str) -> TokenizedText:
    """Splits text into tokens (words, digits, or punctuation).

    Each token is annotated with its character position and type. Tokens
    following a newline or carriage return have `first_token_after_newline`
    set to True.

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
      token = Token(
          index=token_index,
          char_interval=CharInterval(start_pos=start_pos, end_pos=end_pos),
          token_type=TokenType.WORD,
          first_token_after_newline=False,
      )
      if token_index > 0:
        # Optimization: Check gap without slicing.
        has_newline = text.find("\n", previous_end, start_pos) != -1
        if not has_newline:
          has_newline = text.find("\r", previous_end, start_pos) != -1
        if has_newline:
          token.first_token_after_newline = True
      if regex.fullmatch(_DIGITS_PATTERN, matched_text):
        token.token_type = TokenType.NUMBER
      elif _WORD_PATTERN.fullmatch(matched_text):
        token.token_type = TokenType.WORD
      else:
        token.token_type = TokenType.PUNCTUATION
      tokenized.tokens.append(token)
      previous_end = end_pos
    return tokenized


# Default tokenizer instance for backward compatibility
_DEFAULT_TOKENIZER = RegexTokenizer()


def tokenize(
    text: str, tokenizer: Tokenizer = _DEFAULT_TOKENIZER
) -> TokenizedText:
  """Splits text into tokens using the provided tokenizer (default: RegexTokenizer).

  Args:
    text: The text to tokenize.
    tokenizer: The tokenizer instance to use.

  Returns:
    A TokenizedText object.
  """
  return tokenizer.tokenize(text)


_CJK_PATTERN = regex.compile(
    r"\p{Is_Han}|\p{Is_Hiragana}|\p{Is_Katakana}|\p{Is_Hangul}"
)
_NON_SPACED_PATTERN = regex.compile(
    r"\p{Is_Thai}|\p{Is_Lao}|\p{Is_Khmer}|\p{Is_Myanmar}"
)


class Sentinel:
  """Sentinel class for unique object identification."""

  def __init__(self, name: str):
    self.name = name

  def __repr__(self) -> str:
    return f"<{self.name}>"


_NO_GROUP_SCRIPT = Sentinel("NO_GROUP")
_UNKNOWN_SCRIPT = Sentinel("UNKNOWN")
_LATIN_SCRIPT = "Latin"


# Optimization: Direct mapping for common scripts avoids regex overhead.
def _get_script_fast(char: str) -> str | Sentinel:
  # Fast path for ASCII: Avoids regex and unicodedata lookups.
  if ord(char) < 128:
    return _LATIN_SCRIPT

  # Fallback to the robust regex method
  return _get_common_script_cached(char)


def _classify_grapheme(g: str) -> TokenType:
  if not g:
    return TokenType.PUNCTUATION
  c = g[0]
  cat = unicodedata.category(c)
  if cat.startswith("L"):
    return TokenType.WORD
  if cat.startswith("N"):
    return TokenType.NUMBER
  return TokenType.PUNCTUATION


_COMMON_SCRIPTS = [
    "Latin",
    "Cyrillic",
    "Greek",
    "Arabic",
    "Hebrew",
    "Devanagari",
]

_COMMON_SCRIPTS_PATTERN = regex.compile(
    "|".join(
        rf"(?P<{script}>\p{{Script={script}}})" for script in _COMMON_SCRIPTS
    )
)

_GRAPHEME_CLUSTER_PATTERN = regex.compile(r"\X")


@functools.lru_cache(maxsize=4096)
def _get_common_script_cached(c: str) -> str | Sentinel:
  """Determines script using regex, cached for performance."""
  match = _COMMON_SCRIPTS_PATTERN.match(c)
  if match:
    return match.lastgroup
  return _UNKNOWN_SCRIPT


class UnicodeTokenizer(Tokenizer):
  """Unicode-aware tokenizer for better non-English support.

  This tokenizer uses Unicode character properties (Unicode Standard Annex #29)
  via the `regex` library's `\\X` pattern to correctly handle grapheme clusters
  like Emojis and Hangul.


  Unlike some Unicode tokenizers, this class does NOT normalize text to NFC.
  This ensures that token indices exactly match the original input string.

  Note: Grapheme clustering makes this tokenizer slower than RegexTokenizer.
  """

  @debug_utils.debug_log_calls
  def tokenize(self, text: str) -> TokenizedText:
    """Splits text into tokens using Unicode properties.

    Args:
      text: The text to tokenize.

    Returns:
      A TokenizedText object.
    """
    tokens: list[Token] = []

    current_start = 0
    current_type = None
    current_script = None
    previous_end = 0

    for match in regex.finditer(r"\X", text):
      grapheme = match.group()
      start, _ = match.span()

      # 1. Handle Whitespace
      if grapheme.isspace():
        if current_type is not None:
          self._emit_token(
              tokens, text, current_start, start, current_type, previous_end
          )
          previous_end = start
          current_type = None
          current_script = None
        # Keep `previous_end` to detect newlines within the whitespace gap.
        continue

      g_type = _classify_grapheme(grapheme)

      # 2. Determine if we should merge with the current token
      should_merge = False
      if current_type is not None:
        if current_type == g_type:
          if current_type == TokenType.WORD:
            # Script Check
            first_char = grapheme[0]

            # Fast path: Explicit NO_GROUP (CJK/Thai) never merges.
            if current_script is _NO_GROUP_SCRIPT:
              should_merge = False

            # CJK and Non-Spaced scripts require fragmentation.
            elif _CJK_PATTERN.match(first_char) or _NON_SPACED_PATTERN.match(
                first_char
            ):
              should_merge = False

            else:
              g_script = _get_script_fast(first_char)
              # Safety: Do not merge distinct unknown scripts.
              if (
                  current_script == g_script
                  and current_script is not _UNKNOWN_SCRIPT
              ):
                should_merge = True

          elif current_type == TokenType.NUMBER:
            should_merge = True

          elif current_type == TokenType.PUNCTUATION:
            # Heuristic: Merge punctuation only if identical (e.g. "!!").
            last_grapheme = text[current_start:start]
            if last_grapheme == grapheme:
              should_merge = True
            elif len(last_grapheme) >= len(grapheme) and last_grapheme.endswith(
                grapheme
            ):
              should_merge = True

      # 3. State Transition
      if should_merge:
        # Extend current token
        pass
      else:
        # Flush previous token if exists
        if current_type is not None:
          self._emit_token(
              tokens, text, current_start, start, current_type, previous_end
          )
          previous_end = start

        # Start new token
        current_start = start
        current_type = g_type

        # Determine script for the new token
        if current_type == TokenType.WORD:
          c = grapheme[0]
          if _CJK_PATTERN.match(c) or _NON_SPACED_PATTERN.match(c):
            current_script = _NO_GROUP_SCRIPT
          else:
            current_script = _get_script_fast(c)
        else:
          current_script = None

    # 4. Flush final token
    if current_type is not None:
      self._emit_token(
          tokens, text, current_start, len(text), current_type, previous_end
      )

    return TokenizedText(text=text, tokens=tokens)

  def _emit_token(
      self,
      tokens: list[Token],
      text: str,
      start: int,
      end: int,
      token_type: TokenType,
      previous_end: int,
  ):
    """Helper to create and append a token."""
    token = Token(
        index=len(tokens),
        char_interval=CharInterval(start_pos=start, end_pos=end),
        token_type=token_type,
        first_token_after_newline=False,
    )

    # Check for newlines in the gap between the previous token and this one
    if start > previous_end:
      gap = text[previous_end:start]
      if "\n" in gap or "\r" in gap:
        token.first_token_after_newline = True

    tokens.append(token)


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
  if token_interval.start_index == token_interval.end_index:
    return ""

  if (
      token_interval.start_index < 0
      or token_interval.end_index > len(tokenized_text.tokens)
      or token_interval.start_index > token_interval.end_index
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
  """Checks if the next token starts uppercase and follows a newline.

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

  next_token = tokens[current_idx + 1]

  if not next_token.first_token_after_newline:
    return False

  next_token_text = text[
      next_token.char_interval.start_pos : next_token.char_interval.end_pos
  ]
  # Assume break unless lowercase (covers numbers/quotes).
  return bool(next_token_text) and not next_token_text[0].islower()


def find_sentence_range(
    text: str,
    tokens: Sequence[Token],
    start_token_index: int,
    known_abbreviations: Set[str] = _KNOWN_ABBREVIATIONS,
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
    text: The text to analyze.
    tokens: The tokens that make up `text`.
      Note: For UnicodeTokenizer, use normalized text.
    start_token_index: The index of the token to start the sentence from.
    known_abbreviations: A set of strings that are known abbreviations and
      should not be treated as sentence boundaries.


  Returns:
    A TokenInterval representing the sentence range [start_token_index, end). If
    no sentence boundary is found, the end index will be the length of
    `tokens`.

  Raises:
    SentenceRangeError: If `start_token_index` is out of range.
  """
  if not tokens:
    return TokenInterval(0, 0)

  if start_token_index < 0 or start_token_index >= len(tokens):
    raise SentenceRangeError(
        f"start_token_index={start_token_index} out of range. "
        f"Total tokens: {len(tokens)}."
    )

  i = start_token_index
  while i < len(tokens):
    if tokens[i].token_type == TokenType.PUNCTUATION:
      if _is_end_of_sentence_token(text, tokens, i, known_abbreviations):
        end_index = i + 1
        # Consume any trailing closing punctuation (e.g. quotes, parens)
        while end_index < len(tokens):
          next_token_text = text[
              tokens[end_index]
              .char_interval.start_pos : tokens[end_index]
              .char_interval.end_pos
          ]
          if (
              tokens[end_index].token_type == TokenType.PUNCTUATION
              and next_token_text in _CLOSING_PUNCTUATION
          ):
            end_index += 1
          else:
            break
        return TokenInterval(start_index=start_token_index, end_index=end_index)
    if _is_sentence_break_after_newline(text, tokens, i):
      return TokenInterval(start_index=start_token_index, end_index=i + 1)
    i += 1

  return TokenInterval(start_index=start_token_index, end_index=len(tokens))
