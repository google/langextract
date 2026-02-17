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

"""Alignment helpers used by `resolver.WordAligner`.

These utilities are intentionally conservative: they only adjust alignment
spans when a "lesser" match would otherwise produce an unhelpfully short
character interval for a long extraction (common when models paraphrase or
omit words in long, non-English extractions).
"""

from __future__ import annotations

from langextract.core import tokenizer as tokenizer_lib

# Heuristics for when a MATCH_LESSER span is too small to be useful.
_MIN_EXTRACTION_TOKENS_FOR_EXTENSION = 12
_MIN_UNMATCHED_TOKENS_FOR_EXTENSION = 10
_SENTENCE_TERMINATORS = frozenset({".", "?", "!", "。", "？", "！", "\u0964"})


def maybe_extend_lesser_match_end(
    *,
    tokenized_text: tokenizer_lib.TokenizedText,
    start_index: int,
    matched_tokens: int,
    extraction_tokens: int,
) -> int | None:
  """Return an extended end_index (exclusive) for a lesser match.

  Only extends when:
  - the extraction is long, and
  - the exact prefix match accounts for only a small portion of it.

  The extension prefers a sentence boundary when available. If no boundary is
  found, it caps the span using the extraction length to avoid returning an
  entire chunk.

  Args:
    tokenized_text: Tokenized source text (indices must match original text).
    start_index: Start token index (inclusive) of the match in source tokens.
    matched_tokens: Number of source tokens that matched exactly.
    extraction_tokens: Total number of tokens in the extraction text.

  Returns:
    The extended end token index (exclusive), or None if no extension should
    be applied.
  """
  if extraction_tokens < _MIN_EXTRACTION_TOKENS_FOR_EXTENSION:
    return None
  if extraction_tokens - matched_tokens < _MIN_UNMATCHED_TOKENS_FOR_EXTENSION:
    return None

  tokens = tokenized_text.tokens
  if not tokens:
    return None
  if start_index < 0 or start_index >= len(tokens):
    return None

  # Find a conservative "sentence" end for the span.
  try:
    sentence = tokenizer_lib.find_sentence_range(
        tokenized_text.text, tokens, start_index
    )
  except tokenizer_lib.SentenceRangeError:
    return None

  end_index = sentence.end_index
  if end_index == len(tokens):
    # Only cap to extraction length if there is no terminator at all in the
    # remainder of the text (i.e., we truly hit end-of-chunk without finding
    # a sentence boundary).
    if not _contains_sentence_terminator(tokenized_text, start_index):
      end_index = min(end_index, start_index + extraction_tokens)

  min_end = start_index + matched_tokens
  if end_index <= min_end:
    return None
  return end_index


def _contains_sentence_terminator(
    tokenized_text: tokenizer_lib.TokenizedText, start_index: int
) -> bool:
  """Return True if any sentence terminator appears at/after start_index."""
  text = tokenized_text.text
  tokens = tokenized_text.tokens
  for idx in range(max(0, start_index), len(tokens)):
    tok = tokens[idx]
    if tok.token_type != tokenizer_lib.TokenType.PUNCTUATION:
      continue
    token_text = text[tok.char_interval.start_pos : tok.char_interval.end_pos]
    if token_text not in _SENTENCE_TERMINATORS:
      continue
    # Ignore decimal points like "21.56".
    if (
        token_text == "."
        and idx > 0
        and idx + 1 < len(tokens)
        and tokens[idx - 1].token_type == tokenizer_lib.TokenType.NUMBER
        and tokens[idx + 1].token_type == tokenizer_lib.TokenType.NUMBER
    ):
      continue
    return True
  return False

