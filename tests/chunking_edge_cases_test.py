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

"""Edge-case tests for langextract.chunking.

Covers the untested code paths reported in
https://github.com/google/langextract/issues/430:
  - SentenceIterator.__init__ guard clauses
  - create_token_interval / get_token_interval_text / get_char_interval errors
  - ChunkIterator constructor edge cases
  - TextChunk.chunk_text / char_interval when document is None
  - TextChunk.sanitized_chunk_text
  - Lazy caching of _chunk_text / _char_interval
  - make_batches_of_textchunk with various batch sizes
  - broken_sentence flag reset
"""

from absl.testing import absltest
from absl.testing import parameterized

from langextract import chunking
from langextract.core import data
from langextract.core import tokenizer


# ---------------------------------------------------------------------------
# SentenceIterator guard clauses
# ---------------------------------------------------------------------------
class SentenceIteratorGuardTest(absltest.TestCase):
  """Tests for SentenceIterator.__init__ boundary checks."""

  def test_negative_curr_token_pos_raises_index_error(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    with self.assertRaises(IndexError):
      chunking.SentenceIterator(tokenized_text, curr_token_pos=-1)

  def test_curr_token_pos_past_end_raises_index_error(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    num_tokens = len(tokenized_text.tokens)
    with self.assertRaises(IndexError):
      chunking.SentenceIterator(tokenized_text, curr_token_pos=num_tokens + 1)

  def test_curr_token_pos_at_end_yields_nothing(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    num_tokens = len(tokenized_text.tokens)
    sentence_iter = chunking.SentenceIterator(
        tokenized_text, curr_token_pos=num_tokens
    )
    with self.assertRaises(StopIteration):
      next(sentence_iter)

  def test_curr_token_pos_mid_sentence(self):
    """Starting mid-sentence should yield a partial sentence first."""
    tokenized_text = tokenizer.tokenize("Hello world. Goodbye world.")
    sentence_iter = chunking.SentenceIterator(tokenized_text, curr_token_pos=1)
    first = next(sentence_iter)
    text = chunking.get_token_interval_text(tokenized_text, first)
    # Should start from token 1, not token 0
    self.assertEqual(first.start_index, 1)
    self.assertNotIn("Hello", text)


# ---------------------------------------------------------------------------
# create_token_interval error paths
# ---------------------------------------------------------------------------
class CreateTokenIntervalTest(absltest.TestCase):
  """Tests for create_token_interval validation."""

  def test_negative_start_index_raises(self):
    with self.assertRaises(ValueError):
      chunking.create_token_interval(-1, 5)

  def test_start_equals_end_raises(self):
    with self.assertRaises(ValueError):
      chunking.create_token_interval(3, 3)

  def test_start_greater_than_end_raises(self):
    with self.assertRaises(ValueError):
      chunking.create_token_interval(5, 3)

  def test_valid_interval(self):
    interval = chunking.create_token_interval(0, 5)
    self.assertEqual(interval.start_index, 0)
    self.assertEqual(interval.end_index, 5)


# ---------------------------------------------------------------------------
# get_token_interval_text error paths
# ---------------------------------------------------------------------------
class GetTokenIntervalTextTest(absltest.TestCase):
  """Tests for get_token_interval_text validation and TokenUtilError."""

  def test_start_ge_end_raises_value_error(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    bad_interval = tokenizer.TokenInterval(start_index=2, end_index=1)
    with self.assertRaises(ValueError):
      chunking.get_token_interval_text(tokenized_text, bad_interval)

  def test_equal_start_end_raises_value_error(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    bad_interval = tokenizer.TokenInterval(start_index=1, end_index=1)
    with self.assertRaises(ValueError):
      chunking.get_token_interval_text(tokenized_text, bad_interval)

  def test_token_util_error_on_empty_return(self):
    """TokenUtilError when tokenizer returns empty string for non-empty text."""
    tokenized_text = tokenizer.tokenize("Hello world.")
    # Construct an interval whose tokens exist but whose char spans produce
    # an empty string.  We achieve this by mocking: create a TokenizedText
    # with non-empty .text but tokens whose char_intervals map to an empty
    # substring.
    fake_token = tokenizer.Token(
        index=0,
        token_type=tokenizer.TokenType.WORD,
        char_interval=data.CharInterval(start_pos=5, end_pos=5),
    )
    fake_tokenized = tokenizer.TokenizedText(
        text="Hello", tokens=[fake_token, fake_token]
    )
    interval = tokenizer.TokenInterval(start_index=0, end_index=2)
    with self.assertRaises(chunking.TokenUtilError):
      chunking.get_token_interval_text(fake_tokenized, interval)


# ---------------------------------------------------------------------------
# get_char_interval error paths
# ---------------------------------------------------------------------------
class GetCharIntervalTest(absltest.TestCase):
  """Tests for get_char_interval validation."""

  def test_start_ge_end_raises_value_error(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    bad_interval = tokenizer.TokenInterval(start_index=2, end_index=1)
    with self.assertRaises(ValueError):
      chunking.get_char_interval(tokenized_text, bad_interval)

  def test_valid_char_interval(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    interval = tokenizer.TokenInterval(start_index=0, end_index=2)
    char_int = chunking.get_char_interval(tokenized_text, interval)
    self.assertIsNotNone(char_int.start_pos)
    self.assertIsNotNone(char_int.end_pos)
    self.assertLess(char_int.start_pos, char_int.end_pos)


# ---------------------------------------------------------------------------
# ChunkIterator constructor edge cases
# ---------------------------------------------------------------------------
class ChunkIteratorConstructorTest(absltest.TestCase):
  """Tests for ChunkIterator.__init__ edge cases."""

  def test_both_text_and_document_none_raises(self):
    with self.assertRaises(ValueError):
      chunking.ChunkIterator(
          text=None,
          max_char_buffer=100,
          tokenizer_impl=tokenizer.RegexTokenizer(),
          document=None,
      )

  def test_text_none_falls_back_to_document_text(self):
    doc = data.Document(text="From the document.")
    chunk_iter = chunking.ChunkIterator(
        text=None,
        max_char_buffer=200,
        tokenizer_impl=tokenizer.RegexTokenizer(),
        document=doc,
    )
    chunk = next(chunk_iter)
    self.assertEqual(chunk.chunk_text, "From the document.")

  def test_empty_tokenized_text_retokenizes(self):
    """When TokenizedText has no tokens, ChunkIterator re-tokenizes."""
    empty_tt = tokenizer.TokenizedText(text="Re-tokenize me.", tokens=[])
    chunk_iter = chunking.ChunkIterator(
        text=empty_tt,
        max_char_buffer=200,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk = next(chunk_iter)
    self.assertIn("Re-tokenize", chunk.chunk_text)

  def test_empty_tokenized_text_with_document_retokenizes(self):
    """Empty TokenizedText falls back to document.text for re-tokenization."""
    doc = data.Document(text="Document fallback text.")
    empty_tt = tokenizer.TokenizedText(text=None, tokens=[])
    chunk_iter = chunking.ChunkIterator(
        text=empty_tt,
        max_char_buffer=200,
        tokenizer_impl=tokenizer.RegexTokenizer(),
        document=doc,
    )
    chunk = next(chunk_iter)
    self.assertIn("Document fallback", chunk.chunk_text)

  def test_string_input_is_tokenized(self):
    chunk_iter = chunking.ChunkIterator(
        text="Plain string input.",
        max_char_buffer=200,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk = next(chunk_iter)
    self.assertEqual(chunk.chunk_text, "Plain string input.")

  def test_no_document_creates_default_document(self):
    chunk_iter = chunking.ChunkIterator(
        text="Auto document.",
        max_char_buffer=200,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk = next(chunk_iter)
    # Should have a document even though none was provided
    self.assertIsNotNone(chunk.document)
    self.assertIsNotNone(chunk.document_id)


# ---------------------------------------------------------------------------
# TextChunk.chunk_text / char_interval when document is None
# ---------------------------------------------------------------------------
class TextChunkErrorPathTest(absltest.TestCase):
  """Tests for TextChunk property errors when document is missing."""

  def test_chunk_text_raises_without_document(self):
    chunk = chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(start_index=0, end_index=1),
        document=None,
    )
    with self.assertRaises(ValueError):
      _ = chunk.chunk_text

  def test_char_interval_raises_without_document(self):
    chunk = chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(start_index=0, end_index=1),
        document=None,
    )
    with self.assertRaises(ValueError):
      _ = chunk.char_interval


# ---------------------------------------------------------------------------
# TextChunk.sanitized_chunk_text
# ---------------------------------------------------------------------------
class SanitizedChunkTextTest(absltest.TestCase):
  """Tests for TextChunk.sanitized_chunk_text."""

  def test_sanitized_removes_extra_whitespace(self):
    text = "Hello   world.\n\nNew  paragraph."
    doc = data.Document(text=text)
    tokenized_text = tokenizer.tokenize(text)
    chunk_iter = chunking.ChunkIterator(
        text=tokenized_text,
        max_char_buffer=500,
        tokenizer_impl=tokenizer.RegexTokenizer(),
        document=doc,
    )
    chunk = next(chunk_iter)
    sanitized = chunk.sanitized_chunk_text
    # Should not contain newlines or consecutive spaces
    self.assertNotIn("\n", sanitized)
    self.assertNotIn("  ", sanitized)

  def test_sanitized_text_cached(self):
    """Second access should return the same cached object."""
    text = "Caching test."
    doc = data.Document(text=text)
    chunk_iter = chunking.ChunkIterator(
        text=tokenizer.tokenize(text),
        max_char_buffer=500,
        tokenizer_impl=tokenizer.RegexTokenizer(),
        document=doc,
    )
    chunk = next(chunk_iter)
    first = chunk.sanitized_chunk_text
    second = chunk.sanitized_chunk_text
    self.assertIs(first, second)

  def test_sanitize_whitespace_only_raises(self):
    """_sanitize raises ValueError for all-whitespace input."""
    with self.assertRaises(ValueError):
      chunking._sanitize("   \n\t  ")


# ---------------------------------------------------------------------------
# Lazy caching of _chunk_text and _char_interval
# ---------------------------------------------------------------------------
class LazyCachingTest(absltest.TestCase):
  """Verify that chunk_text and char_interval are lazily cached."""

  def _make_chunk(self):
    text = "Cache me."
    doc = data.Document(text=text)
    chunk_iter = chunking.ChunkIterator(
        text=tokenizer.tokenize(text),
        max_char_buffer=500,
        tokenizer_impl=tokenizer.RegexTokenizer(),
        document=doc,
    )
    return next(chunk_iter)

  def test_chunk_text_starts_uncached(self):
    chunk = self._make_chunk()
    self.assertIsNone(chunk._chunk_text)

  def test_chunk_text_cached_after_access(self):
    chunk = self._make_chunk()
    _ = chunk.chunk_text
    self.assertIsNotNone(chunk._chunk_text)

  def test_chunk_text_same_object_on_repeated_access(self):
    chunk = self._make_chunk()
    first = chunk.chunk_text
    second = chunk.chunk_text
    self.assertIs(first, second)

  def test_char_interval_starts_uncached(self):
    chunk = self._make_chunk()
    self.assertIsNone(chunk._char_interval)

  def test_char_interval_cached_after_access(self):
    chunk = self._make_chunk()
    _ = chunk.char_interval
    self.assertIsNotNone(chunk._char_interval)

  def test_char_interval_same_object_on_repeated_access(self):
    chunk = self._make_chunk()
    first = chunk.char_interval
    second = chunk.char_interval
    self.assertIs(first, second)


# ---------------------------------------------------------------------------
# make_batches_of_textchunk with various batch sizes
# ---------------------------------------------------------------------------
class MakeBatchesTest(parameterized.TestCase):
  """Tests for make_batches_of_textchunk with different batch sizes."""

  def _make_chunks(self, n_chunks):
    """Create n_chunks TextChunks from a multi-sentence text."""
    sentences = " ".join(f"Sentence {i}." for i in range(n_chunks))
    doc = data.Document(text=sentences)
    chunk_iter = chunking.ChunkIterator(
        text=tokenizer.tokenize(sentences),
        max_char_buffer=15,  # small enough to get one chunk per sentence
        tokenizer_impl=tokenizer.RegexTokenizer(),
        document=doc,
    )
    return chunk_iter

  def test_batch_size_one(self):
    chunk_iter = self._make_chunks(5)
    batches = list(chunking.make_batches_of_textchunk(chunk_iter, 1))
    for batch in batches:
      self.assertEqual(len(batch), 1)

  def test_batch_size_two(self):
    chunk_iter = self._make_chunks(5)
    batches = list(chunking.make_batches_of_textchunk(chunk_iter, 2))
    # With 5+ chunks and batch_size 2, first batches should have 2
    self.assertEqual(len(batches[0]), 2)

  def test_batch_size_larger_than_chunks(self):
    """When batch_size > total chunks, everything lands in one batch."""
    chunk_iter = self._make_chunks(3)
    batches = list(chunking.make_batches_of_textchunk(chunk_iter, 100))
    self.assertEqual(len(batches), 1)

  def test_empty_iterator(self):
    empty_iter = iter([])
    batches = list(chunking.make_batches_of_textchunk(empty_iter, 5))
    self.assertEqual(len(batches), 0)


# ---------------------------------------------------------------------------
# broken_sentence flag reset
# ---------------------------------------------------------------------------
class BrokenSentenceFlagTest(absltest.TestCase):
  """Tests for the broken_sentence flag lifecycle.

  When a sentence is split across chunks (because it exceeds max_char_buffer),
  `broken_sentence` is set to True. It should be reset to False when the
  broken sentence is fully consumed and subsequent whole sentences are being
  accumulated.
  """

  def test_broken_sentence_set_when_splitting(self):
    """Splitting a long sentence should set broken_sentence."""
    text = "Short. This is a much longer sentence that exceeds the buffer."
    chunk_iter = chunking.ChunkIterator(
        text=tokenizer.tokenize(text),
        max_char_buffer=15,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    # First chunk: "Short." — fits
    next(chunk_iter)
    # At this point broken_sentence should be False (whole sentence fit)
    self.assertFalse(chunk_iter.broken_sentence)

    # Next chunk: beginning of the long sentence — should break it
    next(chunk_iter)
    # broken_sentence should be True because the sentence was split
    self.assertTrue(chunk_iter.broken_sentence)

  def test_broken_sentence_reset_after_completion(self):
    """After all fragments of a broken sentence are consumed, the flag resets."""
    # Two short sentences followed by a long one, then another short one
    text = "A. B. This is definitely a very long sentence that must be split across multiple chunks. C."
    chunk_iter = chunking.ChunkIterator(
        text=tokenizer.tokenize(text),
        max_char_buffer=20,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunks = list(chunk_iter)
    # After consuming everything, broken_sentence should be False
    self.assertFalse(chunk_iter.broken_sentence)
    # Should have more than 3 chunks (the long sentence splits)
    self.assertGreater(len(chunks), 3)

  def test_no_broken_sentence_for_small_text(self):
    """When all sentences fit, broken_sentence stays False throughout."""
    text = "Hi. Bye."
    chunk_iter = chunking.ChunkIterator(
        text=tokenizer.tokenize(text),
        max_char_buffer=200,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    list(chunk_iter)
    self.assertFalse(chunk_iter.broken_sentence)


# ---------------------------------------------------------------------------
# TextChunk.__str__ edge cases
# ---------------------------------------------------------------------------
class TextChunkStrTest(absltest.TestCase):
  """Tests for TextChunk.__str__ including the unavailable-text branch."""

  def test_str_without_document_shows_unavailable(self):
    chunk = chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(start_index=0, end_index=5),
        document=None,
    )
    result = str(chunk)
    self.assertIn("unavailable", result)
    self.assertIn("Document ID: None", result)

  def test_str_with_document(self):
    doc = data.Document(text="Hello.", document_id="myid")
    chunk_iter = chunking.ChunkIterator(
        text=tokenizer.tokenize("Hello."),
        max_char_buffer=200,
        tokenizer_impl=tokenizer.RegexTokenizer(),
        document=doc,
    )
    chunk = next(chunk_iter)
    result = str(chunk)
    self.assertIn("myid", result)
    self.assertIn("Hello.", result)


# ---------------------------------------------------------------------------
# TextChunk.document_id / document_text property coverage
# ---------------------------------------------------------------------------
class TextChunkPropertyCoverageTest(absltest.TestCase):
  """Ensure document_id and document_text return None when document is None."""

  def test_document_id_none_when_no_document(self):
    chunk = chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(start_index=0, end_index=1),
        document=None,
    )
    self.assertIsNone(chunk.document_id)

  def test_document_text_none_when_no_document(self):
    chunk = chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(start_index=0, end_index=1),
        document=None,
    )
    self.assertIsNone(chunk.document_text)


if __name__ == "__main__":
  absltest.main()
