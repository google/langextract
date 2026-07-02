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

import textwrap
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract import chunking
from langextract.core import data
from langextract.core import tokenizer


class SentenceIterTest(absltest.TestCase):

  def test_basic(self):
    text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?"
    tokenized_text = tokenizer.tokenize(text)
    sentence_iter = chunking.SentenceIterator(tokenized_text)
    sentence_interval = next(sentence_iter)
    self.assertEqual(
        tokenizer.TokenInterval(start_index=0, end_index=5), sentence_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, sentence_interval),
        "This is a sentence.",
    )
    sentence_interval = next(sentence_iter)
    self.assertEqual(
        tokenizer.TokenInterval(start_index=5, end_index=11), sentence_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, sentence_interval),
        "This is a longer sentence.",
    )
    sentence_interval = next(sentence_iter)
    self.assertEqual(
        tokenizer.TokenInterval(start_index=11, end_index=17), sentence_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, sentence_interval),
        "Mr. Bond\nasks\nwhy?",
    )
    with self.assertRaises(StopIteration):
      next(sentence_iter)

  def test_empty(self):
    text = ""
    tokenized_text = tokenizer.tokenize(text)
    sentence_iter = chunking.SentenceIterator(tokenized_text)
    with self.assertRaises(StopIteration):
      next(sentence_iter)


class ChunkIteratorTest(absltest.TestCase):

  def test_multi_sentence_chunk(self):
    text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?"
    tokenized_text = tokenizer.tokenize(text)
    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer=50,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=0, end_index=11), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        "This is a sentence. This is a longer sentence.",
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=11, end_index=17), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        "Mr. Bond\nasks\nwhy?",
    )
    with self.assertRaises(StopIteration):
      next(chunk_iter)

  def test_sentence_with_multiple_newlines_and_right_interval(self):
    text = (
        "This is a sentence\n\n"
        + "This is a longer sentence\n\n"
        + "Mr\n\nBond\n\nasks why?"
    )
    tokenized_text = tokenizer.tokenize(text)
    chunk_interval = tokenizer.TokenInterval(
        start_index=0, end_index=len(tokenized_text.tokens)
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        text,
    )

  def test_break_sentence(self):
    text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?"
    tokenized_text = tokenizer.tokenize(text)
    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer=12,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=0, end_index=3), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        "This is a",
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=3, end_index=5), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        "sentence.",
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=5, end_index=8), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        "This is a",
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=8, end_index=9), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        "longer",
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=9, end_index=11), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        "sentence.",
    )
    for _ in range(2):
      next(chunk_iter)
    with self.assertRaises(StopIteration):
      next(chunk_iter)

  def test_long_token_gets_own_chunk(self):
    text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?"
    tokenized_text = tokenizer.tokenize(text)
    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer=7,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=0, end_index=2), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        "This is",
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=2, end_index=3), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval), "a"
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=3, end_index=4), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval),
        "sentence",
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(start_index=4, end_index=5), chunk_interval
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval), "."
    )
    for _ in range(9):
      next(chunk_iter)
    with self.assertRaises(StopIteration):
      next(chunk_iter)

  def test_newline_at_chunk_boundary_does_not_create_empty_interval(self):
    """Test that newlines at chunk boundaries don't create empty token intervals.

    When a newline occurs exactly at a chunk boundary, the chunking algorithm
    should not attempt to create an empty interval (where start_index == end_index).
    This was causing a ValueError in create_token_interval().
    """
    text = "First sentence.\nSecond sentence that is longer.\nThird sentence."
    tokenized_text = tokenizer.tokenize(text)

    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer=20,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunks = list(chunk_iter)

    for chunk in chunks:
      self.assertLess(
          chunk.token_interval.start_index,
          chunk.token_interval.end_index,
          "Chunk should have non-empty interval",
      )

    expected_intervals = [(0, 3), (3, 6), (6, 9), (9, 12)]
    actual_intervals = [
        (chunk.token_interval.start_index, chunk.token_interval.end_index)
        for chunk in chunks
    ]
    self.assertEqual(actual_intervals, expected_intervals)

  def test_chunk_unicode_text(self):
    text = textwrap.dedent("""\
    Chief Complaint:
    ‘swelling of tongue and difficulty breathing and swallowing’
    History of Present Illness:
    77 y o woman in NAD with a h/o CAD, DM2, asthma and HTN on altace.""")
    tokenized_text = tokenizer.tokenize(text)
    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer=200,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk_interval = next(chunk_iter).token_interval
    self.assertEqual(
        tokenizer.TokenInterval(
            start_index=0, end_index=len(tokenized_text.tokens)
        ),
        chunk_interval,
    )
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, chunk_interval), text
    )

  def test_newlines_is_secondary_sentence_break(self):
    text = textwrap.dedent("""\
    Medications:
    Theophyline (Uniphyl) 600 mg qhs – bronchodilator by increasing cAMP used
    for treating asthma
    Diltiazem 300 mg qhs – Ca channel blocker used to control hypertension
    Simvistatin (Zocor) 20 mg qhs- HMGCo Reductase inhibitor for
    hypercholesterolemia
    Ramipril (Altace) 10 mg BID – ACEI for hypertension and diabetes for
    renal protective effect""")
    tokenized_text = tokenizer.tokenize(text)
    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer=200,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )

    first_chunk = next(chunk_iter)
    expected_first_chunk_text = textwrap.dedent("""\
    Medications:
    Theophyline (Uniphyl) 600 mg qhs – bronchodilator by increasing cAMP used
    for treating asthma
    Diltiazem 300 mg qhs – Ca channel blocker used to control hypertension""")
    self.assertEqual(
        chunking.get_token_interval_text(
            tokenized_text, first_chunk.token_interval
        ),
        expected_first_chunk_text,
    )

    self.assertGreater(
        first_chunk.token_interval.end_index,
        first_chunk.token_interval.start_index,
    )

    second_chunk = next(chunk_iter)
    expected_second_chunk_text = textwrap.dedent("""\
    Simvistatin (Zocor) 20 mg qhs- HMGCo Reductase inhibitor for
    hypercholesterolemia
    Ramipril (Altace) 10 mg BID – ACEI for hypertension and diabetes for
    renal protective effect""")
    self.assertEqual(
        chunking.get_token_interval_text(
            tokenized_text, second_chunk.token_interval
        ),
        expected_second_chunk_text,
    )

    with self.assertRaises(StopIteration):
      next(chunk_iter)

  def test_tokenizer_propagation(self):
    """Test that tokenizer is correctly propagated to TextChunk's Document."""
    text = "Some text."
    mock_tokenizer = mock.Mock(spec=tokenizer.Tokenizer)
    mock_tokens = [
        tokenizer.Token(
            index=0,
            token_type=tokenizer.TokenType.WORD,
            char_interval=data.CharInterval(start_pos=0, end_pos=4),
        ),
        tokenizer.Token(
            index=1,
            token_type=tokenizer.TokenType.WORD,
            char_interval=data.CharInterval(start_pos=5, end_pos=9),
        ),
        tokenizer.Token(
            index=2,
            token_type=tokenizer.TokenType.PUNCTUATION,
            char_interval=data.CharInterval(start_pos=9, end_pos=10),
        ),
    ]
    mock_tokenized_text = tokenizer.TokenizedText(text=text, tokens=mock_tokens)
    mock_tokenizer.tokenize.return_value = mock_tokenized_text

    chunk_iter = chunking.ChunkIterator(
        text=text, max_char_buffer=100, tokenizer_impl=mock_tokenizer
    )
    text_chunk = next(chunk_iter)

    self.assertEqual(text_chunk.document_text, mock_tokenized_text)
    self.assertEqual(text_chunk.chunk_text, text)


class BatchingTest(parameterized.TestCase):

  _SAMPLE_DOCUMENT = data.Document(
      text=(
          "Sample text with numerical values such as 120/80 mmHg, 98.6°F, and"
          " 50mg."
      ),
  )

  @parameterized.named_parameters(
      (
          "test_with_data",
          _SAMPLE_DOCUMENT.tokenized_text,
          15,
          10,
          [[
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=0, end_index=1
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=1, end_index=3
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=3, end_index=4
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=4, end_index=5
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=5, end_index=7
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=7, end_index=10
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=10, end_index=14
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=14, end_index=19
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=19, end_index=22
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
          ]],
      ),
      (
          "test_empty_input",
          "",
          15,
          10,
          [],
      ),
  )
  def test_make_batches_of_textchunk(
      self,
      tokenized_text: tokenizer.TokenizedText,
      batch_length: int,
      max_char_buffer: int,
      expected_batches: list[list[chunking.TextChunk]],
  ):
    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    batches_iter = chunking.make_batches_of_textchunk(chunk_iter, batch_length)
    actual_batches = [list(batch) for batch in batches_iter]

    self.assertListEqual(
        actual_batches,
        expected_batches,
        "Batched chunks should match expected structure",
    )


class TextChunkTest(absltest.TestCase):

  def test_string_output(self):
    text = "Example input text."
    expected = textwrap.dedent("""\
    TextChunk(
      interval=[start_index: 0, end_index: 1],
      Document ID: test_doc_123,
      Chunk Text: 'Example'
    )""")
    document = data.Document(text=text, document_id="test_doc_123")
    tokenized_text = tokenizer.tokenize(text)
    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer=7,
        document=document,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    text_chunk = next(chunk_iter)
    self.assertEqual(str(text_chunk), expected)


class TextAdditionalContextTest(absltest.TestCase):

  _ADDITIONAL_CONTEXT = "Some additional context for prompt..."

  def test_text_chunk_additional_context(self):
    document = data.Document(
        text="Sample text.", additional_context=self._ADDITIONAL_CONTEXT
    )
    chunk_iter = chunking.ChunkIterator(
        text=document.tokenized_text,
        max_char_buffer=100,
        document=document,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    text_chunk = next(chunk_iter)
    self.assertEqual(text_chunk.additional_context, self._ADDITIONAL_CONTEXT)

  def test_chunk_iterator_without_additional_context(self):
    document = data.Document(text="Sample text.")
    chunk_iter = chunking.ChunkIterator(
        text=document.tokenized_text,
        max_char_buffer=100,
        document=document,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    text_chunk = next(chunk_iter)
    self.assertIsNone(text_chunk.additional_context)

  def test_multiple_chunks_with_additional_context(self):
    text = "Sentence one. Sentence two. Sentence three."
    document = data.Document(
        text=text, additional_context=self._ADDITIONAL_CONTEXT
    )
    chunk_iter = chunking.ChunkIterator(
        text=document.tokenized_text,
        max_char_buffer=15,
        document=document,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunks = list(chunk_iter)
    self.assertGreater(
        len(chunks), 1, "Should create multiple chunks with small buffer"
    )
    additional_contexts = [chunk.additional_context for chunk in chunks]
    expected_additional_contexts = [self._ADDITIONAL_CONTEXT] * len(chunks)
    self.assertListEqual(additional_contexts, expected_additional_contexts)


class TextChunkPropertyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "with_document",
          "document": data.Document(
              text="Sample text.",
              document_id="doc123",
              additional_context="Additional info",
          ),
          "expected_id": "doc123",
          "expected_text": "Sample text.",
          "expected_context": "Additional info",
      },
      {
          "testcase_name": "no_document",
          "document": None,
          "expected_id": None,
          "expected_text": None,
          "expected_context": None,
      },
      {
          "testcase_name": "no_additional_context",
          "document": data.Document(
              text="Sample text.",
              document_id="doc123",
          ),
          "expected_id": "doc123",
          "expected_text": "Sample text.",
          "expected_context": None,
      },
  )
  def test_text_chunk_properties(
      self, document, expected_id, expected_text, expected_context
  ):
    chunk = chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(start_index=0, end_index=1),
        document=document,
    )
    self.assertEqual(chunk.document_id, expected_id)
    if chunk.document_text:
      self.assertEqual(chunk.document_text.text, expected_text)
    else:
      self.assertIsNone(chunk.document_text)
    self.assertEqual(chunk.additional_context, expected_context)


class SentenceIteratorEdgeCasesTest(absltest.TestCase):

  def test_negative_curr_token_pos_raises_index_error(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    with self.assertRaises(IndexError):
      chunking.SentenceIterator(tokenized_text, curr_token_pos=-1)

  def test_curr_token_pos_beyond_length_raises_index_error(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    with self.assertRaises(IndexError):
      chunking.SentenceIterator(
          tokenized_text,
          curr_token_pos=len(tokenized_text.tokens) + 1,
      )

  def test_curr_token_pos_at_length_raises_stop_iteration(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    sentence_iter = chunking.SentenceIterator(
        tokenized_text,
        curr_token_pos=len(tokenized_text.tokens),
    )
    with self.assertRaises(StopIteration):
      next(sentence_iter)

  def test_mid_document_start(self):
    # "First sentence." = [First, sentence, .] = 3 tokens (indices 0-2).
    # "Second sentence." starts at index 3.
    text = "First sentence. Second sentence."
    tokenized_text = tokenizer.tokenize(text)
    sentence_iter = chunking.SentenceIterator(tokenized_text, curr_token_pos=3)
    sentence_interval = next(sentence_iter)
    self.assertEqual(sentence_interval.start_index, 3)
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, sentence_interval),
        "Second sentence.",
    )

  def test_text_without_punctuation_is_one_sentence(self):
    text = "This text has no punctuation at all"
    tokenized_text = tokenizer.tokenize(text)
    sentence_iter = chunking.SentenceIterator(tokenized_text)
    sentence_interval = next(sentence_iter)
    self.assertEqual(
        chunking.get_token_interval_text(tokenized_text, sentence_interval),
        text,
    )
    with self.assertRaises(StopIteration):
      next(sentence_iter)


class ChunkIteratorConstructorTest(absltest.TestCase):

  def test_no_text_and_no_document_raises_value_error(self):
    with self.assertRaises(ValueError):
      chunking.ChunkIterator(
          text=None,
          max_char_buffer=100,
          tokenizer_impl=tokenizer.RegexTokenizer(),
      )

  def test_none_text_uses_document_text(self):
    document = data.Document(text="Hello world.", document_id="doc1")
    chunk_iter = chunking.ChunkIterator(
        text=None,
        max_char_buffer=100,
        document=document,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk = next(chunk_iter)
    self.assertEqual(chunk.chunk_text, "Hello world.")

  def test_empty_tokenized_text_retokenizes_from_document(self):
    # TokenizedText with no tokens should trigger re-tokenization using
    # document.text as fallback.
    document = data.Document(text="Hello world.")
    empty_tokenized = tokenizer.TokenizedText(text="", tokens=[])
    chunk_iter = chunking.ChunkIterator(
        text=empty_tokenized,
        max_char_buffer=100,
        document=document,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk = next(chunk_iter)
    self.assertEqual(chunk.chunk_text, "Hello world.")

  def test_exact_buffer_size_fits_in_one_chunk(self):
    # "Hello world." is 12 chars; max_char_buffer=12 uses > (not >=),
    # so the text should fit in a single chunk.
    text = "Hello world."
    tokenized_text = tokenizer.tokenize(text)
    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer=12,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunks = list(chunk_iter)
    self.assertLen(chunks, 1)
    self.assertEqual(chunks[0].chunk_text, text)


class CreateTokenIntervalTest(absltest.TestCase):

  def test_negative_start_index_raises(self):
    with self.assertRaises(ValueError):
      chunking.create_token_interval(-1, 5)

  def test_equal_indices_raises(self):
    with self.assertRaises(ValueError):
      chunking.create_token_interval(3, 3)

  def test_start_greater_than_end_raises(self):
    with self.assertRaises(ValueError):
      chunking.create_token_interval(5, 3)


class GetTokenIntervalTextTest(absltest.TestCase):

  def test_invalid_interval_raises_value_error(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    invalid_interval = tokenizer.TokenInterval(start_index=2, end_index=2)
    with self.assertRaises(ValueError):
      chunking.get_token_interval_text(tokenized_text, invalid_interval)

  def test_token_util_error_on_empty_return(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    valid_interval = tokenizer.TokenInterval(start_index=0, end_index=2)
    with mock.patch("langextract.core.tokenizer.tokens_text", return_value=""):
      with self.assertRaises(chunking.TokenUtilError):
        chunking.get_token_interval_text(tokenized_text, valid_interval)


class GetCharIntervalTest(absltest.TestCase):

  def test_invalid_interval_raises_value_error(self):
    tokenized_text = tokenizer.tokenize("Hello world.")
    invalid_interval = tokenizer.TokenInterval(start_index=2, end_index=2)
    with self.assertRaises(ValueError):
      chunking.get_char_interval(tokenized_text, invalid_interval)


class TextChunkMissingDocumentTest(absltest.TestCase):

  def test_chunk_text_raises_when_no_document(self):
    chunk = chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(start_index=0, end_index=1),
        document=None,
    )
    with self.assertRaises(ValueError):
      _ = chunk.chunk_text

  def test_char_interval_raises_when_no_document(self):
    chunk = chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(start_index=0, end_index=1),
        document=None,
    )
    with self.assertRaises(ValueError):
      _ = chunk.char_interval

  def test_str_shows_unavailable_when_no_document(self):
    chunk = chunking.TextChunk(
        token_interval=tokenizer.TokenInterval(start_index=0, end_index=1),
        document=None,
    )
    self.assertIn("<unavailable: document_text not set>", str(chunk))


class SanitizeTest(absltest.TestCase):

  def test_whitespace_only_raises_value_error(self):
    with self.assertRaises(ValueError):
      chunking._sanitize("   \n\t  ")

  def test_mixed_whitespace_collapsed_to_single_space(self):
    result = chunking._sanitize("hello\n\t  world")
    self.assertEqual(result, "hello world")

  def test_leading_trailing_whitespace_stripped(self):
    result = chunking._sanitize("  hello world  ")
    self.assertEqual(result, "hello world")


class SanitizedChunkTextTest(absltest.TestCase):

  def test_sanitized_chunk_text_collapses_whitespace(self):
    text = "Hello\n  world."
    document = data.Document(text=text)
    chunk_iter = chunking.ChunkIterator(
        text=document.tokenized_text,
        max_char_buffer=200,
        document=document,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunk = next(chunk_iter)
    self.assertEqual(chunk.sanitized_chunk_text, "Hello world.")


class ChunkCachingTest(absltest.TestCase):

  def _make_chunk(self) -> chunking.TextChunk:
    text = "Hello world."
    document = data.Document(text=text)
    chunk_iter = chunking.ChunkIterator(
        text=document.tokenized_text,
        max_char_buffer=200,
        document=document,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    return next(chunk_iter)

  def test_chunk_text_is_cached(self):
    chunk = self._make_chunk()
    with mock.patch(
        "langextract.chunking.get_token_interval_text",
        wraps=chunking.get_token_interval_text,
    ) as mock_fn:
      _ = chunk.chunk_text
      _ = chunk.chunk_text
    mock_fn.assert_called_once()

  def test_char_interval_is_cached(self):
    chunk = self._make_chunk()
    first_call = chunk.char_interval
    second_call = chunk.char_interval
    self.assertIs(first_call, second_call)


class MakeBatchesAdditionalTest(absltest.TestCase):

  def _make_chunk_iter(self, text, max_char_buffer):
    document = data.Document(text=text)
    return chunking.ChunkIterator(
        text=document.tokenized_text,
        max_char_buffer=max_char_buffer,
        document=document,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )

  def test_batch_length_one_puts_each_chunk_in_own_batch(self):
    chunk_iter = self._make_chunk_iter("One. Two. Three.", max_char_buffer=6)
    batches = [
        list(b) for b in chunking.make_batches_of_textchunk(chunk_iter, 1)
    ]
    for batch in batches:
      self.assertLen(batch, 1)
    self.assertGreater(len(batches), 1)

  def test_batch_length_larger_than_chunks_produces_one_batch(self):
    chunk_iter = self._make_chunk_iter("Hello.", max_char_buffer=100)
    batches = [
        list(b) for b in chunking.make_batches_of_textchunk(chunk_iter, 1000)
    ]
    self.assertLen(batches, 1)


class BrokenSentenceResetTest(absltest.TestCase):

  def test_merging_resumes_after_broken_sentence(self):
    # "Word word word word." (20 chars) exceeds max_char_buffer=15 and is
    # broken across chunks. Afterwards, "Hi." and "Bye." are each short enough
    # to merge and should appear together in a single final chunk.
    text = "Word word word word. Hi. Bye."
    tokenized_text = tokenizer.tokenize(text)
    chunk_iter = chunking.ChunkIterator(
        tokenized_text,
        max_char_buffer=15,
        tokenizer_impl=tokenizer.RegexTokenizer(),
    )
    chunks = list(chunk_iter)
    last_chunk_text = chunks[-1].chunk_text
    self.assertIn("Hi.", last_chunk_text)
    self.assertIn("Bye.", last_chunk_text)


if __name__ == "__main__":
  absltest.main()
