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

from absl.testing import absltest
from absl.testing import parameterized

from langextract import chunking
from langextract import data
from langextract import tokenizer


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
    chunk_iter = chunking.ChunkIterator(tokenized_text, max_char_buffer=50)
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
    chunk_iter = chunking.ChunkIterator(tokenized_text, max_char_buffer=12)
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
    chunk_iter = chunking.ChunkIterator(tokenized_text, max_char_buffer=7)
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

    chunk_iter = chunking.ChunkIterator(tokenized_text, max_char_buffer=20)
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
    chunk_iter = chunking.ChunkIterator(tokenized_text, max_char_buffer=200)
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
    chunk_iter = chunking.ChunkIterator(tokenized_text, max_char_buffer=200)

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
                      start_index=7, end_index=8
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=8, end_index=12
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=12, end_index=17
                  ),
                  document=_SAMPLE_DOCUMENT,
              ),
              chunking.TextChunk(
                  token_interval=tokenizer.TokenInterval(
                      start_index=17, end_index=20
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
    chunk_iter = chunking.ChunkIterator(tokenized_text, max_char_buffer)
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
        tokenized_text, max_char_buffer=7, document=document
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
        text=document.tokenized_text, max_char_buffer=100, document=document
    )
    text_chunk = next(chunk_iter)
    self.assertEqual(text_chunk.additional_context, self._ADDITIONAL_CONTEXT)

  def test_chunk_iterator_without_additional_context(self):
    document = data.Document(text="Sample text.")
    chunk_iter = chunking.ChunkIterator(
        text=document.tokenized_text, max_char_buffer=100, document=document
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


if __name__ == "__main__":
  absltest.main()
