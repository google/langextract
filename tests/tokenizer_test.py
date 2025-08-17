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

from langextract.core import tokenizer


class TokenizerTest(parameterized.TestCase):

  def assertTokenListEqual(self, actual_tokens, expected_tokens, msg=None):
    self.assertLen(actual_tokens, len(expected_tokens), msg=msg)
    for i, (expected, actual) in enumerate(zip(expected_tokens, actual_tokens)):
      expected = tokenizer.Token(
          index=expected.index,
          token_type=expected.token_type,
          first_token_after_newline=expected.first_token_after_newline,
      )
      actual = tokenizer.Token(
          index=actual.index,
          token_type=actual.token_type,
          first_token_after_newline=actual.first_token_after_newline,
      )
      self.assertDataclassEqual(
          expected,
          actual,
          msg=f"Token mismatch at index {i}",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="basic_text",
          input_text="Hello, world!",
          expected_tokens=[
              tokenizer.Token(index=0, token_type=tokenizer.TokenType.WORD),
              tokenizer.Token(
                  index=1, token_type=tokenizer.TokenType.PUNCTUATION
              ),
              tokenizer.Token(index=2, token_type=tokenizer.TokenType.WORD),
              tokenizer.Token(
                  index=3, token_type=tokenizer.TokenType.PUNCTUATION
              ),
          ],
      ),
      dict(
          testcase_name="multiple_spaces_and_numbers",
          input_text="Age:   25\nWeight=70kg.",
          expected_tokens=[
              tokenizer.Token(index=0, token_type=tokenizer.TokenType.WORD),
              tokenizer.Token(
                  index=1, token_type=tokenizer.TokenType.PUNCTUATION
              ),
              tokenizer.Token(index=2, token_type=tokenizer.TokenType.NUMBER),
              tokenizer.Token(
                  index=3,
                  token_type=tokenizer.TokenType.WORD,
                  first_token_after_newline=True,
              ),
              tokenizer.Token(
                  index=4, token_type=tokenizer.TokenType.PUNCTUATION
              ),
              tokenizer.Token(index=5, token_type=tokenizer.TokenType.NUMBER),
              tokenizer.Token(index=6, token_type=tokenizer.TokenType.WORD),
              tokenizer.Token(
                  index=7, token_type=tokenizer.TokenType.PUNCTUATION
              ),
          ],
      ),
      dict(
          testcase_name="multi_line_input",
          input_text="Line1\nLine2\nLine3",
          expected_tokens=[
              tokenizer.Token(index=0, token_type=tokenizer.TokenType.WORD),
              tokenizer.Token(index=1, token_type=tokenizer.TokenType.NUMBER),
              tokenizer.Token(
                  index=2,
                  token_type=tokenizer.TokenType.WORD,
                  first_token_after_newline=True,
              ),
              tokenizer.Token(index=3, token_type=tokenizer.TokenType.NUMBER),
              tokenizer.Token(
                  index=4,
                  token_type=tokenizer.TokenType.WORD,
                  first_token_after_newline=True,
              ),
              tokenizer.Token(index=5, token_type=tokenizer.TokenType.NUMBER),
          ],
      ),
      dict(
          testcase_name="only_symbols",
          input_text="!!!@#   $$$%",
          expected_tokens=[
              tokenizer.Token(
                  index=0, token_type=tokenizer.TokenType.PUNCTUATION
              ),
              tokenizer.Token(
                  index=1, token_type=tokenizer.TokenType.PUNCTUATION
              ),
          ],
      ),
      dict(
          testcase_name="empty_string",
          input_text="",
          expected_tokens=[],
      ),
  )
  def test_tokenize_various_inputs(self, input_text, expected_tokens):
    tokenized = tokenizer.tokenize(input_text)
    self.assertTokenListEqual(
        tokenized.tokens,
        expected_tokens,
        msg=f"Tokens mismatch for input: {input_text!r}",
    )

  def test_first_token_after_newline_flag(self):
    input_text = "Line1\nLine2\nLine3"
    tokenized = tokenizer.tokenize(input_text)

    expected_tokens = [
        tokenizer.Token(
            index=0,
            token_type=tokenizer.TokenType.WORD,
        ),
        tokenizer.Token(
            index=1,
            token_type=tokenizer.TokenType.NUMBER,
        ),
        tokenizer.Token(
            index=2,
            token_type=tokenizer.TokenType.WORD,
            first_token_after_newline=True,
        ),
        tokenizer.Token(
            index=3,
            token_type=tokenizer.TokenType.NUMBER,
        ),
        tokenizer.Token(
            index=4,
            token_type=tokenizer.TokenType.WORD,
            first_token_after_newline=True,
        ),
        tokenizer.Token(
            index=5,
            token_type=tokenizer.TokenType.NUMBER,
        ),
    ]

    self.assertTokenListEqual(
        tokenized.tokens,
        expected_tokens,
        msg="Newline flags mismatch",
    )


class TokensTextTest(parameterized.TestCase):

  _SENTENCE_WITH_ONE_LINE = "Patient Jane Doe, ID 67890, received 10mg daily."

  @parameterized.named_parameters(
      dict(
          testcase_name="substring_jane_doe",
          input_text=_SENTENCE_WITH_ONE_LINE,
          start_index=1,
          end_index=3,
          expected_substring="Jane Doe",
      ),
      dict(
          testcase_name="substring_with_punctuation",
          input_text=_SENTENCE_WITH_ONE_LINE,
          start_index=0,
          end_index=4,
          expected_substring="Patient Jane Doe,",
      ),
      dict(
          testcase_name="numeric_tokens",
          input_text=_SENTENCE_WITH_ONE_LINE,
          start_index=5,
          end_index=6,
          expected_substring="67890",
      ),
  )
  def test_valid_intervals(
      self, input_text, start_index, end_index, expected_substring
  ):
    input_tokenized = tokenizer.tokenize(input_text)
    interval = tokenizer.TokenInterval(
        start_index=start_index, end_index=end_index
    )
    result_str = tokenizer.tokens_text(input_tokenized, interval)
    self.assertEqual(
        result_str,
        expected_substring,
        msg=f"Wrong substring for interval {start_index}..{end_index}",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="start_index_negative",
          input_text=_SENTENCE_WITH_ONE_LINE,
          start_index=-1,
          end_index=2,
      ),
      dict(
          testcase_name="end_index_out_of_bounds",
          input_text=_SENTENCE_WITH_ONE_LINE,
          start_index=0,
          end_index=999,
      ),
      dict(
          testcase_name="start_index_ge_end_index",
          input_text=_SENTENCE_WITH_ONE_LINE,
          start_index=4,
          end_index=4,
      ),
  )
  def test_invalid_intervals(self, input_text, start_index, end_index):
    input_tokenized = tokenizer.tokenize(input_text)
    interval = tokenizer.TokenInterval(
        start_index=start_index, end_index=end_index
    )
    with self.assertRaises(tokenizer.InvalidTokenIntervalError):
      _ = tokenizer.tokens_text(input_tokenized, interval)


class SentenceRangeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="simple_sentence",
          input_text="This is one sentence. Then another?",
          start_pos=0,
          expected_interval=(0, 5),
      ),
      dict(
          testcase_name="abbreviation_not_boundary",
          input_text="Dr. John visited. Then left.",
          start_pos=0,
          expected_interval=(0, 5),
      ),
      dict(
          testcase_name="second_line_capital_letter_terminates_sentence",
          input_text=textwrap.dedent("""\
              Blood pressure was 160/90 and patient was recommended to
              Atenolol 50 mg daily."""),
          start_pos=0,
          expected_interval=(0, 9),
      ),
  )
  def test_partial_sentence_range(
      self, input_text, start_pos, expected_interval
  ):
    tokenized = tokenizer.tokenize(input_text)
    tokens = tokenized.tokens

    interval = tokenizer.find_sentence_range(input_text, tokens, start_pos)
    expected_start, expected_end = expected_interval
    self.assertEqual(interval.start_index, expected_start)
    self.assertEqual(interval.end_index, expected_end)

  @parameterized.named_parameters(
      dict(
          testcase_name="end_of_text",
          input_text="Only one sentence here",
          start_pos=0,
      ),
  )
  def test_full_sentence_range(self, input_text, start_pos):
    tokenized = tokenizer.tokenize(input_text)
    tokens = tokenized.tokens

    interval = tokenizer.find_sentence_range(input_text, tokens, start_pos)
    self.assertEqual(interval.start_index, 0)
    self.assertLen(tokens, interval.end_index)

  @parameterized.named_parameters(
      dict(
          testcase_name="out_of_range_negative_start",
          input_text="Hello world.",
          start_pos=-1,
      ),
      dict(
          testcase_name="out_of_range_exceeding_length",
          input_text="Hello world.",
          start_pos=999,
      ),
  )
  def test_invalid_start_pos(self, input_text, start_pos):
    tokenized = tokenizer.tokenize(input_text)
    tokens = tokenized.tokens
    with self.assertRaises(tokenizer.SentenceRangeError):
      tokenizer.find_sentence_range(input_text, tokens, start_pos)


if __name__ == "__main__":
  absltest.main()
