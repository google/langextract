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

"""Test for LCS fuzzy alignment IndexError handling bug.

This test verifies that _lcs_fuzzy_align_extraction handles IndexError
consistently with _fuzzy_align_extraction (the legacy algorithm).
"""

from absl.testing import absltest

from langextract import resolver as resolver_lib
from langextract.core import data
from langextract.core import tokenizer


class LcsFuzzyAlignmentIndexErrorTest(absltest.TestCase):
    """Test that LCS fuzzy alignment handles index errors gracefully."""

    def setUp(self):
        super().setUp()
        self.aligner = resolver_lib.WordAligner()

    def test_legacy_fuzzy_alignment_handles_index_error(self):
        """Verify that _fuzzy_align_extraction handles IndexError gracefully.
        
        This is the control test - the legacy algorithm already has
        proper exception handling.
        """
        extraction = data.Extraction(
            extraction_class="test",
            extraction_text="hello",
        )
        
        source_tokens = ["hello", "world"]
        
        tokenized_text = tokenizer.TokenizedText(
            text="hi",
            tokens=[
                tokenizer.Token(
                    index=0,
                    token_type=tokenizer.TokenType.WORD,
                    char_interval=tokenizer.CharInterval(start_pos=0, end_pos=2),
                )
            ],
        )
        
        result = self.aligner._fuzzy_align_extraction(
            extraction=extraction,
            source_tokens=source_tokens,
            tokenized_text=tokenized_text,
            token_offset=0,
            char_offset=0,
            fuzzy_alignment_threshold=0.5,
            tokenizer_impl=None,
        )
        
        self.assertIsNone(result, "Legacy algorithm should return None on IndexError")

    def test_lcs_fuzzy_alignment_handles_index_error(self):
        """Verify that _lcs_fuzzy_align_extraction handles IndexError gracefully.
        
        This is the bug test - _lcs_fuzzy_align_extraction should have
        the same exception handling as _fuzzy_align_extraction.
        
        Bug scenario:
        - source_tokens_norm has length 3 (tokens: ["hello", "world", "target"])
        - extraction_text = "target" (matches at index 2 in source_tokens_norm)
        - tokenized_text.tokens has length 2 (only indices 0 and 1)
        - When LCS finds a match at index 2, trying to access
          tokenized_text.tokens[2] will raise IndexError
        
        Before fix: This test will raise IndexError
        After fix: This test should return None gracefully
        """
        extraction = data.Extraction(
            extraction_class="test",
            extraction_text="target",
        )
        
        source_tokens_norm = ["hello", "world", "target"]
        
        tokenized_text = tokenizer.TokenizedText(
            text="hi world",
            tokens=[
                tokenizer.Token(
                    index=0,
                    token_type=tokenizer.TokenType.WORD,
                    char_interval=tokenizer.CharInterval(start_pos=0, end_pos=2),
                ),
                tokenizer.Token(
                    index=1,
                    token_type=tokenizer.TokenType.WORD,
                    char_interval=tokenizer.CharInterval(start_pos=3, end_pos=8),
                ),
            ],
        )
        
        try:
            result = self.aligner._lcs_fuzzy_align_extraction(
                extraction=extraction,
                source_tokens_norm=source_tokens_norm,
                tokenized_text=tokenized_text,
                token_offset=0,
                char_offset=0,
                fuzzy_alignment_threshold=0.75,
                fuzzy_alignment_min_density=1/3,
                tokenizer_impl=None,
            )
            
            self.assertIsNone(
                result, 
                "LCS algorithm should return None gracefully on IndexError, "
                "consistent with legacy algorithm behavior"
            )
        except IndexError as e:
            self.fail(
                f"_lcs_fuzzy_align_extraction raised IndexError instead of "
                f"handling it gracefully: {e}\n"
                f"This is inconsistent with _fuzzy_align_extraction which has "
                f"try-except protection.\n"
                f"See resolver.py:681-700 for the legacy implementation with "
                f"proper error handling."
            )


if __name__ == "__main__":
    absltest.main()
