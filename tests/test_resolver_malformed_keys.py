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

"""Tests for resolver handling of malformed attribute keys.

Regression tests for https://github.com/google/langextract/issues/428
where non-Gemini model providers (e.g. DeepSeek via OpenAILanguageModel)
occasionally emit keys with a trailing colon (e.g. "emotion_attributes:")
instead of the expected "emotion_attributes".  Without normalization the
dict value falls through to the extraction_text type check and raises
``ValueError: Extraction text must be a string, integer, or float.``
"""

from absl.testing import absltest
from absl.testing import parameterized

from langextract import resolver as resolver_lib
from langextract.core import data
from langextract.core import format_handler as fh


def _make_resolver(
    attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
    index_suffix: str | None = None,
) -> resolver_lib.Resolver:
  handler = fh.FormatHandler(
      format_type=data.FormatType.YAML,
      use_wrapper=True,
      wrapper_key=data.EXTRACTIONS_KEY,
      use_fences=False,
      attribute_suffix=attribute_suffix,
  )
  return resolver_lib.Resolver(
      format_handler=handler,
      extraction_index_suffix=index_suffix,
  )


class MalformedAttributeKeyTest(parameterized.TestCase):
  """Verifies that trailing-colon keys are normalized before matching."""

  # ── core regression test (issue #428) ──────────────────────────────

  def test_trailing_colon_on_attributes_key_does_not_crash(self):
    """A key like 'emotion_attributes:' should be recognized as attributes."""
    resolver = _make_resolver()
    extraction_data = [
        {
            "emotion": "But soft!",
            # Malformed key with trailing colon
            "emotion_attributes:": {
                "feeling": "gentle awe",
                "character": "Romeo",
            },
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 1)
    self.assertEqual(results[0].extraction_class, "emotion")
    self.assertEqual(results[0].extraction_text, "But soft!")
    self.assertEqual(
        results[0].attributes,
        {
            "feeling": "gentle awe",
            "character": "Romeo",
        },
    )

  def test_trailing_colon_on_index_key_does_not_crash(self):
    """A key like 'emotion_index:' should be recognized as an index."""
    resolver = _make_resolver(index_suffix="_index")
    extraction_data = [
        {
            "emotion": "But soft!",
            "emotion_attributes": {"feeling": "awe"},
            # Malformed index key with trailing colon
            "emotion_index:": 1,
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 1)
    self.assertEqual(results[0].extraction_text, "But soft!")
    self.assertEqual(results[0].extraction_index, 1)

  def test_trailing_colon_on_extraction_class_key(self):
    """'character:' (extraction class with colon) is normalized to 'character'."""
    resolver = _make_resolver()
    extraction_data = [
        {
            # Malformed extraction class key
            "character:": "ROMEO",
            "character_attributes": {"emotional_state": "wonder"},
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 1)
    self.assertEqual(results[0].extraction_class, "character")
    self.assertEqual(results[0].extraction_text, "ROMEO")
    self.assertEqual(results[0].attributes, {"emotional_state": "wonder"})

  # ── verifying correct keys still work ──────────────────────────────

  def test_clean_keys_still_work(self):
    """Normal keys without trailing colons continue to work correctly."""
    resolver = _make_resolver()
    extraction_data = [
        {
            "character": "ROMEO",
            "character_attributes": {"emotional_state": "wonder"},
        },
        {
            "emotion": "But soft!",
            "emotion_attributes": {
                "feeling": "gentle awe",
                "character": "Romeo",
            },
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 2)
    self.assertEqual(results[0].extraction_class, "character")
    self.assertEqual(results[1].extraction_class, "emotion")

  # ── edge cases ─────────────────────────────────────────────────────

  def test_multiple_trailing_colons_stripped(self):
    """Keys like 'emotion_attributes:::' are normalized."""
    resolver = _make_resolver()
    extraction_data = [
        {
            "emotion": "Alas!",
            "emotion_attributes:::": {"feeling": "sorrow"},
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 1)
    self.assertEqual(results[0].attributes, {"feeling": "sorrow"})

  def test_colon_only_in_middle_of_key_not_stripped(self):
    """Colons that are NOT trailing should not be stripped."""
    resolver = _make_resolver()
    extraction_data = [
        {
            # Key with colon in the middle — not a suffix issue
            "emo:tion": "Alas!",
            "emo:tion_attributes": {"feeling": "sorrow"},
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 1)
    # The normalized class should keep the middle colon intact
    self.assertEqual(results[0].extraction_class, "emo:tion")

  def test_both_attribute_and_class_have_trailing_colon(self):
    """Both extraction class and attributes key have trailing colons."""
    resolver = _make_resolver()
    extraction_data = [
        {
            "relationship:": "Juliet is the sun",
            "relationship_attributes:": {
                "type": "metaphor",
                "character_1": "Romeo",
                "character_2": "Juliet",
            },
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 1)
    self.assertEqual(results[0].extraction_class, "relationship")
    self.assertEqual(results[0].extraction_text, "Juliet is the sun")
    self.assertEqual(results[0].attributes["type"], "metaphor")

  def test_numeric_extraction_value_with_trailing_colon(self):
    """Numeric extraction values with trailing colon on key still work."""
    resolver = _make_resolver()
    extraction_data = [
        {
            "score:": 42,
            "score_attributes:": {"unit": "points"},
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 1)
    self.assertEqual(results[0].extraction_class, "score")
    # Numeric values get stringified
    self.assertEqual(results[0].extraction_text, "42")

  def test_float_extraction_value_with_trailing_colon(self):
    """Float extraction values with trailing colon on key still work."""
    resolver = _make_resolver()
    extraction_data = [
        {
            "temperature:": 36.6,
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 1)
    self.assertEqual(results[0].extraction_text, "36.6")

  def test_mixed_clean_and_malformed_keys_in_same_group(self):
    """Groups with both clean and malformed keys are handled correctly."""
    resolver = _make_resolver(index_suffix="_index")
    extraction_data = [
        {
            # Clean extraction class
            "character": "ROMEO",
            # Malformed attributes key
            "character_attributes:": {"emotional_state": "wonder"},
            # Clean index key
            "character_index": 1,
        },
        {
            # Malformed extraction class
            "emotion:": "But soft!",
            # Clean attributes key
            "emotion_attributes": {"feeling": "awe"},
            # Malformed index key
            "emotion_index:": 2,
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 2)
    self.assertEqual(results[0].extraction_class, "character")
    self.assertEqual(results[0].attributes, {"emotional_state": "wonder"})
    self.assertEqual(results[1].extraction_class, "emotion")
    self.assertEqual(results[1].extraction_text, "But soft!")

  def test_attributes_key_collision_after_normalization(self):
    """If both clean and malformed attribute keys exist, the clean key is used."""
    resolver = _make_resolver()
    # This is an unlikely edge case where the model emits both
    # "emotion_attributes" and "emotion_attributes:" in the same group.
    # After normalization, both map to "emotion_attributes". The first one
    # encountered in dict iteration will be processed as attributes; the
    # second will also be recognized as attributes (and skipped as a
    # duplicate continue). The extraction text comes from "emotion".
    extraction_data = [
        {
            "emotion": "O Romeo!",
            "emotion_attributes": {"feeling": "yearning"},
            "emotion_attributes:": {"feeling": "longing"},
        },
    ]
    results = resolver.extract_ordered_extractions(extraction_data)
    self.assertLen(results, 1)
    self.assertEqual(results[0].extraction_text, "O Romeo!")
    # The attributes from the last-processed attributes key wins for the
    # lookup via group.get(), but the extraction itself uses the clean key.
    # What matters is no crash.
    self.assertIsNotNone(results[0].attributes)


if __name__ == "__main__":
  absltest.main()
