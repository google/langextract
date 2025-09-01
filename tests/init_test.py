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

"""Tests for the main package functions in __init__.py."""

import textwrap
from unittest import mock

from absl.testing import absltest

from langextract import prompting
import langextract as lx
from langextract.core import data
from langextract.core import types
from langextract.providers import schemas


class InitTest(absltest.TestCase):
  """Test cases for the main package functions."""

  @mock.patch.object(
      schemas.gemini.GeminiSchema, "from_examples", autospec=True
  )
  @mock.patch("langextract.extraction.factory.create_model")
  def test_lang_extract_as_lx_extract(
      self, mock_create_model, mock_gemini_schema
  ):

    input_text = "Patient takes Aspirin 100mg every morning."

    mock_model = mock.MagicMock()
    mock_model.infer.return_value = [[
        types.ScoredOutput(
            output=textwrap.dedent("""\
            ```json
            {
              "extractions": [
                {
                  "entity": "Aspirin",
                  "entity_attributes": {
                    "class": "medication"
                  }
                },
                {
                  "entity": "100mg",
                  "entity_attributes": {
                    "frequency": "every morning",
                    "class": "dosage"
                  }
                }
              ]
            }
            ```"""),
            score=0.9,
        )
    ]]

    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model

    mock_gemini_schema.return_value = None

    expected_result = data.AnnotatedDocument(
        document_id=None,
        extractions=[
            data.Extraction(
                extraction_class="entity",
                extraction_text="Aspirin",
                char_interval=data.CharInterval(start_pos=14, end_pos=21),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                extraction_index=1,
                group_index=0,
                description=None,
                attributes={"class": "medication"},
            ),
            data.Extraction(
                extraction_class="entity",
                extraction_text="100mg",
                char_interval=data.CharInterval(start_pos=22, end_pos=27),
                alignment_status=data.AlignmentStatus.MATCH_EXACT,
                extraction_index=2,
                group_index=1,
                description=None,
                attributes={"frequency": "every morning", "class": "dosage"},
            ),
        ],
        text="Patient takes Aspirin 100mg every morning.",
    )

    mock_description = textwrap.dedent("""\
        Extract medication and dosage information in order of occurrence.
        """)

    mock_examples = [
        lx.data.ExampleData(
            text="Patient takes Tylenol 500mg daily.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Tylenol",
                    attributes={
                        "type": "analgesic",
                        "class": "medication",
                    },
                ),
            ],
        )
    ]
    mock_prompt_template = prompting.PromptTemplateStructured(
        description=mock_description, examples=mock_examples
    )

    prompt_generator = prompting.QAPromptGenerator(
        template=mock_prompt_template, format_type=lx.data.FormatType.JSON
    )

    actual_result = lx.extract(
        text_or_documents=input_text,
        prompt_description=mock_description,
        examples=mock_examples,
        api_key="some_api_key",
        fence_output=True,
        use_schema_constraints=False,
    )

    mock_gemini_schema.assert_not_called()
    mock_create_model.assert_called_once()
    mock_model.infer.assert_called_once_with(
        batch_prompts=[prompt_generator.render(input_text)],
        max_workers=10,  # Default value from extract()
    )

    self.assertDataclassEqual(expected_result, actual_result)

  @mock.patch("langextract.extraction.resolver.Resolver.align")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_extract_resolver_params_alignment_passthrough(
      self, mock_create_model, mock_align
  ):
    mock_model = mock.MagicMock()
    mock_model.infer.return_value = [
        [types.ScoredOutput(output='{"extractions":[]}')]
    ]
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_align.return_value = []

    mock_examples = [
        lx.data.ExampleData(
            text="Patient takes Tylenol 500mg daily.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Tylenol",
                    attributes={
                        "type": "analgesic",
                        "class": "medication",
                    },
                ),
            ],
        )
    ]

    lx.extract(
        text_or_documents="test text",
        prompt_description="desc",
        examples=mock_examples,
        api_key="test_key",
        resolver_params={
            "enable_fuzzy_alignment": False,
            "fuzzy_alignment_threshold": 0.8,
            "accept_match_lesser": False,
            "extraction_attributes_suffix": "_attrs",
        },
    )

    mock_align.assert_called()
    _, kwargs = mock_align.call_args
    self.assertFalse(kwargs.get("enable_fuzzy_alignment"))
    self.assertEqual(kwargs.get("fuzzy_alignment_threshold"), 0.8)
    self.assertFalse(kwargs.get("accept_match_lesser"))
    self.assertNotIn("extraction_attributes_suffix", kwargs)

  @mock.patch("langextract.extraction.resolver.Resolver")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_extract_resolver_params_none_handling(
      self, mock_create_model, mock_resolver_class
  ):
    mock_model = mock.MagicMock()
    mock_model.infer.return_value = [
        [types.ScoredOutput(output='{"extractions":[]}')]
    ]
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model

    mock_resolver = mock.MagicMock()
    mock_resolver_class.return_value = mock_resolver

    mock_examples = [
        lx.data.ExampleData(
            text="Test text",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="test",
                ),
            ],
        )
    ]

    with mock.patch(
        "langextract.annotation.Annotator.annotate_text"
    ) as mock_annotate:
      mock_annotate.return_value = lx.data.AnnotatedDocument(
          text="test", extractions=[]
      )

      lx.extract(
          text_or_documents="test text",
          prompt_description="desc",
          examples=mock_examples,
          api_key="test_key",
          resolver_params={
              "enable_fuzzy_alignment": None,
              "fuzzy_alignment_threshold": 0.8,
              "extraction_attributes_suffix": "_attrs",
          },
      )

      _, resolver_kwargs = mock_resolver_class.call_args
      self.assertNotIn("enable_fuzzy_alignment", resolver_kwargs)
      self.assertNotIn("fuzzy_alignment_threshold", resolver_kwargs)
      self.assertEqual(
          resolver_kwargs["extraction_attributes_suffix"], "_attrs"
      )

      _, annotate_kwargs = mock_annotate.call_args
      self.assertNotIn("enable_fuzzy_alignment", annotate_kwargs)
      self.assertEqual(annotate_kwargs["fuzzy_alignment_threshold"], 0.8)

  @mock.patch("langextract.extraction.factory.create_model")
  def test_extract_resolver_params_typo_error(self, mock_create_model):
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model

    mock_examples = [
        lx.data.ExampleData(
            text="Test",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="test",
                ),
            ],
        )
    ]

    with self.assertRaisesRegex(TypeError, "Unknown key in resolver_params"):
      lx.extract(
          text_or_documents="test",
          prompt_description="desc",
          examples=mock_examples,
          api_key="test_key",
          resolver_params={
              "fuzzy_alignment_treshold": (
                  0.5
              ),  # Typo: treshold instead of threshold
          },
      )

  @mock.patch("langextract.annotation.Annotator.annotate_documents")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_extract_resolver_params_docs_path_passthrough(
      self, mock_create_model, mock_annotate_docs
  ):
    mock_model = mock.MagicMock()
    mock_model.infer.return_value = [
        [types.ScoredOutput(output='{"extractions":[]}')]
    ]
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_annotate_docs.return_value = []

    docs = [lx.data.Document(text="doc1")]
    examples = [
        lx.data.ExampleData(
            text="Example text",
            extractions=[
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="example",
                ),
            ],
        )
    ]

    lx.extract(
        text_or_documents=docs,
        prompt_description="desc",
        examples=examples,
        api_key="k",
        resolver_params={
            "enable_fuzzy_alignment": False,
            "fuzzy_alignment_threshold": 0.9,
            "accept_match_lesser": False,
        },
    )

    _, kwargs = mock_annotate_docs.call_args
    self.assertFalse(kwargs.get("enable_fuzzy_alignment"))
    self.assertEqual(kwargs.get("fuzzy_alignment_threshold"), 0.9)
    self.assertFalse(kwargs.get("accept_match_lesser"))

  @mock.patch("langextract.annotation.Annotator.annotate_text")
  @mock.patch("langextract.extraction.resolver.Resolver")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_extract_resolver_params_none_threshold(
      self, mock_create_model, mock_resolver_cls, mock_annotate
  ):
    mock_model = mock.MagicMock()
    mock_model.infer.return_value = [
        [types.ScoredOutput(output='{"extractions":[]}')]
    ]
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_resolver_cls.return_value = mock.MagicMock()
    mock_annotate.return_value = lx.data.AnnotatedDocument(
        text="t", extractions=[]
    )

    lx.extract(
        text_or_documents="t",
        prompt_description="d",
        examples=[
            lx.data.ExampleData(
                text="example",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="entity",
                        extraction_text="ex",
                    ),
                ],
            )
        ],
        api_key="k",
        resolver_params={"fuzzy_alignment_threshold": None},
    )

    _, resolver_kwargs = mock_resolver_cls.call_args
    self.assertNotIn("fuzzy_alignment_threshold", resolver_kwargs)

    _, annotate_kwargs = mock_annotate.call_args
    self.assertNotIn("fuzzy_alignment_threshold", annotate_kwargs)

  @mock.patch.object(
      schemas.gemini.GeminiSchema, "from_examples", autospec=True
  )
  @mock.patch("langextract.extraction.factory.create_model")
  def test_extract_custom_params_reach_inference(
      self, mock_create_model, mock_gemini_schema
  ):
    """Sanity check that custom parameters reach the inference layer."""
    input_text = "Test text"

    mock_model = mock.MagicMock()
    mock_model.infer.return_value = [[
        types.ScoredOutput(
            output='```json\n{"extractions": []}\n```',
            score=0.9,
        )
    ]]

    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model
    mock_gemini_schema.return_value = None

    mock_examples = [
        lx.data.ExampleData(
            text="Example",
            extractions=[
                lx.data.Extraction(
                    extraction_class="test",
                    extraction_text="example",
                ),
            ],
        )
    ]

    lx.extract(
        text_or_documents=input_text,
        prompt_description="Test extraction",
        examples=mock_examples,
        api_key="test_key",
        max_workers=5,
        fence_output=True,
        use_schema_constraints=False,
    )

    mock_model.infer.assert_called_once()
    _, kwargs = mock_model.infer.call_args
    self.assertEqual(kwargs.get("max_workers"), 5)

  def test_data_module_exports_via_compatibility_shim(self):
    """Verify data module exports are accessible via lx.data."""
    expected_exports = [
        "AlignmentStatus",
        "CharInterval",
        "Extraction",
        "Document",
        "AnnotatedDocument",
        "ExampleData",
        "FormatType",
    ]

    for name in expected_exports:
      with self.subTest(export=name):
        self.assertTrue(
            hasattr(lx.data, name),
            f"lx.data.{name} not accessible via compatibility shim",
        )

  def test_tokenizer_module_exports_via_compatibility_shim(self):
    """Verify tokenizer module exports are accessible via lx.tokenizer."""
    expected_exports = [
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

    for name in expected_exports:
      with self.subTest(export=name):
        self.assertTrue(
            hasattr(lx.tokenizer, name),
            f"lx.tokenizer.{name} not accessible via compatibility shim",
        )


if __name__ == "__main__":
  absltest.main()
