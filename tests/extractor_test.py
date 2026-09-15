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

"""Tests for reusable Extractor."""

from unittest import mock

from absl.testing import absltest

import langextract as lx
from langextract.core import data


class ExtractorReuseTest(absltest.TestCase):
  """Extractor should build the pipeline once and reuse it."""

  def setUp(self):
    super().setUp()
    self.examples = [
        data.ExampleData(
            text="Anne Teak is 45.",
            extractions=[
                data.Extraction(
                    extraction_class="person",
                    extraction_text="Anne Teak",
                )
            ],
        )
    ]
    self.description = "Extract people mentioned in the text."

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_reuses_annotator_across_calls(
      self, mock_create_model, mock_annotator_cls
  ):
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_model.schema = None
    mock_create_model.return_value = mock_model

    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_text.side_effect = ["first", "second"]

    extractor = lx.Extractor(
        prompt_description=self.description,
        examples=self.examples,
        model_id="gemini-3.5-flash",
        api_key="test-key",
        use_schema_constraints=False,
    )

    self.assertEqual(extractor.extract("Anne Teak is 45."), "first")
    self.assertEqual(extractor.extract("Scott Chegg is 61."), "second")

    mock_create_model.assert_called_once()
    mock_annotator_cls.assert_called_once()
    self.assertEqual(mock_annotator.annotate_text.call_count, 2)

    first_text = mock_annotator.annotate_text.call_args_list[0].kwargs["text"]
    second_text = mock_annotator.annotate_text.call_args_list[1].kwargs["text"]
    self.assertEqual(first_text, "Anne Teak is 45.")
    self.assertEqual(second_text, "Scott Chegg is 61.")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_per_call_additional_context(
      self, mock_create_model, mock_annotator_cls
  ):
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_model.schema = None
    mock_create_model.return_value = mock_model
    mock_annotator_cls.return_value.annotate_text.return_value = "ok"

    extractor = lx.Extractor(
        prompt_description=self.description,
        examples=self.examples,
        model_id="gemini-3.5-flash",
        api_key="test-key",
        use_schema_constraints=False,
    )
    extractor.extract("text", additional_context="prior turn")

    _, kwargs = mock_annotator_cls.return_value.annotate_text.call_args
    self.assertEqual(kwargs["additional_context"], "prior turn")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_extract_function_still_works(
      self, mock_create_model, mock_annotator_cls
  ):
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_model.schema = None
    mock_create_model.return_value = mock_model
    mock_annotator_cls.return_value.annotate_text.return_value = "ok"

    result = lx.extract(
        text_or_documents="Anne Teak is 45.",
        prompt_description=self.description,
        examples=self.examples,
        model_id="gemini-3.5-flash",
        api_key="test-key",
        use_schema_constraints=False,
    )

    self.assertEqual(result, "ok")
    mock_create_model.assert_called_once()

  def test_requires_examples_or_schema(self):
    with self.assertRaisesRegex(ValueError, "Examples are required"):
      lx.Extractor(prompt_description=self.description)


if __name__ == "__main__":
  absltest.main()
