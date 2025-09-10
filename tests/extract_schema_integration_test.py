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

"""Integration tests for extract function with new schema system."""

from unittest import mock
import warnings

from absl.testing import absltest

import langextract as lx
from langextract.core import data


class ExtractSchemaIntegrationTest(absltest.TestCase):
  """Tests for extract function with schema system integration."""

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    self.examples = [
        data.ExampleData(
            text="Patient has diabetes",
            extractions=[
                data.Extraction(
                    extraction_class="condition",
                    extraction_text="diabetes",
                    attributes={"severity": "moderate"},
                )
            ],
        )
    ]
    self.test_text = "Patient has hypertension"

  @mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"})
  def test_extract_with_gemini_uses_schema(self):
    """Test that extract with Gemini automatically uses schema."""
    with mock.patch(
        "langextract.providers.gemini.GeminiLanguageModel.__init__",
        return_value=None,
    ) as mock_init:
      with mock.patch(
          "langextract.providers.gemini.GeminiLanguageModel.infer",
          return_value=iter([[mock.Mock(output='{"extractions": []}')]]),
      ):
        with mock.patch(
            "langextract.annotation.Annotator.annotate_text",
            return_value=data.AnnotatedDocument(
                text=self.test_text, extractions=[]
            ),
        ):
          result = lx.extract(
              text_or_documents=self.test_text,
              prompt_description="Extract conditions",
              examples=self.examples,
              model_id="gemini-2.5-flash",
              use_schema_constraints=True,
              fence_output=None,  # Let it compute
          )

          # Should have been called with response_schema
          call_kwargs = mock_init.call_args[1]
          self.assertIn("response_schema", call_kwargs)

          # Result should be an AnnotatedDocument
          self.assertIsInstance(result, data.AnnotatedDocument)

  @mock.patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://localhost:11434"})
  def test_extract_with_ollama_uses_json_mode(self):
    """Test that extract with Ollama uses JSON mode."""
    with mock.patch(
        "langextract.providers.ollama.OllamaLanguageModel.__init__",
        return_value=None,
    ) as mock_init:
      with mock.patch(
          "langextract.providers.ollama.OllamaLanguageModel.infer",
          return_value=iter([[mock.Mock(output='{"extractions": []}')]]),
      ):
        with mock.patch(
            "langextract.annotation.Annotator.annotate_text",
            return_value=data.AnnotatedDocument(
                text=self.test_text, extractions=[]
            ),
        ):
          result = lx.extract(
              text_or_documents=self.test_text,
              prompt_description="Extract conditions",
              examples=self.examples,
              model_id="gemma2:2b",
              use_schema_constraints=True,
              fence_output=None,  # Let it compute
          )

          # Should have been called with format="json"
          call_kwargs = mock_init.call_args[1]
          self.assertIn("format", call_kwargs)
          self.assertEqual(call_kwargs["format"], "json")

          # Result should be an AnnotatedDocument
          self.assertIsInstance(result, data.AnnotatedDocument)

  def test_extract_explicit_fence_respected(self):
    """Test that explicit fence_output is respected in extract."""
    with mock.patch(
        "langextract.providers.gemini.GeminiLanguageModel.__init__",
        return_value=None,
    ):
      with mock.patch(
          "langextract.providers.gemini.GeminiLanguageModel.infer",
          return_value=iter([[mock.Mock(output='{"extractions": []}')]]),
      ):
        with mock.patch(
            "langextract.annotation.Annotator.__init__", return_value=None
        ) as mock_annotator_init:
          with mock.patch(
              "langextract.annotation.Annotator.annotate_text",
              return_value=data.AnnotatedDocument(
                  text=self.test_text, extractions=[]
              ),
          ):
            _ = lx.extract(
                text_or_documents=self.test_text,
                prompt_description="Extract conditions",
                examples=self.examples,
                model_id="gemini-2.5-flash",
                api_key="test_key",
                use_schema_constraints=True,
                fence_output=True,  # Explicitly set
            )

            # Annotator should be created with format_handler that has use_fences=True
            call_kwargs = mock_annotator_init.call_args[1]
            self.assertIn("format_handler", call_kwargs)
            self.assertTrue(call_kwargs["format_handler"].use_fences)

  def test_extract_gemini_schema_deprecation_warning(self):
    """Test that passing gemini_schema triggers deprecation warning."""
    with mock.patch(
        "langextract.providers.gemini.GeminiLanguageModel.__init__",
        return_value=None,
    ):
      with mock.patch(
          "langextract.providers.gemini.GeminiLanguageModel.infer",
          return_value=iter([[mock.Mock(output='{"extractions": []}')]]),
      ):
        with mock.patch(
            "langextract.annotation.Annotator.annotate_text",
            return_value=data.AnnotatedDocument(
                text=self.test_text, extractions=[]
            ),
        ):
          with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = lx.extract(
                text_or_documents=self.test_text,
                prompt_description="Extract conditions",
                examples=self.examples,
                model_id="gemini-2.5-flash",
                api_key="test_key",
                language_model_params={
                    "gemini_schema": "some_schema"
                },  # Deprecated
            )

            # Should have triggered deprecation warning
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
                and "gemini_schema" in str(warning.message)
            ]
            self.assertGreater(len(deprecation_warnings), 0)

  def test_extract_no_schema_when_disabled(self):
    """Test that no schema is used when use_schema_constraints=False."""
    with mock.patch(
        "langextract.providers.gemini.GeminiLanguageModel.__init__",
        return_value=None,
    ) as mock_init:
      with mock.patch(
          "langextract.providers.gemini.GeminiLanguageModel.infer",
          return_value=iter([[mock.Mock(output='{"extractions": []}')]]),
      ):
        with mock.patch(
            "langextract.annotation.Annotator.annotate_text",
            return_value=data.AnnotatedDocument(
                text=self.test_text, extractions=[]
            ),
        ):
          _ = lx.extract(
              text_or_documents=self.test_text,
              prompt_description="Extract conditions",
              examples=self.examples,
              model_id="gemini-2.5-flash",
              api_key="test_key",
              use_schema_constraints=False,  # Disabled
          )

          # Should NOT have response_schema
          call_kwargs = mock_init.call_args[1]
          self.assertNotIn("response_schema", call_kwargs)


if __name__ == "__main__":
  absltest.main()
