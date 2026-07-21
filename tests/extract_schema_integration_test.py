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

from langextract import factory
import langextract as lx
from langextract.core import data
from langextract.core import exceptions
from langextract.providers import gemini


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
              model_id="gemini-3.5-flash",
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
                model_id="gemini-3.5-flash",
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
                model_id="gemini-3.5-flash",
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
    # Create a mock instance with required attributes
    mock_model = mock.MagicMock()
    mock_model._schema = None
    mock_model._fence_output_override = None
    mock_model.gemini_schema = None
    mock_model.requires_fence_output = True
    mock_model.infer.return_value = iter(
        [[mock.Mock(output='{"extractions": []}')]]
    )

    # Track the kwargs passed to the constructor
    constructor_kwargs = {}

    def mock_constructor(**kwargs):
      constructor_kwargs.update(kwargs)
      return mock_model

    with mock.patch(
        "langextract.providers.gemini.GeminiLanguageModel",
        side_effect=mock_constructor,
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
            model_id="gemini-3.5-flash",
            api_key="test_key",
            use_schema_constraints=False,  # Disabled
        )

        # Should NOT have response_schema when schema constraints are disabled
        self.assertNotIn("response_schema", constructor_kwargs)
        self.assertNotIn("gemini_schema", constructor_kwargs)

  @mock.patch("langextract.factory.create_model")
  def test_validation_triggers_warning_for_gemini(self, mock_create_model):
    """Test that Gemini schema validation triggers warnings."""

    # Setup mock model with Gemini schema
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_model.infer.return_value = [
        [mock.MagicMock(output='{"extractions": []}', score=1.0)]
    ]

    # Create a mock Gemini schema with validate_format that issues warnings
    mock_schema = mock.MagicMock()

    def mock_validate_format(format_handler, level=None):
      # Simulate the warning that would be issued
      warnings.warn(
          "Gemini outputs native JSON via"
          " response_mime_type='application/json'",
          UserWarning,
          stacklevel=3,
      )

    mock_schema.validate_format = mock_validate_format
    mock_model.schema = mock_schema

    mock_create_model.return_value = mock_model

    # Run extraction with warnings captured
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      result = lx.extract(
          text_or_documents="Sample text",
          prompt_description="Extract entities",
          examples=self.examples,
          model_id="gemini-pro",
          api_key="test_key",
          use_schema_constraints=True,
      )

      # Check that a warning was issued
      warning_messages = [str(warning.message) for warning in w]
      self.assertTrue(
          any("Gemini outputs native JSON" in msg for msg in warning_messages),
          f"Expected Gemini-specific warning not found in: {warning_messages}",
      )

    # Result should still be returned
    self.assertIsNotNone(result)

  @mock.patch("langextract.factory.create_model")
  def test_no_validation_without_schema(self, mock_create_model):
    """Test that validation is skipped when no schema is present."""

    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = False
    mock_model.schema = None  # No schema
    mock_model.infer.return_value = [
        [mock.MagicMock(output='{"extractions": []}', score=1.0)]
    ]

    mock_create_model.return_value = mock_model

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      result = lx.extract(
          text_or_documents="Sample text",
          prompt_description="Extract",
          examples=self.examples,
          model_id="some-model",
          api_key="key",
          use_schema_constraints=False,  # No schema constraints
      )

      # No format compatibility warnings should be issued
      warning_messages = [str(warning.message) for warning in w]
      self.assertFalse(
          any("Format compatibility" in msg for msg in warning_messages),
          f"Unexpected format warning found in: {warning_messages}",
      )

    self.assertIsNotNone(result)


@mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"})
class ExtractOutputSchemaTest(absltest.TestCase):
  """Tests for extract() with user-provided output_schema."""

  def setUp(self):
    super().setUp()
    self.output_schema = lx.schema.extractions_schema(
        lx.schema.extraction_item_schema("condition")
    )
    self.test_text = "Patient has hypertension"

  def _patch_infer(self):
    return mock.patch.object(
        gemini.GeminiLanguageModel,
        "infer",
        autospec=True,
        return_value=iter(
            [[mock.Mock(output='{"extractions": [{"condition": "fever"}]}')]]
        ),
    )

  def test_extract_with_output_schema_allows_no_examples(self):
    with self._patch_infer() as mock_infer:
      result = lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model_id="gemini-3.5-flash",
          output_schema=self.output_schema,
      )

    model = mock_infer.call_args[0][0]
    self.assertEqual(
        model.schema.to_provider_config()["response_json_schema"],
        self.output_schema,
    )
    self.assertLen(result.extractions, 1)
    self.assertEqual(result.extractions[0].extraction_class, "condition")
    self.assertEqual(result.extractions[0].extraction_text, "fever")

  def test_extract_output_schema_overrides_example_schema(self):
    examples = [
        data.ExampleData(
            text="Patient has diabetes",
            extractions=[
                data.Extraction(
                    extraction_class="condition",
                    extraction_text="diabetes",
                )
            ],
        )
    ]

    with self._patch_infer() as mock_infer:
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          examples=examples,
          model_id="gemini-3.5-flash",
          use_schema_constraints=True,
          output_schema=self.output_schema,
      )

    model = mock_infer.call_args[0][0]
    provider_config = model.schema.to_provider_config()
    self.assertEqual(
        provider_config["response_json_schema"], self.output_schema
    )
    self.assertNotIn("response_schema", provider_config)

  def test_extract_requires_examples_without_output_schema(self):
    with self.assertRaisesRegex(ValueError, "output_schema"):
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model_id="gemini-3.5-flash",
      )

  def test_extract_with_preconfigured_output_schema_model(self):
    model = factory.create_model_from_id(
        "gemini-3.5-flash", output_schema=self.output_schema
    )

    with self._patch_infer():
      result = lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model=model,
      )

    self.assertLen(result.extractions, 1)

  def test_extract_applies_output_schema_to_plain_model(self):
    model = factory.create_model_from_id("gemini-3.5-flash")

    with self._patch_infer():
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model=model,
          output_schema=self.output_schema,
      )

    self.assertTrue(model.schema.from_output_schema)

  def test_extract_reapplies_same_output_schema_idempotently(self):
    model = factory.create_model_from_id(
        "gemini-3.5-flash", output_schema=self.output_schema
    )

    with self._patch_infer():
      result = lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model=model,
          output_schema=self.output_schema,
      )

    self.assertIsNotNone(result)

  def test_extract_output_schema_conflicts_with_example_schema_model(self):
    examples = [
        data.ExampleData(
            text="Patient has diabetes",
            extractions=[
                data.Extraction(
                    extraction_class="condition",
                    extraction_text="diabetes",
                )
            ],
        )
    ]
    model = factory.create_model(
        factory.ModelConfig(model_id="gemini-3.5-flash"),
        examples=examples,
        use_schema_constraints=True,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "already has a schema"
    ):
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model=model,
          output_schema=self.output_schema,
      )

  def test_extract_output_schema_rejects_fence_output(self):
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "fence_output"
    ):
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model_id="gemini-3.5-flash",
          output_schema=self.output_schema,
          fence_output=True,
      )

  def test_extract_output_schema_rejects_yaml_format(self):
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "format_type=JSON"
    ):
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model_id="gemini-3.5-flash",
          output_schema=self.output_schema,
          format_type=data.FormatType.YAML,
      )

  def test_extract_output_schema_rejects_fenced_resolver_params(self):
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "fence_output"
    ):
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model_id="gemini-3.5-flash",
          output_schema=self.output_schema,
          resolver_params={"fence_output": True},
      )

  def test_extract_output_schema_rejects_unwrapped_resolver_output(self):
    with self.assertRaisesRegex(exceptions.InferenceConfigError, "envelope"):
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model_id="gemini-3.5-flash",
          output_schema=self.output_schema,
          resolver_params={"require_extractions_key": False},
      )

  def test_extract_output_schema_rejects_custom_attribute_suffix(self):
    with self.assertRaisesRegex(exceptions.InferenceConfigError, "envelope"):
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model_id="gemini-3.5-flash",
          output_schema=self.output_schema,
          resolver_params={"attribute_suffix": "_props"},
      )

  def test_extract_fence_conflict_leaves_caller_model_unmodified(self):
    model = factory.create_model_from_id("gemini-3.5-flash")

    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "fence_output"
    ):
      lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model=model,
          output_schema=self.output_schema,
          fence_output=True,
      )

    self.assertIsNone(model.schema)
    with self._patch_infer():
      result = lx.extract(
          text_or_documents=self.test_text,
          prompt_description="Extract conditions",
          model=model,
          output_schema=self.output_schema,
      )
    self.assertIsNotNone(result)


if __name__ == "__main__":
  absltest.main()
