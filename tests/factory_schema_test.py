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

"""Tests for factory schema integration and fence defaulting."""

from unittest import mock

from absl.testing import absltest

from langextract import factory
from langextract import schema
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.providers import schemas


class FactorySchemaIntegrationTest(absltest.TestCase):
  """Tests for create_model_with_schema factory function."""

  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    self.examples = [
        data.ExampleData(
            text="Test text",
            extractions=[
                data.Extraction(
                    extraction_class="test_class",
                    extraction_text="test extraction",
                )
            ],
        )
    ]

  @mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"})
  def test_gemini_with_schema_returns_false_fence(self):
    """Test that Gemini with schema returns fence_output=False."""
    config = factory.ModelConfig(
        model_id="gemini-3.5-flash", provider_kwargs={"api_key": "test_key"}
    )

    with mock.patch(
        "langextract.providers.gemini.GeminiLanguageModel.__init__",
        return_value=None,
    ) as mock_init:
      model = factory._create_model_with_schema(
          config=config,
          examples=self.examples,
          use_schema_constraints=True,
          fence_output=None,
      )

      mock_init.assert_called_once()
      call_kwargs = mock_init.call_args[1]
      self.assertIn("response_schema", call_kwargs)

      self.assertFalse(model.requires_fence_output)

  @mock.patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://localhost:11434"})
  def test_ollama_with_schema_returns_false_fence(self):
    """Test that Ollama with JSON mode returns fence_output=False."""
    config = factory.ModelConfig(model_id="gemma2:2b")

    with mock.patch(
        "langextract.providers.ollama.OllamaLanguageModel.__init__",
        return_value=None,
    ) as mock_init:
      model = factory._create_model_with_schema(
          config=config,
          examples=self.examples,
          use_schema_constraints=True,
          fence_output=None,
      )

      mock_init.assert_called_once()
      call_kwargs = mock_init.call_args[1]
      self.assertIn("format", call_kwargs)
      self.assertEqual(call_kwargs["format"], "json")

      self.assertFalse(model.requires_fence_output)

  def test_openai_with_schema_returns_false_fence(self):
    """OpenAI schema constraints use raw JSON by default."""
    config = factory.ModelConfig(
        model_id="gpt-4o-mini", provider_kwargs={"api_key": "test_key"}
    )

    with mock.patch("openai.OpenAI", autospec=True):
      model = factory._create_model_with_schema(
          config=config,
          examples=self.examples,
          use_schema_constraints=True,
          fence_output=None,
      )

      self.assertIsInstance(model.schema, schemas.openai.OpenAISchema)
      self.assertIs(model.requires_fence_output, False)

  def test_openai_explicit_fence_output_respected(self):
    """Explicit OpenAI fence_output overrides schema defaults."""
    config = factory.ModelConfig(
        model_id="gpt-4o-mini", provider_kwargs={"api_key": "test_key"}
    )

    with mock.patch("openai.OpenAI", autospec=True):
      model = factory._create_model_with_schema(
          config=config,
          examples=self.examples,
          use_schema_constraints=True,
          fence_output=True,
      )

      self.assertIs(model.requires_fence_output, True)

  def test_explicit_fence_output_respected(self):
    """Test that explicit fence_output is not overridden."""
    config = factory.ModelConfig(
        model_id="gemini-3.5-flash", provider_kwargs={"api_key": "test_key"}
    )

    with mock.patch(
        "langextract.providers.gemini.GeminiLanguageModel.__init__",
        return_value=None,
    ):
      model = factory._create_model_with_schema(
          config=config,
          examples=self.examples,
          use_schema_constraints=True,
          fence_output=True,
      )

      self.assertTrue(model.requires_fence_output)

  def test_no_schema_defaults_to_true_fence(self):
    """Test that models without schema support default to fence_output=True."""

    class NoSchemaModel(base_model.BaseLanguageModel):  # pylint: disable=too-few-public-methods

      def infer(self, batch_prompts, **kwargs):
        yield []

    config = factory.ModelConfig(model_id="test-model")

    with mock.patch(
        "langextract.providers.registry.resolve", return_value=NoSchemaModel
    ):
      with mock.patch.object(NoSchemaModel, "__init__", return_value=None):
        model = factory._create_model_with_schema(
            config=config,
            examples=self.examples,
            use_schema_constraints=True,
            fence_output=None,
        )

        self.assertTrue(model.requires_fence_output)

  def test_schema_disabled_returns_true_fence(self):
    """Test that disabling schema constraints returns fence_output=True."""
    config = factory.ModelConfig(
        model_id="gemini-3.5-flash", provider_kwargs={"api_key": "test_key"}
    )

    with mock.patch(
        "langextract.providers.gemini.GeminiLanguageModel.__init__",
        return_value=None,
    ) as mock_init:
      model = factory._create_model_with_schema(
          config=config,
          examples=self.examples,
          use_schema_constraints=False,
          fence_output=None,
      )

      call_kwargs = mock_init.call_args[1]
      self.assertNotIn("response_schema", call_kwargs)

      self.assertTrue(model.requires_fence_output)

  def test_caller_overrides_schema_config(self):
    """Test that caller's provider_kwargs override schema configuration."""
    config = factory.ModelConfig(
        model_id="gemma2:2b",
        provider_kwargs={"format": "yaml"},
    )

    with mock.patch(
        "langextract.providers.ollama.OllamaLanguageModel.__init__",
        return_value=None,
    ) as mock_init:
      _ = factory._create_model_with_schema(
          config=config,
          examples=self.examples,
          use_schema_constraints=True,
          fence_output=None,
      )

      mock_init.assert_called_once()
      call_kwargs = mock_init.call_args[1]
      self.assertIn("format", call_kwargs)
      self.assertEqual(call_kwargs["format"], "yaml")

  def test_no_examples_no_schema(self):
    """Test that no examples means no schema is created."""
    config = factory.ModelConfig(
        model_id="gemini-3.5-flash", provider_kwargs={"api_key": "test_key"}
    )

    with mock.patch(
        "langextract.providers.gemini.GeminiLanguageModel.__init__",
        return_value=None,
    ) as mock_init:
      model = factory._create_model_with_schema(
          config=config,
          examples=None,
          use_schema_constraints=True,
          fence_output=None,
      )

      call_kwargs = mock_init.call_args[1]
      self.assertNotIn("response_schema", call_kwargs)

      self.assertTrue(model.requires_fence_output)


class SchemaApplicationTest(absltest.TestCase):
  """Tests for apply_schema being called on models."""

  def test_apply_schema_called_when_supported(self):
    """Test that apply_schema is called on models that support it."""
    examples = [
        data.ExampleData(
            text="Test",
            extractions=[
                data.Extraction(extraction_class="test", extraction_text="test")
            ],
        )
    ]

    class SchemaAwareModel(base_model.BaseLanguageModel):

      @classmethod
      def get_schema_class(cls):
        return schema.GeminiSchema

      def infer(self, batch_prompts, **kwargs):
        yield []

    config = factory.ModelConfig(model_id="test-model")

    with mock.patch(
        "langextract.providers.registry.resolve", return_value=SchemaAwareModel
    ):
      with mock.patch.object(SchemaAwareModel, "__init__", return_value=None):
        with mock.patch.object(SchemaAwareModel, "apply_schema") as mock_apply:
          _ = factory._create_model_with_schema(
              config=config,
              examples=examples,
              use_schema_constraints=True,
          )

          mock_apply.assert_called_once()
          schema_arg = mock_apply.call_args[0][0]
          self.assertIsInstance(schema_arg, schema.GeminiSchema)


@mock.patch.dict(
    "os.environ",
    {"GEMINI_API_KEY": "test_key", "OPENAI_API_KEY": "test_key"},
)
class FactoryOutputSchemaTest(absltest.TestCase):
  """Tests for create_model with user-provided output_schema."""

  def setUp(self):
    super().setUp()
    self.output_schema = schema.extractions_schema(
        schema.extraction_item_schema("condition")
    )
    self.examples = [
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

  def test_gemini_output_schema_configures_json_schema(self):
    model = factory.create_model_from_id(
        "gemini-3.5-flash", output_schema=self.output_schema
    )

    self.assertIsInstance(model.schema, schemas.gemini.GeminiSchema)
    self.assertTrue(model.schema.from_output_schema)
    self.assertEqual(
        model.schema.to_provider_config()["response_json_schema"],
        self.output_schema,
    )
    self.assertFalse(model.requires_fence_output)

  def test_openai_output_schema_configures_response_format(self):
    model = factory.create_model_from_id(
        "gpt-4o", output_schema=self.output_schema
    )

    self.assertIsInstance(model.schema, schemas.openai.OpenAISchema)
    self.assertTrue(model.schema.from_output_schema)
    self.assertEqual(
        model.schema.response_format["json_schema"]["schema"],
        self.output_schema,
    )
    self.assertFalse(model.requires_fence_output)

  def test_output_schema_overrides_example_schema(self):
    model = factory.create_model(
        factory.ModelConfig(model_id="gemini-3.5-flash"),
        examples=self.examples,
        use_schema_constraints=True,
        output_schema=self.output_schema,
    )

    provider_config = model.schema.to_provider_config()
    self.assertEqual(
        provider_config["response_json_schema"], self.output_schema
    )
    self.assertNotIn("response_schema", provider_config)

  def test_output_schema_rejects_provider_schema_kwargs(self):
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "response_schema"
    ):
      factory.create_model_from_id(
          "gemini-3.5-flash",
          output_schema=self.output_schema,
          response_schema={"type": "object"},
      )

  def test_output_schema_ignores_none_provider_schema_kwargs(self):
    model = factory.create_model_from_id(
        "gemini-3.5-flash",
        output_schema=self.output_schema,
        response_schema=None,
    )

    self.assertEqual(
        model.schema.to_provider_config()["response_mime_type"],
        "application/json",
    )

  def test_output_schema_rejects_fence_output(self):
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "fence_output"
    ):
      factory.create_model(
          factory.ModelConfig(model_id="gemini-3.5-flash"),
          output_schema=self.output_schema,
          fence_output=True,
      )

  def test_output_schema_rejects_yaml_format(self):
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "format_type=JSON"
    ):
      factory.create_model_from_id(
          "gemini-3.5-flash",
          output_schema=self.output_schema,
          format_type=data.FormatType.YAML,
      )

  def test_output_schema_rejects_unsupported_provider(self):
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "does not support output_schema"
    ):
      factory.create_model_from_id(
          "gemma2:2b", output_schema=self.output_schema
      )


if __name__ == "__main__":
  absltest.main()
