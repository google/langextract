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

"""Tests for the schema module.

Note: This file contains test helper classes that intentionally have
few public methods. The too-few-public-methods warnings are expected.
"""

from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized

from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import format_handler as fh
from langextract.core import schema
from langextract.providers import schemas


class BaseSchemaTest(absltest.TestCase):
  """Tests for BaseSchema abstract class."""

  def test_abstract_methods_required(self):
    """Test that BaseSchema cannot be instantiated directly."""
    with self.assertRaises(TypeError):
      schema.BaseSchema()  # pylint: disable=abstract-class-instantiated

  def test_subclass_must_implement_all_methods(self):
    """Test that subclasses must implement all abstract methods."""

    class IncompleteSchema(schema.BaseSchema):  # pylint: disable=too-few-public-methods

      @classmethod
      def from_examples(cls, examples_data, attribute_suffix="_attributes"):
        return cls()

    with self.assertRaises(TypeError):
      IncompleteSchema()  # pylint: disable=abstract-class-instantiated


class BaseLanguageModelSchemaTest(absltest.TestCase):
  """Tests for BaseLanguageModel schema methods."""

  def test_get_schema_class_returns_none_by_default(self):
    """Test that get_schema_class returns None by default."""

    class TestModel(base_model.BaseLanguageModel):  # pylint: disable=too-few-public-methods

      def infer(self, batch_prompts, **kwargs):
        yield []

    self.assertIsNone(TestModel.get_schema_class())

  def test_apply_schema_stores_instance(self):
    """Test that apply_schema stores the schema instance."""

    class TestModel(base_model.BaseLanguageModel):  # pylint: disable=too-few-public-methods

      def infer(self, batch_prompts, **kwargs):
        yield []

    model = TestModel()

    mock_schema = mock.Mock(spec=schema.BaseSchema)

    model.apply_schema(mock_schema)

    self.assertEqual(model._schema, mock_schema)

    model.apply_schema(None)
    self.assertIsNone(model._schema)


class GeminiSchemaTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_extractions",
          examples_data=[],
          expected_schema={
              "type": "object",
              "properties": {
                  data.EXTRACTIONS_KEY: {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {},
                      },
                  },
              },
              "required": [data.EXTRACTIONS_KEY],
          },
      ),
      dict(
          testcase_name="single_extraction_no_attributes",
          examples_data=[
              data.ExampleData(
                  text="Patient has diabetes.",
                  extractions=[
                      data.Extraction(
                          extraction_text="diabetes",
                          extraction_class="condition",
                      )
                  ],
              )
          ],
          expected_schema={
              "type": "object",
              "properties": {
                  data.EXTRACTIONS_KEY: {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {
                              "condition": {"type": "string"},
                              "condition_attributes": {
                                  "type": "object",
                                  "properties": {
                                      "_unused": {"type": "string"},
                                  },
                                  "nullable": True,
                              },
                          },
                      },
                  },
              },
              "required": [data.EXTRACTIONS_KEY],
          },
      ),
      dict(
          testcase_name="single_extraction",
          examples_data=[
              data.ExampleData(
                  text="Patient has diabetes.",
                  extractions=[
                      data.Extraction(
                          extraction_text="diabetes",
                          extraction_class="condition",
                          attributes={"chronicity": "chronic"},
                      )
                  ],
              )
          ],
          expected_schema={
              "type": "object",
              "properties": {
                  data.EXTRACTIONS_KEY: {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {
                              "condition": {"type": "string"},
                              "condition_attributes": {
                                  "type": "object",
                                  "properties": {
                                      "chronicity": {"type": "string"},
                                  },
                                  "nullable": True,
                              },
                          },
                      },
                  },
              },
              "required": [data.EXTRACTIONS_KEY],
          },
      ),
      dict(
          testcase_name="multiple_extraction_classes",
          examples_data=[
              data.ExampleData(
                  text="Patient has diabetes.",
                  extractions=[
                      data.Extraction(
                          extraction_text="diabetes",
                          extraction_class="condition",
                          attributes={"chronicity": "chronic"},
                      )
                  ],
              ),
              data.ExampleData(
                  text="Patient is John Doe",
                  extractions=[
                      data.Extraction(
                          extraction_text="John Doe",
                          extraction_class="patient",
                          attributes={"id": "12345"},
                      )
                  ],
              ),
          ],
          expected_schema={
              "type": "object",
              "properties": {
                  data.EXTRACTIONS_KEY: {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {
                              "condition": {"type": "string"},
                              "condition_attributes": {
                                  "type": "object",
                                  "properties": {
                                      "chronicity": {"type": "string"}
                                  },
                                  "nullable": True,
                              },
                              "patient": {"type": "string"},
                              "patient_attributes": {
                                  "type": "object",
                                  "properties": {
                                      "id": {"type": "string"},
                                  },
                                  "nullable": True,
                              },
                          },
                      },
                  },
              },
              "required": [data.EXTRACTIONS_KEY],
          },
      ),
  )
  def test_from_examples_constructs_expected_schema(
      self, examples_data, expected_schema
  ):
    gemini_schema = schemas.gemini.GeminiSchema.from_examples(examples_data)
    actual_schema = gemini_schema.schema_dict
    self.assertEqual(actual_schema, expected_schema)

  def test_to_provider_config_returns_response_schema(self):
    """Test that to_provider_config returns the correct provider kwargs."""
    examples_data = [
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

    gemini_schema = schemas.gemini.GeminiSchema.from_examples(examples_data)
    provider_config = gemini_schema.to_provider_config()

    self.assertIn("response_schema", provider_config)
    self.assertEqual(
        provider_config["response_schema"], gemini_schema.schema_dict
    )

  def test_requires_raw_output_returns_true(self):
    """Test that GeminiSchema requires raw output."""
    examples_data = [
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

    gemini_schema = schemas.gemini.GeminiSchema.from_examples(examples_data)
    self.assertTrue(gemini_schema.requires_raw_output)


class OpenAISchemaTest(parameterized.TestCase):
  """Tests for OpenAI structured output schema generation."""

  def test_response_format_returns_json_schema_response_format(self):
    """OpenAI schema exposes Chat Completions structured outputs."""
    examples_data = [
        data.ExampleData(
            text="Patient has diabetes.",
            extractions=[
                data.Extraction(
                    extraction_text="diabetes",
                    extraction_class="condition",
                    attributes={"chronicity": "chronic"},
                )
            ],
        )
    ]

    openai_schema = schemas.openai.OpenAISchema.from_examples(examples_data)

    response_format = openai_schema.response_format
    self.assertEqual(response_format["type"], "json_schema")
    self.assertEqual(
        response_format["json_schema"]["name"], "langextract_extractions"
    )
    self.assertTrue(response_format["json_schema"]["strict"])
    self.assertIs(
        response_format["json_schema"]["schema"], openai_schema.schema_dict
    )

  def test_to_provider_config_returns_schema_instance(self):
    """OpenAI provider config passes schema state to the provider."""
    openai_schema = schemas.openai.OpenAISchema.from_examples([])

    provider_config = openai_schema.to_provider_config()

    self.assertIs(provider_config["openai_schema"], openai_schema)

  def test_from_examples_constructs_strict_openai_schema(self):
    """OpenAI schema uses strict-compatible extraction variants."""
    examples_data = [
        data.ExampleData(
            text="Patient has diabetes.",
            extractions=[
                data.Extraction(
                    extraction_text="diabetes",
                    extraction_class="condition",
                    attributes={"chronicity": "chronic"},
                ),
                data.Extraction(
                    extraction_text="metformin",
                    extraction_class="medication",
                    attributes={"route": "oral"},
                ),
            ],
        )
    ]

    openai_schema = schemas.openai.OpenAISchema.from_examples(examples_data)

    self.assertEqual(openai_schema.schema_dict["type"], "object")
    self.assertIs(openai_schema.schema_dict["additionalProperties"], False)
    self.assertEqual(
        openai_schema.schema_dict["required"], [data.EXTRACTIONS_KEY]
    )
    extraction_items = openai_schema.schema_dict["properties"][
        data.EXTRACTIONS_KEY
    ]["items"]
    self.assertIn("anyOf", extraction_items)
    self.assertLen(extraction_items["anyOf"], 2)
    variant_properties = [
        set(variant["properties"]) for variant in extraction_items["anyOf"]
    ]
    self.assertIn({"condition", "condition_attributes"}, variant_properties)
    self.assertIn({"medication", "medication_attributes"}, variant_properties)
    for variant in extraction_items["anyOf"]:
      self.assertIs(variant["additionalProperties"], False)
      self.assertCountEqual(variant["properties"].keys(), variant["required"])
      for property_schema in variant["properties"].values():
        if property_schema.get("anyOf"):
          attribute_schema = property_schema["anyOf"][0]
          self.assertFalse(attribute_schema["additionalProperties"])
          self.assertCountEqual(
              attribute_schema["properties"].keys(),
              attribute_schema["required"],
          )

  def test_from_examples_preserves_list_attribute_schema(self):
    """OpenAI schema accepts list attributes from examples."""
    examples_data = [
        data.ExampleData(
            text="Patient has diabetes with fatigue.",
            extractions=[
                data.Extraction(
                    extraction_text="diabetes",
                    extraction_class="condition",
                    attributes={"symptoms": ["fatigue"]},
                )
            ],
        )
    ]

    openai_schema = schemas.openai.OpenAISchema.from_examples(examples_data)

    variant = openai_schema.schema_dict["properties"][data.EXTRACTIONS_KEY][
        "items"
    ]["anyOf"][0]
    attributes_schema = variant["properties"]["condition_attributes"]["anyOf"][
        0
    ]
    symptoms_schema = attributes_schema["properties"]["symptoms"]
    self.assertIn(
        {"type": "array", "items": {"type": "string"}},
        symptoms_schema["anyOf"],
    )
    self.assertIn({"type": "null"}, symptoms_schema["anyOf"])

  def test_from_examples_empty_examples_allow_empty_extraction_objects(self):
    """OpenAI schema handles empty example sets deterministically."""
    openai_schema = schemas.openai.OpenAISchema.from_examples([])

    extraction_items = openai_schema.schema_dict["properties"][
        data.EXTRACTIONS_KEY
    ]["items"]
    self.assertEqual(
        extraction_items,
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

  def test_validate_format_rejects_yaml(self):
    """OpenAI structured outputs are JSON-only."""
    openai_schema = schemas.openai.OpenAISchema.from_examples([])
    format_handler = fh.FormatHandler(format_type=data.FormatType.YAML)

    with self.assertRaisesRegex(
        exceptions.InferenceConfigError,
        "OpenAI structured output only supports JSON format",
    ):
      openai_schema.validate_format(format_handler)

  def test_requires_raw_output_returns_true(self):
    """OpenAI structured outputs emit raw JSON without fences."""
    openai_schema = schemas.openai.OpenAISchema.from_examples([])

    self.assertTrue(openai_schema.requires_raw_output)

  def test_validate_format_warns_when_fences_enabled(self):
    """OpenAI schema warns when raw JSON would be wrapped in fences."""
    openai_schema = schemas.openai.OpenAISchema.from_examples([])
    format_handler = fh.FormatHandler(
        format_type=data.FormatType.JSON,
        use_fences=True,
    )

    with self.assertWarnsRegex(
        UserWarning, "OpenAI structured outputs emit native JSON"
    ):
      openai_schema.validate_format(format_handler)

  def test_validate_format_warns_with_wrong_wrapper_key(self):
    """OpenAI schema warns when resolver wrapper settings drift."""
    openai_schema = schemas.openai.OpenAISchema.from_examples([])
    format_handler = fh.FormatHandler(
        format_type=data.FormatType.JSON,
        use_fences=False,
        wrapper_key="items",
    )

    with self.assertWarnsRegex(
        UserWarning,
        f"response_format schema expects wrapper_key='{data.EXTRACTIONS_KEY}'",
    ):
      openai_schema.validate_format(format_handler)


class SchemaValidationTest(parameterized.TestCase):
  """Tests for schema format validation."""

  def _create_test_schema(self):
    """Helper to create a test schema."""
    examples = [
        data.ExampleData(
            text="Test",
            extractions=[
                data.Extraction(
                    extraction_class="entity",
                    extraction_text="test",
                )
            ],
        )
    ]
    return schemas.gemini.GeminiSchema.from_examples(examples)

  @parameterized.named_parameters(
      dict(
          testcase_name="warns_about_fences",
          use_fences=True,
          use_wrapper=True,
          wrapper_key=data.EXTRACTIONS_KEY,
          expected_warning="fence_output=True may cause parsing issues",
      ),
      dict(
          testcase_name="warns_about_wrong_wrapper_key",
          use_fences=False,
          use_wrapper=True,
          wrapper_key="wrong_key",
          expected_warning="response_schema expects wrapper_key='extractions'",
      ),
      dict(
          testcase_name="no_warning_with_correct_settings",
          use_fences=False,
          use_wrapper=True,
          wrapper_key=data.EXTRACTIONS_KEY,
          expected_warning=None,
      ),
  )
  def test_gemini_validation(
      self, use_fences, use_wrapper, wrapper_key, expected_warning
  ):
    """Test GeminiSchema validation with various settings."""
    schema_obj = self._create_test_schema()
    format_handler = fh.FormatHandler(
        format_type=data.FormatType.JSON,
        use_fences=use_fences,
        use_wrapper=use_wrapper,
        wrapper_key=wrapper_key,
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      schema_obj.validate_format(format_handler)

      if expected_warning:
        self.assertLen(
            w,
            1,
            f"Expected exactly one warning containing '{expected_warning}'",
        )
        self.assertIn(
            expected_warning,
            str(w[0].message),
            f"Warning message should contain '{expected_warning}'",
        )
      else:
        self.assertEmpty(w, "No warnings should be issued for correct settings")

  def test_base_schema_no_validation(self):
    """Test that base schema has no validation by default."""
    schema_obj = schema.FormatModeSchema()
    format_handler = fh.FormatHandler(
        format_type=data.FormatType.JSON,
        use_fences=True,
    )

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      schema_obj.validate_format(format_handler)

      self.assertEmpty(
          w, "FormatModeSchema should not issue validation warnings"
      )


if __name__ == "__main__":
  absltest.main()
