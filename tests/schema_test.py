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

import string
import textwrap
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract import data
from langextract import schema


class GeminiSchemaTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_extractions",
          examples_data=[],
          expected_schema={
              "type": "object",
              "properties": {
                  schema.EXTRACTIONS_KEY: {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {},
                      },
                  },
              },
              "required": [schema.EXTRACTIONS_KEY],
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
                  schema.EXTRACTIONS_KEY: {
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
              "required": [schema.EXTRACTIONS_KEY],
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
                  schema.EXTRACTIONS_KEY: {
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
              "required": [schema.EXTRACTIONS_KEY],
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
                  schema.EXTRACTIONS_KEY: {
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
              "required": [schema.EXTRACTIONS_KEY],
          },
      ),
  )
  def test_from_examples_constructs_expected_schema(
      self, examples_data, expected_schema
  ):
    gemini_schema = schema.GeminiSchema.from_examples(examples_data)
    actual_schema = gemini_schema.schema_dict
    self.assertEqual(actual_schema, expected_schema)


class OpenAISchemaTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_extractions",
          examples_data=[],
          expected_schema={
              "type": "object",
              "properties": {
                  schema.EXTRACTIONS_KEY: {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {},
                          "required": [],
                          "additionalProperties": False,
                      },
                  },
              },
              "required": [schema.EXTRACTIONS_KEY],
              "additionalProperties": False,
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
                  schema.EXTRACTIONS_KEY: {
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
                                  "additionalProperties": False,
                              },
                          },
                          "required": ["condition", "condition_attributes"],
                          "additionalProperties": False,
                      },
                  },
              },
              "required": [schema.EXTRACTIONS_KEY],
              "additionalProperties": False,
          },
      ),
      dict(
          testcase_name="single_extraction_with_attributes",
          examples_data=[
              data.ExampleData(
                  text="Patient has diabetes.",
                  extractions=[
                      data.Extraction(
                          extraction_text="diabetes",
                          extraction_class="condition",
                          attributes={
                              "severity": "mild",
                              "duration": "5 years",
                          },
                      )
                  ],
              )
          ],
          expected_schema={
              "type": "object",
              "properties": {
                  schema.EXTRACTIONS_KEY: {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {
                              "condition": {"type": "string"},
                              "condition_attributes": {
                                  "type": "object",
                                  "properties": {
                                      "severity": {"type": "string"},
                                      "duration": {"type": "string"},
                                  },
                                  "additionalProperties": False,
                              },
                          },
                          "required": ["condition", "condition_attributes"],
                          "additionalProperties": False,
                      },
                  },
              },
              "required": [schema.EXTRACTIONS_KEY],
              "additionalProperties": False,
          },
      ),
      dict(
          testcase_name="extraction_with_list_attribute",
          examples_data=[
              data.ExampleData(
                  text="Patient has multiple conditions.",
                  extractions=[
                      data.Extraction(
                          extraction_text="conditions",
                          extraction_class="diagnosis",
                          attributes={
                              "conditions": ["diabetes", "hypertension"]
                          },
                      )
                  ],
              )
          ],
          expected_schema={
              "type": "object",
              "properties": {
                  schema.EXTRACTIONS_KEY: {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "properties": {
                              "diagnosis": {"type": "string"},
                              "diagnosis_attributes": {
                                  "type": "object",
                                  "properties": {
                                      "conditions": {
                                          "type": "array",
                                          "items": {"type": "string"},
                                      },
                                  },
                                  "additionalProperties": False,
                              },
                          },
                          "required": ["diagnosis", "diagnosis_attributes"],
                          "additionalProperties": False,
                      },
                  },
              },
              "required": [schema.EXTRACTIONS_KEY],
              "additionalProperties": False,
          },
      ),
  )
  def test_from_examples_constructs_expected_schema(
      self, examples_data, expected_schema
  ):
    openai_schema = schema.OpenAISchema.from_examples(examples_data)
    actual_schema = openai_schema.schema_dict
    self.assertEqual(actual_schema, expected_schema)

  def test_openai_schema_properties(self):
    # Test default properties
    openai_schema = schema.OpenAISchema({})
    self.assertEqual(openai_schema.name, "langextract_extraction")
    self.assertTrue(openai_schema.strict)

  def test_openai_schema_custom_suffix(self):
    # Test with custom attribute suffix
    examples_data = [
        data.ExampleData(
            text="Test",
            extractions=[
                data.Extraction(
                    extraction_text="test",
                    extraction_class="entity",
                    attributes={"value": "test"},
                )
            ],
        )
    ]
    openai_schema = schema.OpenAISchema.from_examples(
        examples_data, attribute_suffix="_props"
    )

    # Check that custom suffix is used
    items_props = openai_schema.schema_dict["properties"][
        schema.EXTRACTIONS_KEY
    ]["items"]["properties"]
    self.assertIn("entity_props", items_props)
    self.assertNotIn("entity_attributes", items_props)


if __name__ == "__main__":
  absltest.main()
