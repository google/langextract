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


if __name__ == "__main__":
  absltest.main()
