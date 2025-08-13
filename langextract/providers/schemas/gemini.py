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

"""Gemini provider schema implementation."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any

from langextract import data
from langextract import schema

EXTRACTIONS_KEY = schema.EXTRACTIONS_KEY


@dataclasses.dataclass
class GeminiSchema(schema.BaseSchema):
  """Schema implementation for Gemini structured output.

  Converts ExampleData objects into an OpenAPI/JSON-schema definition
  that Gemini can interpret via 'response_schema'.
  """

  _schema_dict: dict

  @property
  def schema_dict(self) -> dict:
    """Returns the schema dictionary."""
    return self._schema_dict

  @schema_dict.setter
  def schema_dict(self, schema_dict: dict) -> None:
    """Sets the schema dictionary."""
    self._schema_dict = schema_dict

  def to_provider_config(self) -> dict[str, Any]:
    """Convert schema to Gemini-specific configuration.

    Returns:
      Dictionary with response_schema and response_mime_type for Gemini API.
    """
    return {
        "response_schema": self._schema_dict,
        "response_mime_type": "application/json",
    }

  @property
  def supports_strict_mode(self) -> bool:
    """Gemini enforces strict JSON schema constraints.

    Returns:
      True, as Gemini can enforce structure strictly via response_schema.
    """
    return True

  @classmethod
  def from_examples(
      cls,
      examples_data: Sequence[data.ExampleData],
      attribute_suffix: str = "_attributes",
  ) -> GeminiSchema:
    """Creates a GeminiSchema from example extractions.

    Builds a JSON-based schema with a top-level "extractions" array. Each
    element in that array is an object containing the extraction class name
    and an accompanying "<class>_attributes" object for its attributes.

    Args:
      examples_data: A sequence of ExampleData objects containing extraction
        classes and attributes.
      attribute_suffix: String appended to each class name to form the
        attributes field name (defaults to "_attributes").

    Returns:
      A GeminiSchema with internal dictionary represents the JSON constraint.
    """
    # Track attribute types for each category
    extraction_categories: dict[str, dict[str, set[type]]] = {}
    for example in examples_data:
      for extraction in example.extractions:
        category = extraction.extraction_class
        if category not in extraction_categories:
          extraction_categories[category] = {}

        if extraction.attributes:
          for attr_name, attr_value in extraction.attributes.items():
            if attr_name not in extraction_categories[category]:
              extraction_categories[category][attr_name] = set()
            extraction_categories[category][attr_name].add(type(attr_value))

    extraction_properties: dict[str, dict[str, Any]] = {}

    for category, attrs in extraction_categories.items():
      extraction_properties[category] = {"type": "string"}

      attributes_field = f"{category}{attribute_suffix}"
      attr_properties = {}

      # Default property for categories without attributes
      if not attrs:
        attr_properties["_unused"] = {"type": "string"}
      else:
        for attr_name, attr_types in attrs.items():
          # List attributes become arrays
          if list in attr_types:
            attr_properties[attr_name] = {
                "type": "array",
                "items": {"type": "string"},
            }
          else:
            attr_properties[attr_name] = {"type": "string"}

      extraction_properties[attributes_field] = {
          "type": "object",
          "properties": attr_properties,
          "nullable": True,
      }

    extraction_schema = {
        "type": "object",
        "properties": extraction_properties,
    }

    schema_dict = {
        "type": "object",
        "properties": {
            EXTRACTIONS_KEY: {"type": "array", "items": extraction_schema}
        },
        "required": [EXTRACTIONS_KEY],
    }

    return cls(_schema_dict=schema_dict)
