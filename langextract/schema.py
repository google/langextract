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

"""Schema definitions and abstractions for structured prompt outputs."""

from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
import enum
from typing import Any


from langextract import data


class ConstraintType(enum.Enum):
  """Enumeration of constraint types."""

  NONE = "none"


# TODO: Remove and decouple Constraint and ConstraintType from Schema class.
@dataclasses.dataclass
class Constraint:
  """Represents a constraint for model output decoding.

  Attributes:
    constraint_type: The type of constraint applied.
  """

  constraint_type: ConstraintType = ConstraintType.NONE


EXTRACTIONS_KEY = "extractions"


class BaseSchema(abc.ABC):
  """Abstract base class for generating structured constraints from examples."""

  @classmethod
  @abc.abstractmethod
  def from_examples(
      cls,
      examples_data: Sequence[data.ExampleData],
      attribute_suffix: str = "_attributes",
  ) -> BaseSchema:
    """Factory method to build a schema instance from example data."""


@dataclasses.dataclass
class GeminiSchema(BaseSchema):
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

      # If no attributes were found for this category, add a default property.
      if not attrs:
        attr_properties["_unused"] = {"type": "string"}
      else:
        for attr_name, attr_types in attrs.items():
          # If we see list type, use array of strings
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
