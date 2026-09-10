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
# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Sequence
import copy
import dataclasses
from typing import Any
import warnings

from langextract.core import data
from langextract.core import format_handler as fh
from langextract.core import output_schema as output_schema_lib
from langextract.core import schema
from langextract.core import types as core_types


@dataclasses.dataclass
class GeminiSchema(schema.BaseSchema):
  """Schema implementation for Gemini structured output.

  Converts ExampleData objects into Gemini's OpenAPI-style
  `response_schema`. User-authored schemas stay on Gemini's native JSON Schema
  `response_json_schema` field.
  """

  _schema_dict: dict[str, Any]
  _use_json_schema: bool = False
  from_output_schema: bool = dataclasses.field(
      default=False, repr=False, compare=False
  )

  def __post_init__(self) -> None:
    # Direct construction should get the same caller-mutation isolation as
    # from_schema_dict().
    self._schema_dict = copy.deepcopy(self._schema_dict)

  @property
  def schema_dict(self) -> dict[str, Any]:
    """Returns the schema dictionary."""
    return self._schema_dict

  @schema_dict.setter
  def schema_dict(self, schema_dict: dict[str, Any]) -> None:
    """Sets the schema dictionary."""
    self._schema_dict = copy.deepcopy(schema_dict)

  def to_provider_config(self) -> dict[str, Any]:
    """Convert schema to Gemini-specific configuration.

    Returns:
      Dictionary with Gemini response schema config for the provider API.
    """
    schema_key = (
        "response_json_schema" if self._use_json_schema else "response_schema"
    )
    return {
        schema_key: self._schema_dict,
        "response_mime_type": "application/json",
    }

  def output_schema_reserved_provider_kwargs(self) -> frozenset[str]:
    """Provider kwargs that would override an explicit output_schema."""
    return frozenset({
        "gemini_schema",
        "response_json_schema",
        "response_mime_type",
        "response_schema",
    })

  @property
  def requires_raw_output(self) -> bool:
    """Gemini outputs raw JSON via response_mime_type."""
    return True

  def validate_format(self, format_handler: fh.FormatHandler) -> None:
    """Validate Gemini's format requirements.

    Gemini requires:
    - No fence markers (outputs raw JSON via response_mime_type)
    - Wrapper with EXTRACTIONS_KEY (built into LangExtract schemas)
    """
    if format_handler.use_fences:
      warnings.warn(
          "Gemini outputs native JSON via"
          " response_mime_type='application/json'. Using fence_output=True may"
          " cause parsing issues. Set fence_output=False.",
          UserWarning,
          stacklevel=3,
      )

    if (
        not format_handler.use_wrapper
        or format_handler.wrapper_key != data.EXTRACTIONS_KEY
    ):
      warnings.warn(
          "Gemini's response_schema expects"
          f" wrapper_key='{data.EXTRACTIONS_KEY}'. Current settings:"
          f" use_wrapper={format_handler.use_wrapper},"
          f" wrapper_key='{format_handler.wrapper_key}'",
          UserWarning,
          stacklevel=3,
      )

  @classmethod
  def from_examples(
      cls,
      examples_data: Sequence[data.ExampleData],
      attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
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
      A GeminiSchema whose internal dictionary represents the JSON constraint.
    """
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

      if not attrs:
        attr_properties["_unused"] = {"type": "string"}
      else:
        for attr_name, attr_types in attrs.items():
          if list in attr_types:
            attr_properties[attr_name] = {
                "type": "array",
                "items": {"type": "string"},  # type: ignore[dict-item]
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
            data.EXTRACTIONS_KEY: {"type": "array", "items": extraction_schema}
        },
        "required": [data.EXTRACTIONS_KEY],
    }

    return cls(_schema_dict=schema_dict)

  @classmethod
  def from_schema_dict(
      cls, output_schema: core_types.JsonSchema
  ) -> GeminiSchema:
    """Creates a GeminiSchema from a user-provided output schema."""
    return cls(
        _schema_dict=output_schema_lib.validate_output_schema(output_schema),
        _use_json_schema=True,
        from_output_schema=True,
    )
