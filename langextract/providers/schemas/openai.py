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

"""OpenAI provider schema implementation.

This schema enables OpenAI "Structured Outputs" by providing a JSON Schema via
the `response_format` parameter (type `json_schema`).

Reference: https://platform.openai.com/docs/guides/structured-outputs
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any
import warnings

from langextract.core import data
from langextract.core import exceptions
from langextract.core import format_handler as fh
from langextract.core import schema


_DEFAULT_SCHEMA_NAME = "langextract_extractions"


@dataclasses.dataclass
class OpenAISchema(schema.BaseSchema):
  """Schema implementation for OpenAI Structured Outputs (JSON Schema)."""

  _schema_dict: dict[str, Any]
  _schema_name: str = _DEFAULT_SCHEMA_NAME
  _strict: bool = True

  @property
  def schema_dict(self) -> dict[str, Any]:
    """Returns the JSON schema dictionary."""
    return self._schema_dict

  def to_provider_config(self) -> dict[str, Any]:
    """Convert schema to OpenAI Chat Completions configuration."""
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema_name,
                "schema": self._schema_dict,
                "strict": self._strict,
            },
        }
    }

  @property
  def requires_raw_output(self) -> bool:
    """OpenAI structured outputs return raw JSON without fences."""
    return True

  def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
    """Validate provider kwargs are compatible with OpenAI schema mode."""
    fmt = kwargs.get("format_type", data.FormatType.JSON)
    if fmt != data.FormatType.JSON:
      raise exceptions.InferenceConfigError(
          "OpenAI structured output only supports JSON format. "
          "Set format_type=JSON or use_schema_constraints=False."
      )

  def validate_format(self, format_handler: fh.FormatHandler) -> None:
    """Warn on format settings that conflict with OpenAI structured output."""
    if format_handler.format_type != data.FormatType.JSON:
      warnings.warn(
          "OpenAI structured output only supports JSON format.",
          UserWarning,
          stacklevel=3,
      )
    if format_handler.use_fences:
      warnings.warn(
          "OpenAI structured output returns native JSON. Using fence_output=True"
          " may cause parsing issues. Set fence_output=False.",
          UserWarning,
          stacklevel=3,
      )
    if (
        not format_handler.use_wrapper
        or format_handler.wrapper_key != data.EXTRACTIONS_KEY
    ):
      warnings.warn(
          "OpenAI structured output schema expects wrapper_key="
          f"'{data.EXTRACTIONS_KEY}'. Current settings: use_wrapper="
          f"{format_handler.use_wrapper}, wrapper_key='{format_handler.wrapper_key}'",
          UserWarning,
          stacklevel=3,
      )

  @classmethod
  def from_examples(
      cls,
      examples_data: Sequence[data.ExampleData],
      attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
  ) -> OpenAISchema:
    """Creates an OpenAISchema from example extractions.

    Builds a JSON Schema with a top-level "extractions" array. Each element in
    that array is an object containing the extraction class name and an
    accompanying "<class>_attributes" object for its attributes.

    Notes:
      - OpenAI expects JSON Schema (not OpenAPI). Use `type: ["object", "null"]`
        to represent nullable attribute objects.
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
      attr_properties: dict[str, Any] = {}

      # Default property for categories without attributes
      if not attrs:
        attr_properties["_unused"] = {"type": "string"}
      else:
        for attr_name, attr_types in attrs.items():
          # List attributes become arrays
          if list in attr_types:
            attr_properties[attr_name] = {
                "type": "array",
                "items": {"type": "string"},  # type: ignore[dict-item]
            }
          else:
            attr_properties[attr_name] = {"type": "string"}

      extraction_properties[attributes_field] = {
          "type": ["object", "null"],
          "properties": attr_properties,
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

