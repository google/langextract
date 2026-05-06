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

"""OpenAI provider schema implementation."""

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
JSON_SCHEMA_FORMAT_ERROR = (
    "OpenAI structured output only supports JSON format. "
    "Set format_type=JSON or use_schema_constraints=False."
)


def _nullable(schema_dict: dict[str, Any]) -> dict[str, Any]:
  return {"anyOf": [schema_dict, {"type": "null"}]}


def _attribute_value_schema(attr_types: set[type]) -> dict[str, Any]:
  options: list[dict[str, Any]] = []
  if list in attr_types:
    options.append({"type": "array", "items": {"type": "string"}})
  if not attr_types or attr_types - {list}:
    options.append({"type": "string"})
  options.append({"type": "null"})
  return {"anyOf": options}


def _collect_extraction_categories(
    examples_data: Sequence[data.ExampleData],
) -> dict[str, dict[str, set[type]]]:
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
  return extraction_categories


def _build_extraction_variant(
    category: str,
    attrs: dict[str, set[type]],
    attribute_suffix: str,
) -> dict[str, Any]:
  properties: dict[str, Any] = {category: {"type": "string"}}

  # OpenAI strict mode requires all object keys to be listed in `required`;
  # use null unions for values the model may omit in practice.
  attr_properties = {
      attr_name: _attribute_value_schema(attr_types)
      for attr_name, attr_types in attrs.items()
  }
  attributes_field = f"{category}{attribute_suffix}"
  properties[attributes_field] = _nullable({
      "type": "object",
      "properties": attr_properties,
      "required": list(attr_properties),
      "additionalProperties": False,
  })

  return {
      "type": "object",
      "properties": properties,
      "required": list(properties),
      "additionalProperties": False,
  }


@dataclasses.dataclass
class OpenAISchema(schema.BaseSchema):
  """Schema implementation for OpenAI structured outputs."""

  _schema_dict: dict[str, Any]
  _schema_name: str = _DEFAULT_SCHEMA_NAME
  _strict: bool = True

  @property
  def schema_dict(self) -> dict[str, Any]:
    """Returns the JSON schema dictionary."""
    return self._schema_dict

  @property
  def response_format(self) -> dict[str, Any]:
    """Returns the Chat Completions structured output config."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": self._schema_name,
            "schema": self._schema_dict,
            "strict": self._strict,
        },
    }

  def to_provider_config(self) -> dict[str, Any]:
    """Convert schema to OpenAI-specific configuration."""
    return {"openai_schema": self}

  @property
  def requires_raw_output(self) -> bool:
    """OpenAI structured outputs emit raw JSON without fences."""
    return True

  def validate_format(self, format_handler: fh.FormatHandler) -> None:
    """Validate OpenAI structured output format compatibility."""
    if format_handler.format_type != data.FormatType.JSON:
      raise exceptions.InferenceConfigError(JSON_SCHEMA_FORMAT_ERROR)

    if format_handler.use_fences:
      warnings.warn(
          "OpenAI structured outputs emit native JSON via response_format. "
          "Using fence_output=True may cause parsing issues. Set "
          "fence_output=False.",
          UserWarning,
          stacklevel=3,
      )

    if (
        not format_handler.use_wrapper
        or format_handler.wrapper_key != data.EXTRACTIONS_KEY
    ):
      warnings.warn(
          "OpenAI's response_format schema expects"
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
  ) -> OpenAISchema:
    """Creates an OpenAISchema from example extractions."""
    extraction_categories = _collect_extraction_categories(examples_data)
    variants = [
        _build_extraction_variant(category, attrs, attribute_suffix)
        for category, attrs in extraction_categories.items()
    ]

    if variants:
      extraction_item_schema = {"anyOf": variants}
    else:
      extraction_item_schema = {
          "type": "object",
          "properties": {},
          "required": [],
          "additionalProperties": False,
      }

    schema_dict = {
        "type": "object",
        "properties": {
            data.EXTRACTIONS_KEY: {
                "type": "array",
                "items": extraction_item_schema,
            }
        },
        "required": [data.EXTRACTIONS_KEY],
        "additionalProperties": False,
    }

    return cls(_schema_dict=schema_dict)
