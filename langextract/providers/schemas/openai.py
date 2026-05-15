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
# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Sequence
import copy
import dataclasses
from typing import Any
import warnings

from langextract.core import data
from langextract.core import exceptions
from langextract.core import format_handler as fh
from langextract.core import schema

DEFAULT_SCHEMA_NAME = "langextract_extractions"
JSON_SCHEMA_FORMAT_ERROR = (
    "OpenAI structured output only supports JSON format. "
    "Set format_type=JSON or use_schema_constraints=False."
)


def _nullable(schema_dict: dict[str, Any]) -> dict[str, Any]:
  """Allows optional LangExtract fields under OpenAI strict mode.

  Strict mode requires every object key to be listed as required. A null
  branch preserves LangExtract's optional-attribute semantics.
  """
  return {"anyOf": [schema_dict, {"type": "null"}]}


def _attribute_value_schema(attr_types: set[type]) -> dict[str, Any]:
  """Maps example-observed Python values into a strict JSON-Schema union.

  OpenAI strict mode requires declared object keys to be present, so null
  is always included. Unknown value types fall back to strings because the
  examples are a guide for model output, not a full Python type system.

  Args:
    attr_types: Python types observed for this attribute across all
      examples. May be empty if the attribute was declared but never
      assigned a concrete value.

  Returns:
    A JSON-Schema anyOf fragment.
  """
  options: list[dict[str, Any]] = []
  if list in attr_types:
    options.append({"type": "array", "items": {"type": "string"}})
  if bool in attr_types:
    options.append({"type": "boolean"})
  if int in attr_types:
    options.append({"type": "integer"})
  if float in attr_types:
    options.append({"type": "number"})
  recognized_scalars = {bool, int, float, str, list}
  needs_string_fallback = (
      not attr_types or str in attr_types or attr_types - recognized_scalars
  )
  if needs_string_fallback:
    options.append({"type": "string"})
  options.append({"type": "null"})
  return {"anyOf": options}


def _collect_extraction_categories(
    examples_data: Sequence[data.ExampleData],
) -> dict[str, dict[str, set[type]]]:
  """Keeps schema variants aligned with the examples' extraction classes.

  Args:
    examples_data: Example extractions to inspect.

  Returns:
    A nested mapping from category to attribute name to observed Python
    value types.
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
  return extraction_categories


def _build_extraction_variant(
    category: str,
    attrs: dict[str, set[type]],
    attribute_suffix: str,
) -> dict[str, Any]:
  """Creates the shape OpenAI must enforce for one extraction class.

  Each variant has two top-level keys: category for extracted text and
  category plus suffix for attributes. Both keys are required so OpenAI
  strict mode accepts the shape; the attributes object is wrapped in a
  null union so the model may omit it.

  Args:
    category: The extraction class name, used as the literal property key.
    attrs: Mapping from attribute name to observed Python value types.
    attribute_suffix: Suffix appended to category to form the attributes
      object property key.

  Returns:
    A JSON-Schema object fragment.
  """
  properties: dict[str, Any] = {category: {"type": "string"}}

  # Null unions preserve optional attributes while satisfying OpenAI strict
  # mode's required-key rule.
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


@dataclasses.dataclass(frozen=True)
class OpenAISchema(schema.BaseSchema):
  """Schema implementation for OpenAI structured outputs.

  Converts ExampleData objects into a JSON-Schema that OpenAI's Chat
  Completions API accepts via response_format. Instances are frozen because
  parallel inference shares one schema across worker threads.
  """

  schema_dict: dict[str, Any]
  schema_name: str = DEFAULT_SCHEMA_NAME
  strict: bool = True

  def __post_init__(self) -> None:
    # Copy before publishing the schema to worker threads so caller mutations
    # cannot change in-flight requests.
    object.__setattr__(self, "schema_dict", copy.deepcopy(self.schema_dict))

  @property
  def response_format(self) -> dict[str, Any]:
    """Per-call Chat Completions response_format payload."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": self.schema_name,
            "schema": copy.deepcopy(self.schema_dict),
            "strict": self.strict,
        },
    }

  def to_provider_config(self) -> dict[str, Any]:
    """OpenAI schemas are applied through the provider schema hook."""
    return {}

  @property
  def requires_raw_output(self) -> bool:
    """OpenAI structured outputs emit raw JSON without fences."""
    return True

  def validate_format(self, format_handler: fh.FormatHandler) -> None:
    """Validates OpenAI structured output format compatibility.

    Args:
      format_handler: Format handler describing the caller's desired
        output shape (format type, fence usage, wrapper key).

    Raises:
      InferenceConfigError: if format_type is not JSON.
    """
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
      strict: bool = True,
  ) -> OpenAISchema:
    """Creates an OpenAISchema from example extractions.

    Builds a JSON schema with a top-level extractions array whose items are
    an anyOf union of one strict-mode object variant per extraction class.

    Args:
      examples_data: A sequence of ExampleData objects containing
        extraction classes and attributes.
      attribute_suffix: String appended to each class name to form the
        attributes-object key. Defaults to "_attributes".
      strict: Whether to emit the schema under OpenAI strict mode. Defaults
        to True. Set to False for schemas OpenAI rejects in strict mode. The
        generated schema shape remains constrained in either mode.

    Returns:
      An OpenAISchema instance ready to pass to a provider.
    """
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

    return cls(schema_dict=schema_dict, strict=strict)
