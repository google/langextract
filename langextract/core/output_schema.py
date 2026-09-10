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

"""Helpers for user-provided LangExtract output schemas."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Any

from langextract.core import data
from langextract.core import exceptions
from langextract.core import types as core_types

__all__ = [
    "extraction_item_schema",
    "extractions_schema",
    "validate_output_schema",
]

_RESERVED_EXTRACTION_ITEM_KEYS = frozenset({
    "attributes",
    "extraction_class",
    "extraction_text",
})


def is_json_format_type(format_type: Any) -> bool:
  """Returns True when format_type is JSON or unset."""
  if format_type is None:
    return True
  if isinstance(format_type, data.FormatType):
    return format_type is data.FormatType.JSON
  if isinstance(format_type, str):
    return format_type.lower() == data.FormatType.JSON.value
  return False


def validate_output_schema_format_handler(format_handler: Any) -> None:
  """Rejects resolver output settings that conflict with output_schema.

  output_schema constrains the model to LangExtract's raw JSON envelope, so
  the resolver must parse unfenced JSON with the default "extractions"
  wrapper and "_attributes" suffix.

  Args:
    format_handler: Normalized FormatHandler built from resolver params.

  Raises:
    InferenceConfigError: If the handler cannot parse the schema envelope.
  """
  if format_handler.use_fences:
    raise exceptions.output_schema_fence_error()
  if not is_json_format_type(format_handler.format_type):
    raise exceptions.output_schema_format_error()
  if (
      not format_handler.use_wrapper
      or format_handler.wrapper_key != data.EXTRACTIONS_KEY
      or getattr(format_handler, "attribute_suffix", data.ATTRIBUTE_SUFFIX)
      != data.ATTRIBUTE_SUFFIX
  ):
    raise exceptions.InferenceConfigError(
        "output_schema requires LangExtract's default JSON envelope: keep "
        f"use_wrapper=True, wrapper_key={data.EXTRACTIONS_KEY!r}, and "
        f"attribute_suffix={data.ATTRIBUTE_SUFFIX!r}."
    )


def _is_string_sequence(value: Any) -> bool:
  return (
      isinstance(value, Sequence)
      and not isinstance(value, str)
      and all(isinstance(item, str) for item in value)
  )


def _item_schema_branches(items_value: Any) -> list[Mapping[str, Any]]:
  """Returns object-schema branches from a direct object or anyOf union."""
  if not isinstance(items_value, Mapping):
    return []
  if items_value.get("type") == "object":
    return [items_value]
  branches = items_value.get("anyOf")
  if (
      isinstance(branches, Sequence)
      and not isinstance(branches, str)
      and branches
      and all(
          isinstance(branch, Mapping) and branch.get("type") == "object"
          for branch in branches
      )
  ):
    return list(branches)
  return []


def validate_output_schema(
    output_schema: core_types.JsonSchema,
) -> dict[str, Any]:
  """Validates the LangExtract output envelope and returns an isolated copy.

  LangExtract's resolver parses a top-level JSON object with an "extractions"
  array whose items are objects keyed by extraction class, optionally with
  "<class>_attributes" objects. This check only enforces that envelope; the
  provider API validates the JSON schema itself.

  Args:
    output_schema: User-provided JSON schema for the raw model output.

  Returns:
    A deep copy of output_schema.

  Raises:
    InferenceConfigError: If output_schema cannot describe LangExtract's
      output envelope.
  """
  if not isinstance(output_schema, Mapping):
    raise exceptions.InferenceConfigError(
        "output_schema must be a mapping describing a JSON object."
    )

  schema_dict = copy.deepcopy(dict(output_schema))
  if not schema_dict:
    raise exceptions.InferenceConfigError("output_schema must not be empty.")
  if schema_dict.get("type") != "object":
    raise exceptions.InferenceConfigError(
        "output_schema top-level type must be 'object'."
    )

  required = schema_dict.get("required")
  if not _is_string_sequence(required) or data.EXTRACTIONS_KEY not in required:
    raise exceptions.InferenceConfigError(
        "output_schema top-level required must include 'extractions'."
    )

  properties = schema_dict.get("properties")
  if not isinstance(properties, Mapping):
    raise exceptions.InferenceConfigError(
        "output_schema top-level properties must be a mapping."
    )

  extractions_property = properties.get(data.EXTRACTIONS_KEY)
  if (
      not isinstance(extractions_property, Mapping)
      or extractions_property.get("type") != "array"
  ):
    raise exceptions.InferenceConfigError(
        "output_schema must declare 'extractions' as an array property."
    )

  item_branches = _item_schema_branches(extractions_property.get("items"))
  if not item_branches:
    raise exceptions.InferenceConfigError(
        "output_schema must declare 'extractions.items' as an inline object "
        "schema or an inline anyOf of object schemas."
    )

  for branch in item_branches:
    branch_properties = branch.get("properties")
    if not isinstance(branch_properties, Mapping) or not branch_properties:
      raise exceptions.InferenceConfigError(
          "output_schema extraction items must declare extraction-class "
          "properties, such as 'condition'."
      )
    reserved_keys = sorted(
        set(branch_properties).intersection(_RESERVED_EXTRACTION_ITEM_KEYS)
    )
    if reserved_keys:
      raise exceptions.InferenceConfigError(
          "output_schema extraction items use extraction-class keys such as "
          "'condition', not LangExtract's internal field names: "
          + ", ".join(reserved_keys)
      )

  return schema_dict


def _copy_schema_mapping(
    schema_mapping: core_types.JsonSchema,
    argument_name: str,
) -> dict[str, Any]:
  if not isinstance(schema_mapping, Mapping):
    raise exceptions.InferenceConfigError(
        f"{argument_name} must be a mapping describing a JSON schema."
    )
  return copy.deepcopy(dict(schema_mapping))


def extractions_schema(
    item_schema: core_types.JsonSchema,
    *additional_item_schemas: core_types.JsonSchema,
    additional_properties: bool = False,
) -> dict[str, Any]:
  """Wraps extraction item schemas in LangExtract's output envelope.

  Args:
    item_schema: JSON schema for each entry in the "extractions" array. When
      more than one item schema is provided, the helper wraps them in `anyOf`.
      Hand-written item schemas are copied as-is and should include any
      provider-required fields, such as `required` and `additionalProperties`
      for OpenAI strict mode.
    *additional_item_schemas: Additional item schemas for heterogeneous
      extraction classes.
    additional_properties: Value for the envelope's additionalProperties
      setting. Defaults to False so helper output works with OpenAI strict
      structured outputs and Gemini's JSON Schema path.

  Returns:
    A JSON schema dictionary suitable for `extract(output_schema=...)`.
  """
  copied_item_schemas = [
      _copy_schema_mapping(item_schema, "item_schema"),
      *[
          _copy_schema_mapping(
              schema_mapping, f"additional_item_schemas[{index}]"
          )
          for index, schema_mapping in enumerate(additional_item_schemas)
      ],
  ]
  if len(copied_item_schemas) == 1:
    items_schema = copied_item_schemas[0]
  else:
    items_schema = {"anyOf": copied_item_schemas}

  return {
      "type": "object",
      "properties": {
          data.EXTRACTIONS_KEY: {
              "type": "array",
              "items": items_schema,
          }
      },
      "required": [data.EXTRACTIONS_KEY],
      "additionalProperties": additional_properties,
  }


def extraction_item_schema(
    extraction_class: str,
    *,
    attributes: Mapping[str, core_types.JsonSchema] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
  """Builds a schema for one LangExtract extraction object.

  Pair this with `extractions_schema()` to produce the full output envelope.

  Args:
    extraction_class: Extraction class name, such as "emotion".
    attributes: Optional mapping from attribute name to JSON schema.
    additional_properties: Value for each generated object's
      additionalProperties setting, including both the outer extraction item
      object and the nested "<extraction_class>_attributes" object.

  Returns:
    A JSON schema dictionary for an item in the "extractions" array.

  Raises:
    InferenceConfigError: If arguments cannot describe a valid extraction
      object schema.
  """
  if not isinstance(extraction_class, str) or not extraction_class:
    raise exceptions.InferenceConfigError(
        "extraction_class must be a non-empty string."
    )
  if extraction_class.endswith(data.ATTRIBUTE_SUFFIX):
    raise exceptions.InferenceConfigError(
        "extraction_class must not end with reserved suffix "
        f"{data.ATTRIBUTE_SUFFIX!r}."
    )
  if extraction_class in _RESERVED_EXTRACTION_ITEM_KEYS:
    raise exceptions.InferenceConfigError(
        "extraction_class must not use reserved generic key "
        f"{extraction_class!r}."
    )
  if attributes is not None and not isinstance(attributes, Mapping):
    raise exceptions.InferenceConfigError(
        "attributes must be a mapping from names to JSON schemas."
    )
  if attributes is not None and not attributes:
    attributes = None

  properties: dict[str, Any] = {extraction_class: {"type": "string"}}
  required = [extraction_class]

  if attributes is not None:
    attr_properties = {}
    for attr_name, attr_schema in attributes.items():
      if not isinstance(attr_name, str) or not attr_name:
        raise exceptions.InferenceConfigError(
            "attribute names must be non-empty strings."
        )
      attr_properties[attr_name] = _copy_schema_mapping(
          attr_schema, f"attributes[{attr_name!r}]"
      )

    attributes_field = f"{extraction_class}{data.ATTRIBUTE_SUFFIX}"
    properties[attributes_field] = {
        "type": "object",
        "properties": attr_properties,
        "required": list(attr_properties),
        "additionalProperties": additional_properties,
    }
    required.append(attributes_field)

  return {
      "type": "object",
      "properties": properties,
      "required": required,
      "additionalProperties": additional_properties,
  }
