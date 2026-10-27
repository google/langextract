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

"""Example custom schema implementation for provider plugins."""

from __future__ import annotations

from typing import Any, Sequence

import langextract as lx
from langextract.core import schema as core_schema


class CustomProviderSchema(core_schema.BaseSchema):
  """Example custom schema implementation for a provider plugin.

  This demonstrates how plugins can provide their own schema implementations
  that integrate with LangExtract's schema system. Custom schemas allow
  providers to:

  1. Generate provider-specific constraints from examples
  2. Control output formatting and validation
  3. Optimize for their specific model capabilities

  This example generates a JSON schema from the examples and passes it to
  the Gemini backend (which this example provider wraps) for structured output.
  """

  def __init__(self, schema_dict: dict[str, Any], raw_output: bool = True):
    """Initialize the custom schema.

    Args:
      schema_dict: The generated JSON schema dictionary.
      raw_output: Whether the provider emits raw JSON without fence markers
        (True when JSON mode is guaranteed; False when output needs fencing).
    """
    self._schema_dict = schema_dict
    self._raw_output = raw_output

  @classmethod
  def from_examples(
      cls,
      examples_data: Sequence[lx.data.ExampleData],
      attribute_suffix: str = "_attributes",
  ) -> CustomProviderSchema:
    """Generate schema from example data.

    This method analyzes the provided examples to build a schema that
    captures the structure of expected extractions. Called automatically
    by LangExtract when use_schema_constraints=True.

    Args:
      examples_data: Example extractions to learn from.
      attribute_suffix: Suffix for attribute fields (unused in this example).

    Returns:
      A configured CustomProviderSchema instance.

    Example:
      If examples contain extractions with class "condition" and attribute
      "severity", the schema will constrain the model to only output those
      specific classes and attributes.
    """
    extraction_classes = set()
    attribute_keys = set()

    for example in examples_data:
      for extraction in example.extractions:
        extraction_classes.add(extraction.extraction_class)
        if extraction.attributes:
          attribute_keys.update(extraction.attributes.keys())

    schema_dict = {
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "extraction_class": {
                            "type": "string",
                            "enum": (
                                list(extraction_classes)
                                if extraction_classes
                                else None
                            ),
                        },
                        "extraction_text": {"type": "string"},
                        "attributes": {
                            "type": "object",
                            "properties": {
                                key: {"type": "string"}
                                for key in attribute_keys
                            },
                        },
                    },
                    "required": ["extraction_class", "extraction_text"],
                },
            },
        },
        "required": ["extractions"],
    }

    # Remove enum if no classes found
    if not extraction_classes:
      del schema_dict["properties"]["extractions"]["items"]["properties"][
          "extraction_class"
      ]["enum"]

    return cls(schema_dict, raw_output=True)

  def to_provider_config(self) -> dict[str, Any]:
    """Convert schema to provider-specific configuration.

    This is called after from_examples() and returns kwargs that will be
    passed to the provider's __init__ method. The provider can then use
    these during inference.

    Returns:
      Dictionary of provider kwargs that will be passed to the model.
      In this example, we return both the schema and a flag to enable
      structured output mode.

    Note:
      These kwargs are merged with user-provided kwargs, with user values
      taking precedence (caller-wins merge semantics).
    """
    return {
        "response_schema": self._schema_dict,
        "enable_structured_output": True,
        "output_format": "json",
    }

  @property
  def requires_raw_output(self) -> bool:
    """Whether the provider emits raw JSON/YAML without fence markers.

    Required abstract property of `BaseSchema`. Return True when the
    provider guarantees syntactically valid JSON (so no fence markers
    are needed), False when output should be wrapped in fences.
    """
    return self._raw_output

  @property
  def schema_dict(self) -> dict[str, Any]:
    """Access the underlying schema dictionary.

    Returns:
      The JSON schema dictionary.
    """
    return self._schema_dict
