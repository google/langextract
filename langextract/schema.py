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

"""Public schema helpers and compatibility layer.

New helper functions return plain JSON schema dictionaries for
`extract(output_schema=...)`. Older schema classes remain available here with
deprecation warnings.
"""

from __future__ import annotations

# pylint: disable=undefined-all-variable
import warnings

from langextract._compat import schema
from langextract.core import output_schema as output_schema_lib
from langextract.core import types as core_types

extraction_item_schema = output_schema_lib.extraction_item_schema
extractions_schema = output_schema_lib.extractions_schema
JsonSchema = core_types.JsonSchema
JsonValue = core_types.JsonValue
validate_output_schema = output_schema_lib.validate_output_schema

_COMPAT_NAMES = [
    "ATTRIBUTE_SUFFIX",
    "BaseSchema",
    "Constraint",
    "ConstraintType",
    "EXTRACTIONS_KEY",
    "FormatModeSchema",
    "GeminiSchema",
]

__all__ = [
    *_COMPAT_NAMES,
    "extraction_item_schema",
    "extractions_schema",
    "JsonSchema",
    "JsonValue",
    "validate_output_schema",
]


def __getattr__(name: str):
  """Handle imports with appropriate warnings."""
  core_items = {
      "BaseSchema": ("langextract.core.schema", "BaseSchema"),
      "Constraint": ("langextract.core.schema", "Constraint"),
      "ConstraintType": ("langextract.core.schema", "ConstraintType"),
      "EXTRACTIONS_KEY": ("langextract.core.data", "EXTRACTIONS_KEY"),
      "ATTRIBUTE_SUFFIX": ("langextract.core.data", "ATTRIBUTE_SUFFIX"),
      "FormatModeSchema": ("langextract.core.schema", "FormatModeSchema"),
  }

  if name in core_items:
    mod, attr = core_items[name]
    warnings.warn(
        f"`langextract.schema.{name}` has moved to `{mod}.{attr}`. Please"
        " update your imports. This compatibility layer will be removed in"
        " v2.0.0.",
        FutureWarning,
        stacklevel=2,
    )
    module = __import__(mod, fromlist=[attr])
    return getattr(module, attr)
  elif name == "GeminiSchema":
    return schema.__getattr__(name)

  raise AttributeError(f"module 'langextract.schema' has no attribute '{name}'")
