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

EXTRACTIONS_KEY = "extractions"  # Shared key for extraction arrays in JSON/YAML


class ConstraintType(enum.Enum):
  """Enumeration of constraint types."""

  NONE = "none"


# TODO(v2.0.0): Remove and decouple Constraint and ConstraintType from Schema class.
@dataclasses.dataclass
class Constraint:
  """Represents a constraint for model output decoding.

  Attributes:
    constraint_type: The type of constraint applied.
  """

  constraint_type: ConstraintType = ConstraintType.NONE


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

  @abc.abstractmethod
  def to_provider_config(self) -> dict[str, Any]:
    """Convert schema to provider-specific configuration.

    Returns:
      Dictionary of provider kwargs (e.g., response_schema for Gemini).
      Should be a pure data mapping with no side effects.
    """

  @property
  @abc.abstractmethod
  def supports_strict_mode(self) -> bool:
    """Whether the provider emits valid output without needing Markdown fences.

    Returns:
      True when the provider will emit syntactically valid JSON (or other
      machine-parseable format) without needing Markdown fences. This says
      nothing about attribute-level schema enforcement. False otherwise.
    """

  def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
    """Hook to update schema state based on provider kwargs.

    This allows schemas to adjust their behavior based on caller overrides.
    For example, FormatModeSchema uses this to sync its format when the caller
    overrides it, ensuring supports_strict_mode stays accurate.

    Default implementation does nothing. Override if your schema needs to
    respond to provider kwargs.

    Args:
      kwargs: The effective provider kwargs after merging.
    """


class FormatModeSchema(BaseSchema):
  """Generic schema for providers that support format modes (JSON/YAML).

  This schema doesn't enforce structure, only output format. Useful for
  providers that can guarantee syntactically valid JSON or YAML but don't
  support field-level constraints.
  """

  def __init__(self, format_mode: str = "json"):
    """Initialize with a format mode.

    Args:
      format_mode: The output format ("json", "yaml", etc.).
    """
    self._format = format_mode

  @classmethod
  def from_examples(
      cls,
      examples_data: Sequence[data.ExampleData],
      attribute_suffix: str = "_attributes",
  ) -> FormatModeSchema:
    """Create a FormatModeSchema instance.

    Since format mode doesn't use examples for constraints, this
    simply returns a JSON-mode instance.

    Args:
      examples_data: Ignored (kept for interface compatibility).
      attribute_suffix: Ignored (kept for interface compatibility).

    Returns:
      A FormatModeSchema configured for JSON output.
    """
    return cls(format_mode="json")

  def to_provider_config(self) -> dict[str, Any]:
    """Convert to provider configuration.

    Returns:
      Dictionary with format parameter.
    """
    return {"format": self._format}

  @property
  def supports_strict_mode(self) -> bool:
    """Whether the format guarantees valid output without fences.

    Returns:
      True for JSON (guaranteed valid syntax), False for YAML/others.
    """
    return self._format == "json"

  def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
    """Update format based on provider kwargs.

    Args:
      kwargs: The effective provider kwargs after merging.
    """
    if "format" in kwargs:
      self._format = kwargs["format"]


# TODO(v2.0.0): Remove GeminiSchema re-export
# pylint: disable=wrong-import-position,cyclic-import
from langextract.providers.schemas import gemini as gemini_schema

GeminiSchema = gemini_schema.GeminiSchema

__all__ = [
    "BaseSchema",
    "FormatModeSchema",
    "Constraint",
    "ConstraintType",
    "GeminiSchema",
    "EXTRACTIONS_KEY",
]
