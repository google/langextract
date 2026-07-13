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

"""Core error types for LangExtract.

This module defines all base exceptions for LangExtract. These are the
foundational error types that are used throughout the codebase.
"""

from __future__ import annotations

__all__ = [
    "LangExtractError",
    "InferenceError",
    "InferenceConfigError",
    "InferenceRuntimeError",
    "InferenceOutputError",
    "InternalError",
    "InvalidDocumentError",
    "ProviderError",
    "SchemaError",
    "FormatError",
    "FormatParseError",
    "unsupported_output_schema_error",
    "output_schema_fence_error",
    "output_schema_format_error",
    "output_schema_provider_kwargs_error",
]


class LangExtractError(Exception):
  """Base exception for all LangExtract errors.

  All exceptions raised by LangExtract should inherit from this class.
  This allows users to catch all LangExtract-specific errors with a single
  except clause.
  """


class InferenceError(LangExtractError):
  """Base exception for inference-related errors."""


class InferenceConfigError(InferenceError):
  """Exception raised for configuration errors.

  This includes missing API keys, invalid model IDs, or other
  configuration-related issues that prevent model instantiation.
  """


def unsupported_output_schema_error(target: str) -> InferenceConfigError:
  """Builds a consistent error for providers without user-schema support."""
  return InferenceConfigError(
      f"{target} does not support output_schema. "
      "Built-in support is available for Gemini and OpenAI."
  )


def output_schema_fence_error() -> InferenceConfigError:
  """Builds a consistent error for incompatible output schema fences."""
  return InferenceConfigError(
      "output_schema uses provider structured output/raw JSON and cannot be "
      "combined with fence_output=True. Leave fence_output unset or set "
      "fence_output=False."
  )


def output_schema_format_error() -> InferenceConfigError:
  """Builds a consistent error for non-JSON explicit output schemas."""
  return InferenceConfigError(
      "output_schema uses provider structured output/raw JSON and requires "
      "format_type=JSON. Leave format_type unset or set format_type=JSON."
  )


def output_schema_provider_kwargs_error(
    conflicting_keys: list[str],
) -> InferenceConfigError:
  """Builds a consistent error for duplicate provider schema controls."""
  return InferenceConfigError(
      "output_schema cannot be combined with provider kwargs that configure "
      f"provider schema output: {', '.join(conflicting_keys)}. Remove those "
      "provider kwargs and let output_schema configure structured output."
  )


class InferenceRuntimeError(InferenceError):
  """Exception raised for runtime inference errors.

  This includes API call failures, network errors, or other issues
  that occur during inference execution.
  """

  def __init__(
      self,
      message: str,
      *,
      original: BaseException | None = None,
      provider: str | None = None,
  ) -> None:
    """Initialize the runtime error.

    Args:
      message: Error message.
      original: Original exception from the provider SDK.
      provider: Name of the provider that raised the error.
    """
    super().__init__(message)
    self.original = original
    self.provider = provider


class InferenceOutputError(LangExtractError):
  """Exception raised when no scored outputs are available from the language model."""

  def __init__(self, message: str):
    self.message = message
    super().__init__(self.message)


class InvalidDocumentError(LangExtractError):
  """Exception raised when document input is invalid.

  This includes cases like duplicate document IDs or malformed documents.
  """


class InternalError(LangExtractError):
  """Exception raised for internal invariant violations.

  This indicates a bug in LangExtract itself rather than user error.
  """


class ProviderError(LangExtractError):
  """Provider/backend specific error."""


class SchemaError(LangExtractError):
  """Schema validation/serialization error."""


class FormatError(LangExtractError):
  """Base exception for format handling errors."""


class FormatParseError(FormatError):
  """Raised when format parsing fails.

  This consolidates all parsing errors including:
  - Missing fence markers when required
  - Multiple fenced blocks
  - JSON/YAML decode errors
  - Missing wrapper keys
  - Invalid structure
  """
