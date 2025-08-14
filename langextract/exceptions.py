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

"""Base exceptions for LangExtract."""

from __future__ import annotations

__all__ = [
    "LangExtractError",
    "InferenceError",
    "InferenceConfigError",
    "InferenceRuntimeError",
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
