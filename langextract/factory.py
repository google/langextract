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

"""Factory for creating language model instances.

This module provides a factory pattern for instantiating language models
based on configuration, with support for environment variable resolution
and provider-specific defaults.
"""

from __future__ import annotations

import dataclasses
import os
import typing

from langextract import exceptions
from langextract import inference
from langextract.providers import registry


@dataclasses.dataclass(slots=True, frozen=True)
class ModelConfig:
  """Configuration for instantiating a language model provider.

  Attributes:
    model_id: The model identifier (e.g., "gemini-2.5-flash", "gpt-4o").
    provider: Optional explicit provider name or class name. Use this to
      disambiguate when multiple providers support the same model_id.
    provider_kwargs: Optional provider-specific keyword arguments.
  """

  model_id: str | None = None
  provider: str | None = None
  provider_kwargs: dict[str, typing.Any] = dataclasses.field(
      default_factory=dict
  )


def _kwargs_with_environment_defaults(
    model_id: str, kwargs: dict[str, typing.Any]
) -> dict[str, typing.Any]:
  """Add environment-based defaults to provider kwargs.

  Args:
    model_id: The model identifier.
    kwargs: Existing keyword arguments.

  Returns:
    Updated kwargs with environment defaults.
  """
  resolved = dict(kwargs)

  if "api_key" not in resolved:
    model_lower = model_id.lower()
    env_vars_by_provider = {
        "gemini": ("GEMINI_API_KEY", "LANGEXTRACT_API_KEY"),
        "gpt": ("OPENAI_API_KEY", "LANGEXTRACT_API_KEY"),
    }

    for provider_prefix, env_vars in env_vars_by_provider.items():
      if provider_prefix in model_lower:
        for env_var in env_vars:
          api_key = os.getenv(env_var)
          if api_key:
            resolved["api_key"] = api_key
            break
        break

  if "ollama" in model_id.lower() and "base_url" not in resolved:
    resolved["base_url"] = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )

  return resolved


def create_model(config: ModelConfig) -> inference.BaseLanguageModel:
  """Create a language model instance from configuration.

  Args:
    config: Model configuration with optional model_id and/or provider.

  Returns:
    An instantiated language model provider.

  Raises:
    ValueError: If neither model_id nor provider is specified.
    ValueError: If no provider is registered for the model_id.
    InferenceConfigError: If provider instantiation fails.
  """
  if not config.model_id and not config.provider:
    raise ValueError("Either model_id or provider must be specified")

  try:
    if config.provider:
      provider_class = registry.resolve_provider(config.provider)
    else:
      provider_class = registry.resolve(config.model_id)
  except (ModuleNotFoundError, ImportError) as e:
    raise exceptions.InferenceConfigError(
        "Failed to load provider. "
        "This may be due to missing dependencies. "
        f"Check that all required packages are installed. Error: {e}"
    ) from e

  model_id = config.model_id

  kwargs = _kwargs_with_environment_defaults(
      model_id or config.provider or "", config.provider_kwargs
  )

  if model_id:
    kwargs["model_id"] = model_id

  try:
    return provider_class(**kwargs)
  except (ValueError, TypeError) as e:
    raise exceptions.InferenceConfigError(
        f"Failed to create provider {provider_class.__name__}: {e}"
    ) from e


def create_model_from_id(
    model_id: str | None = None,
    provider: str | None = None,
    **provider_kwargs: typing.Any,
) -> inference.BaseLanguageModel:
  """Convenience function to create a model.

  Args:
    model_id: The model identifier (e.g., "gemini-2.5-flash").
    provider: Optional explicit provider name to disambiguate.
    **provider_kwargs: Optional provider-specific keyword arguments.

  Returns:
    An instantiated language model provider.
  """
  config = ModelConfig(
      model_id=model_id, provider=provider, provider_kwargs=provider_kwargs
  )
  return create_model(config)
