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
import warnings

from langextract import providers
from langextract.core import base_model
from langextract.core import exceptions
from langextract.core import output_schema as output_schema_lib
from langextract.core import schema as core_schema
from langextract.core import types as core_types
from langextract.providers import router


@dataclasses.dataclass(slots=True, frozen=True)
class ModelConfig:
  """Configuration for instantiating a language model provider.

  Attributes:
    model_id: The model identifier (e.g., "gemini-3.5-flash", "gpt-4o").
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

  if "api_key" not in resolved and not resolved.get("vertexai", False):
    model_lower = model_id.lower()
    env_vars_by_provider = {
        "gemini": ("GEMINI_API_KEY", "LANGEXTRACT_API_KEY"),
        "gpt": ("OPENAI_API_KEY", "LANGEXTRACT_API_KEY"),
        "o1": ("OPENAI_API_KEY", "LANGEXTRACT_API_KEY"),
    }

    for provider_prefix, env_vars in env_vars_by_provider.items():
      if provider_prefix in model_lower:
        found_keys = []
        for env_var in env_vars:
          key_val = os.getenv(env_var)
          if key_val:
            found_keys.append((env_var, key_val))

        if found_keys:
          resolved["api_key"] = found_keys[0][1]

          if len(found_keys) > 1:
            keys_list = ", ".join(k[0] for k in found_keys)
            warnings.warn(
                f"Multiple API keys detected in environment: {keys_list}. "
                f"Using {found_keys[0][0]} and ignoring others.",
                UserWarning,
                stacklevel=3,
            )
        break

  if "ollama" in model_id.lower() and "base_url" not in resolved:
    resolved["base_url"] = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )

  return resolved


def create_model(
    config: ModelConfig,
    examples: typing.Sequence[typing.Any] | None = None,
    use_schema_constraints: bool = False,
    fence_output: bool | None = None,
    return_fence_output: bool = False,
    output_schema: core_types.JsonSchema | None = None,
) -> base_model.BaseLanguageModel | tuple[base_model.BaseLanguageModel, bool]:
  """Create a language model instance from configuration.

  Args:
    config: Model configuration with optional model_id and/or provider.
    examples: Optional examples for schema generation (if use_schema_constraints=True).
    use_schema_constraints: Whether to apply schema constraints from examples.
    fence_output: Explicit fence output preference. If None, computed from schema.
    return_fence_output: If True, also return computed fence_output value.
    output_schema: Optional user-provided JSON schema for the raw LangExtract
      output. When provided, it is used verbatim and example-derived schema
      generation is skipped, regardless of use_schema_constraints. Cannot be
      combined with fence_output=True.

  Returns:
    An instantiated language model provider.
    If return_fence_output=True: Tuple of (model, model.requires_fence_output).

  Raises:
    ValueError: If neither model_id nor provider is specified.
    ValueError: If no provider is registered for the model_id.
    InferenceConfigError: If provider instantiation fails.
  """
  if not config.model_id and not config.provider:
    raise ValueError("Either model_id or provider must be specified")

  if (
      use_schema_constraints
      or fence_output is not None
      or output_schema is not None
  ):
    model = _create_model_with_schema(
        config=config,
        examples=examples,
        use_schema_constraints=use_schema_constraints,
        fence_output=fence_output,
        output_schema=output_schema,
    )
    if return_fence_output:
      return model, model.requires_fence_output
    return model

  providers.load_builtins_once()
  providers.load_plugins_once()

  try:
    if config.provider:
      provider_class = router.resolve_provider(config.provider)
    else:
      provider_class = router.resolve(config.model_id)
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
    model = provider_class(**kwargs)
    if return_fence_output:
      return model, model.requires_fence_output
    return model
  except (ValueError, TypeError) as e:
    raise exceptions.InferenceConfigError(
        f"Failed to create provider {provider_class.__name__}: {e}"
    ) from e


def create_model_from_id(
    model_id: str | None = None,
    provider: str | None = None,
    *,
    output_schema: core_types.JsonSchema | None = None,
    **provider_kwargs: typing.Any,
) -> base_model.BaseLanguageModel:
  """Convenience function to create a model.

  Args:
    model_id: The model identifier (e.g., "gemini-3.5-flash").
    provider: Optional explicit provider name to disambiguate.
    output_schema: Optional user-provided JSON schema for the raw LangExtract
      output.
    **provider_kwargs: Optional provider-specific keyword arguments.

  Returns:
    An instantiated language model provider.
  """
  config = ModelConfig(
      model_id=model_id, provider=provider, provider_kwargs=provider_kwargs
  )
  return create_model(config, output_schema=output_schema)


def _unsupported_output_schema_error(
    config: ModelConfig,
    provider_class: type[base_model.BaseLanguageModel],
) -> exceptions.InferenceConfigError:
  """Build an error naming the model or provider without output_schema support."""
  if config.model_id:
    target = f"model_id={config.model_id!r}"
  elif config.provider:
    target = f"provider={config.provider!r}"
  else:
    target = provider_class.__name__
  return exceptions.unsupported_output_schema_error(f"Provider for {target}")


def _create_model_with_schema(
    config: ModelConfig,
    examples: typing.Sequence[typing.Any] | None = None,
    use_schema_constraints: bool = True,
    fence_output: bool | None = None,
    output_schema: core_types.JsonSchema | None = None,
) -> base_model.BaseLanguageModel:
  """Internal helper to create a model with optional schema constraints.

  This function creates a language model and optionally configures it with
  schema constraints derived from the provided examples. It also computes
  appropriate fence defaulting based on the schema's capabilities.

  Args:
    config: Model configuration with model_id and/or provider.
    examples: Optional sequence of ExampleData for schema generation.
    use_schema_constraints: Whether to generate and apply schema constraints.
    fence_output: Whether to wrap output in markdown fences. If None,
      will be computed based on schema's requires_raw_output.
    output_schema: Optional user-provided JSON schema for the raw LangExtract
      output. When provided, it replaces example-derived schema generation.

  Returns:
    A model instance with fence_output configured appropriately.
  """
  if output_schema is not None and fence_output is True:
    raise exceptions.output_schema_fence_error()
  if output_schema is not None and not output_schema_lib.is_json_format_type(
      config.provider_kwargs.get("format_type")
  ):
    raise exceptions.output_schema_format_error()

  # Must run before resolution regardless of config path.
  providers.load_builtins_once()
  providers.load_plugins_once()

  if config.provider:
    provider_class = router.resolve_provider(config.provider)
  else:
    provider_class = router.resolve(config.model_id)

  schema_instance = None
  if output_schema is not None:
    schema_class = provider_class.get_schema_class()
    if schema_class is None:
      raise _unsupported_output_schema_error(config, provider_class)
    try:
      schema_instance = schema_class.from_schema_dict(output_schema)
    except NotImplementedError as e:
      raise _unsupported_output_schema_error(config, provider_class) from e
    core_schema.mark_from_output_schema(schema_instance)
  elif use_schema_constraints and examples:
    schema_class = provider_class.get_schema_class()
    if schema_class is not None:
      schema_instance = schema_class.from_examples(examples)

  if schema_instance:
    kwargs = schema_instance.to_provider_config()
    provider_kwargs = config.provider_kwargs
    if output_schema is not None:
      reserved = schema_instance.output_schema_reserved_provider_kwargs()
      conflicts = sorted(
          key for key in reserved if provider_kwargs.get(key) is not None
      )
      if conflicts:
        raise exceptions.output_schema_provider_kwargs_error(conflicts)
      provider_kwargs = {
          key: value
          for key, value in provider_kwargs.items()
          if value is not None or key not in reserved
      }
    kwargs.update(provider_kwargs)
  else:
    kwargs = dict(config.provider_kwargs)

  if schema_instance:
    schema_instance.sync_with_provider_kwargs(kwargs)

  # Add environment defaults
  model_id = config.model_id
  kwargs = _kwargs_with_environment_defaults(
      model_id or config.provider or "", kwargs
  )

  if model_id:
    kwargs["model_id"] = model_id

  try:
    model = provider_class(**kwargs)
  except (ValueError, TypeError) as e:
    raise exceptions.InferenceConfigError(
        f"Failed to create provider {provider_class.__name__}: {e}"
    ) from e

  model.apply_schema(schema_instance)
  model.set_fence_output(fence_output)

  return model
