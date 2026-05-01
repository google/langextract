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

"""Provider discovery and registration system.

This module provides centralized provider discovery without circular imports.
It supports both built-in providers and third-party providers via entry points.
"""
from __future__ import annotations

import functools
import importlib
from importlib import metadata

from absl import logging

from langextract.core import base_model

__all__ = ["available_providers", "get_provider_class"]

# Static mapping for built-in providers (always available)
_BUILTINS: dict[str, str] = {
    "gemini": "langextract.providers.gemini:GeminiLanguageModel",
    "ollama": "langextract.providers.ollama:OllamaLanguageModel",
}

# Optional built-in providers (require extra dependencies)
_OPTIONAL_BUILTINS: dict[str, str] = {
    "openai": "langextract.providers.openai:OpenAILanguageModel",
}


def _safe_entry_points(group: str) -> list:
  """Get entry points with Python 3.8-3.12 compatibility.

  Args:
    group: Entry point group name.

  Returns:
    List of entry points in the specified group.
  """
  eps = metadata.entry_points()
  try:
    # Python 3.10+
    return list(eps.select(group=group))
  except AttributeError:
    # Python 3.8-3.9
    return list(getattr(eps, "get")(group, []))


@functools.lru_cache(maxsize=1)
def _discovered() -> dict[str, str]:
  """Cache discovered third-party providers.

  Returns:
    Dictionary mapping provider names to import specs.
  """
  discovered: dict[str, str] = {}
  for ep in _safe_entry_points("langextract.providers"):
    # Handle both old and new entry_points API
    if hasattr(ep, "value"):

      discovered.setdefault(ep.name, ep.value)
    else:
      # Legacy API - construct from module and attr
      value = f"{ep.module}:{ep.attr}" if ep.attr else ep.module
      discovered.setdefault(ep.name, value)

  if discovered:
    logging.debug(
        "Discovered third-party providers: %s", list(discovered.keys())
    )

  return discovered


def available_providers(
    allow_override: bool = False, include_optional: bool = True
) -> dict[str, str]:
  """Get all available providers (built-in + optional + third-party).

  Args:
    allow_override: If True, third-party providers can override built-ins.
                   If False (default), built-ins take precedence.
    include_optional: If True (default), include optional built-in providers
                     that may require extra dependencies.

  Returns:
    Dictionary mapping provider names to import specifications.
  """

  providers = dict(_discovered())

  if include_optional:
    if allow_override:
      # Third-party can override optional built-ins
      providers.update(_OPTIONAL_BUILTINS)
    else:
      # Optional built-ins override third-party
      providers = {**providers, **_OPTIONAL_BUILTINS}

  # Always add core built-ins with highest precedence (unless allow_override)
  if allow_override:
    # Third-party and optional can override core built-ins
    providers.update(_BUILTINS)
  else:
    # Core built-ins take precedence over everything
    providers = {**providers, **_BUILTINS}

  return providers


def _load_class(spec: str) -> type[base_model.BaseLanguageModel]:
  """Load a provider class from module:Class specification.

  Args:
    spec: Import specification in format "module.path:ClassName".

  Returns:
    The loaded provider class.

  Raises:
    ImportError: If the spec is invalid or module cannot be imported.
    TypeError: If the loaded class is not a BaseLanguageModel.
  """
  module_path, _, class_name = spec.partition(":")
  if not module_path or not class_name:
    raise ImportError(
        f"Invalid provider spec '{spec}' - expected 'module:Class'"
    )

  try:
    module = importlib.import_module(module_path)
  except ImportError as e:
    raise ImportError(
        f"Failed to import provider module '{module_path}': {e}"
    ) from e

  try:
    cls = getattr(module, class_name)
  except AttributeError as e:
    raise ImportError(
        f"Provider class '{class_name}' not found in module '{module_path}'"
    ) from e

  # Validate it's a language model
  if not isinstance(cls, type) or not issubclass(
      cls, base_model.BaseLanguageModel
  ):
    # Fallback: check structural compatibility for non-ABC classes
    missing = []
    for method in ("infer", "parse_output"):
      if not hasattr(cls, method):
        missing.append(method)

    if missing:
      raise TypeError(
          f"{cls} is not a BaseLanguageModel and missing required methods:"
          f" {missing}"
      )

    logging.warning(
        "Provider %s does not inherit from BaseLanguageModel but appears"
        " compatible",
        cls,
    )

  return cls


@functools.lru_cache(maxsize=None)  # Cache all loaded classes
def get_provider_class(
    name: str, allow_override: bool = False, include_optional: bool = True
) -> type[base_model.BaseLanguageModel]:
  """Get a provider class by name.

  Args:
    name: Provider name (e.g., "gemini", "openai", "ollama").
    allow_override: If True, allow third-party providers to override built-ins.
    include_optional: If True (default), include optional providers that
                     may require extra dependencies.

  Returns:
    The provider class.

  Raises:
    KeyError: If the provider name is not found.
    ImportError: If the provider module cannot be imported (including
                missing optional dependencies).
    TypeError: If the provider class is not compatible.
  """
  providers = available_providers(allow_override, include_optional)

  if name not in providers:
    available = sorted(providers.keys())
    raise KeyError(
        f"Unknown provider '{name}'. Available providers:"
        f" {', '.join(available) if available else 'none'}.\nHint: Did you"
        " install the necessary extras (e.g., pip install"
        f" langextract[{name}])?"
    )

  return _load_class(providers[name])
