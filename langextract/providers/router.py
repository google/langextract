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

"""Runtime registry that maps model-ID patterns to provider classes.

This module provides a lazy registration system for LLM providers, allowing
providers to be registered without importing their dependencies until needed.
"""
# pylint: disable=duplicate-code

from __future__ import annotations

import dataclasses
import functools
import importlib
import re
import typing

from absl import logging

from langextract.core import base_model
from langextract.core import exceptions

TLanguageModel = typing.TypeVar("TLanguageModel", bound=base_model.BaseLanguageModel)


@dataclasses.dataclass(frozen=True, slots=True)
class _Entry:
    """Registry entry for a provider."""

    patterns: tuple[re.Pattern[str], ...]
    loader: typing.Callable[[], type[base_model.BaseLanguageModel]]
    priority: int


_ENTRIES: list[_Entry] = []
_ENTRY_KEYS: set[
    tuple[str, tuple[str, ...], int]
] = set()  # (provider_id, patterns, priority)


def _add_entry(
    *,
    provider_id: str,
    patterns: tuple[re.Pattern[str], ...],
    loader: typing.Callable[[], type[base_model.BaseLanguageModel]],
    priority: int,
) -> None:
    """Add an entry to the registry with deduplication."""
    key = (provider_id, tuple(p.pattern for p in patterns), priority)
    if key in _ENTRY_KEYS:
        logging.debug(
            "Skipping duplicate registration for %s with patterns %s at" " priority %d",
            provider_id,
            [p.pattern for p in patterns],
            priority,
        )
        return
    _ENTRY_KEYS.add(key)
    _ENTRIES.append(_Entry(patterns=patterns, loader=loader, priority=priority))
    logging.debug(
        "Registered provider %s with patterns %s at priority %d",
        provider_id,
        [p.pattern for p in patterns],
        priority,
    )


def register_lazy(
    *patterns: str | re.Pattern[str], target: str, priority: int = 0
) -> None:
    """Register a provider lazily using string import path.

    Args:
      *patterns: One or more regex patterns to match model IDs.
      target: Import path in format "module.path:ClassName".
      priority: Priority for resolution (higher wins on conflicts).
    """
    compiled = tuple(re.compile(p) if isinstance(p, str) else p for p in patterns)

    def _loader() -> type[base_model.BaseLanguageModel]:
        module_path, class_name = target.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    _add_entry(
        provider_id=target,
        patterns=compiled,
        loader=_loader,
        priority=priority,
    )


def register(
    *patterns: str | re.Pattern[str], priority: int = 0
) -> typing.Callable[[type[TLanguageModel]], type[TLanguageModel]]:
    """Decorator to register a provider class directly.

    Args:
      *patterns: One or more regex patterns to match model IDs.
      priority: Priority for resolution (higher wins on conflicts).

    Returns:
      Decorator function that registers the class.
    """
    compiled = tuple(re.compile(p) if isinstance(p, str) else p for p in patterns)

    def _decorator(cls: type[TLanguageModel]) -> type[TLanguageModel]:
        def _loader() -> type[base_model.BaseLanguageModel]:
            return cls

        provider_id = f"{cls.__module__}:{cls.__name__}"
        _add_entry(
            provider_id=provider_id,
            patterns=compiled,
            loader=_loader,
            priority=priority,
        )
        return cls

    return _decorator


@functools.lru_cache(maxsize=128)
def resolve(model_id: str) -> type[base_model.BaseLanguageModel]:
    """Resolve a model ID to a provider class.

    Args:
      model_id: The model identifier to resolve.

    Returns:
      The provider class that handles this model ID.

    Raises:
      ValueError: If no provider is registered for the model ID.
    """
    # Providers should be loaded by the caller (e.g., factory.create_model)
    # Router doesn't load providers to avoid circular dependencies

    sorted_entries = sorted(_ENTRIES, key=lambda e: e.priority, reverse=True)

    for entry in sorted_entries:
        if any(pattern.search(model_id) for pattern in entry.patterns):
            return entry.loader()

    available_patterns = [str(p.pattern) for e in _ENTRIES for p in e.patterns]
    raise exceptions.InferenceConfigError(
        f"No provider registered for model_id={model_id!r}. "
        f"Available patterns: {available_patterns}\n"
        "Tip: You can explicitly specify a provider using 'config' parameter "
        "with factory.ModelConfig and a provider class."
    )


@functools.lru_cache(maxsize=128)
def resolve_provider(provider_name: str) -> type[base_model.BaseLanguageModel]:
    """Resolve a provider name to a provider class.

    This allows explicit provider selection by name or class name.

    Args:
      provider_name: The provider name (e.g., "gemini", "openai") or
        class name (e.g., "GeminiLanguageModel").

    Returns:
      The provider class.

    Raises:
      ValueError: If no provider matches the name.
    """
    # Providers should be loaded by the caller (e.g., factory.create_model)
    # Router doesn't load providers to avoid circular dependencies

    for entry in _ENTRIES:
        for pattern in entry.patterns:
            if pattern.pattern == f"^{re.escape(provider_name)}$":
                return entry.loader()

    for entry in _ENTRIES:
        try:
            provider_class = entry.loader()
            class_name = provider_class.__name__
            if provider_name.lower() in class_name.lower():
                return provider_class
        except (ImportError, AttributeError):
            continue

    try:
        pattern = re.compile(f"^{provider_name}$", re.IGNORECASE)
        for entry in _ENTRIES:
            for entry_pattern in entry.patterns:
                if pattern.pattern == entry_pattern.pattern:
                    return entry.loader()
    except re.error:
        pass

    raise exceptions.InferenceConfigError(
        f"No provider found matching: {provider_name!r}. "
        "Available providers can be listed with list_providers()"
    )


def clear() -> None:
    """Clear all registered providers. Mainly for testing."""
    global _ENTRIES  # pylint: disable=global-statement
    _ENTRIES = []
    _ENTRY_KEYS.clear()  # Also clear dedup keys to allow re-registration
    resolve.cache_clear()
    resolve_provider.cache_clear()


def list_providers() -> list[tuple[tuple[str, ...], int]]:
    """List all registered providers with their patterns and priorities.

    Returns:
      List of (patterns, priority) tuples for debugging.
    """
    return [
        (tuple(p.pattern for p in entry.patterns), entry.priority) for entry in _ENTRIES
    ]


def list_entries() -> list[tuple[list[str], int]]:
    """List all registered patterns and priorities. Mainly for debugging.

    Returns:
      List of (patterns, priority) tuples.
    """
    return [([p.pattern for p in e.patterns], e.priority) for e in _ENTRIES]
