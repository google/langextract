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

"""Provider package for LangExtract.

This package contains provider implementations for various LLM backends.
Each provider can be imported independently for fine-grained dependency
management in build systems.
"""

import importlib
from importlib import metadata
import os
import re

from langextract._logging import get_logger
from langextract.core.base_model import (
    GenerateResult,
    LLMProvider,
    Usage,
)
from langextract.providers import builtin_registry
from langextract.providers import router
from langextract.providers.registry import (
    MockProvider,
    ProviderInfo,
    ProviderRegistry,
)

logger = get_logger(__name__)

registry = router

__all__ = [
    "gemini",
    "openai",
    "ollama",
    "router",
    "registry",
    "schemas",
    "load_plugins_once",
    "load_builtins_once",
    "GenerateResult",
    "LLMProvider",
    "Usage",
    "MockProvider",
    "ProviderInfo",
    "ProviderRegistry",
]

_plugins_loaded = False
_builtins_loaded = False


def load_builtins_once() -> None:
    """Load built-in providers to register their patterns.

    Idempotent function that ensures provider patterns are available
    for model resolution. Uses lazy registration to ensure providers
    can be re-registered after registry.clear() even if their modules
    are already in sys.modules.
    """
    global _builtins_loaded

    if _builtins_loaded:
        return

    for config in builtin_registry.BUILTIN_PROVIDERS:
        router.register_lazy(
            *config["patterns"],
            target=config["target"],
            priority=config["priority"],
        )

    _builtins_loaded = True


def _parse_entry_point_value(value: str) -> tuple[str, int | None]:
    """Parse entry point value for priority suffix.

    Supports format: "module.path:ClassName" or "module.path:ClassName:priority=N"

    Args:
        value: The entry point value string.

    Returns:
        Tuple of (target_path, priority). priority is None if not specified.
    """
    priority_match = re.search(r':priority=(\d+)$', value)
    if priority_match:
        target_path = value[:priority_match.start()]
        priority = int(priority_match.group(1))
        return target_path, priority
    return value, None


def _load_entry_point_class(entry_point) -> tuple[type, int | None]:
    """Load a class from an entry point.

    Args:
        entry_point: The entry point object.

    Returns:
        Tuple of (provider_class, priority). priority is None if not specified.
    """
    value = getattr(entry_point, 'value', None)
    priority = None

    if value:
        value, priority = _parse_entry_point_value(value)

    provider_class = entry_point.load()
    return provider_class, priority


def load_plugins_once() -> None:
    """Load provider plugins from installed packages.

    Discovers and loads langextract provider plugins using entry points.
    This function is idempotent - multiple calls have no effect.

    Entry point format:
        [project.entry-points."langextract.providers"]
        my_provider = "my_pkg.provider:MyProvider"
        my_provider_high_prio = "my_pkg.provider:MyProvider:priority=100"

    Priority suffix syntax:
        - ":priority=N" can be appended to set registration priority
        - Higher priority wins when multiple providers match the same pattern
        - Default plugin priority is 20 if not specified
    """
    global _plugins_loaded
    if _plugins_loaded:
        return

    if os.environ.get("LANGEXTRACT_DISABLE_PLUGINS", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.info("Plugin loading disabled via LANGEXTRACT_DISABLE_PLUGINS")
        _plugins_loaded = True
        return

    load_builtins_once()

    try:
        eps = metadata.entry_points()

        if hasattr(eps, "select"):
            provider_eps = eps.select(group="langextract.providers")
        elif hasattr(eps, "get"):
            provider_eps = eps.get("langextract.providers", [])
        else:
            provider_eps = [
                ep
                for ep in eps
                if getattr(ep, "group", None) == "langextract.providers"
            ]

        for entry_point in provider_eps:
            try:
                provider_class, ep_priority = _load_entry_point_class(entry_point)
                logger.info("Loaded provider plugin: %s", entry_point.name)

                if hasattr(provider_class, "get_model_patterns"):
                    patterns = provider_class.get_model_patterns()

                    class_priority = getattr(
                        provider_class,
                        "pattern_priority",
                        20,
                    )

                    priority = ep_priority if ep_priority is not None else class_priority

                    for pattern in patterns:
                        router.register(
                            pattern,
                            priority=priority,
                        )(provider_class)
                    logger.info(
                        "Registered %d patterns for %s with priority %d",
                        len(patterns),
                        entry_point.name,
                        priority,
                    )
                else:
                    class_name = provider_class.__name__
                    if class_name.endswith("LanguageModel"):
                        base_name = class_name[:-13].lower()
                    else:
                        base_name = class_name.lower()

                    class_priority = getattr(
                        provider_class,
                        "pattern_priority",
                        20,
                    )
                    priority = ep_priority if ep_priority is not None else class_priority

                    router.register(
                        f"^{base_name}",
                        priority=priority,
                    )(provider_class)
                    logger.info(
                        "Registered provider %s with pattern ^%s and priority %d",
                        entry_point.name,
                        base_name,
                        priority,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to load provider plugin %s: %s", entry_point.name, e
                )

    except Exception as e:
        logger.warning("Error discovering provider plugins: %s", e)

    _plugins_loaded = True


def _reset_for_testing() -> None:
    """Reset plugin loading state for testing. Should only be used in tests."""
    global _plugins_loaded, _builtins_loaded
    _plugins_loaded = False
    _builtins_loaded = False


def __getattr__(name: str):
    """Lazy loading for submodules."""
    if name == "router":
        return importlib.import_module("langextract.providers.router")
    elif name == "schemas":
        return importlib.import_module("langextract.providers.schemas")
    elif name == "_plugins_loaded":
        return _plugins_loaded
    elif name == "_builtins_loaded":
        return _builtins_loaded
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
