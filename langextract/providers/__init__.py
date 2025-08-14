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

This package contains the registry system and provider implementations
for various LLM backends.
"""
# pylint: disable=cyclic-import

from importlib import metadata
import os

from absl import logging

from langextract.providers import registry

# Track provider loading for lazy initialization
_PLUGINS_LOADED = False
_BUILTINS_LOADED = False


def load_builtins_once() -> None:
  """Load built-in providers to register their patterns.

  Idempotent function that ensures provider patterns are available
  for model resolution.
  """
  global _BUILTINS_LOADED  # pylint: disable=global-statement
  if _BUILTINS_LOADED:
    return

  # pylint: disable=import-outside-toplevel
  from langextract.providers import gemini  # noqa: F401
  from langextract.providers import ollama  # noqa: F401

  try:
    from langextract.providers import openai  # noqa: F401
  except ImportError:
    logging.debug("OpenAI provider not available (optional dependency)")
  # pylint: enable=import-outside-toplevel

  _BUILTINS_LOADED = True


def load_plugins_once() -> None:
  """Load third-party providers via entry points.

  This function is idempotent and will only load plugins once.
  Set LANGEXTRACT_DISABLE_PLUGINS=1 to disable plugin loading.
  """
  global _PLUGINS_LOADED  # pylint: disable=global-statement
  if _PLUGINS_LOADED:
    return

  if os.getenv("LANGEXTRACT_DISABLE_PLUGINS") == "1":
    logging.info("Plugin loading disabled by LANGEXTRACT_DISABLE_PLUGINS=1")
    _PLUGINS_LOADED = True
    return

  try:
    entry_points_group = metadata.entry_points(group="langextract.providers")
  except Exception as exc:
    logging.debug("No third-party provider entry points found: %s", exc)
    return

  # Set flag after successful entry point query to avoid disabling discovery
  # on transient failures during enumeration.
  _PLUGINS_LOADED = True

  for entry_point in entry_points_group:
    try:
      provider = entry_point.load()
      # Validate provider subclasses but don't auto-register - plugins must
      # use their own @register decorators to control patterns.
      if isinstance(provider, type):
        # pylint: disable=import-outside-toplevel
        # Late import to avoid circular dependency
        from langextract import inference

        if issubclass(provider, inference.BaseLanguageModel):
          logging.info(
              "Loaded third-party provider class from entry point: %s",
              entry_point.name,
          )
        else:
          logging.warning(
              "Entry point %s returned non-provider class %r; ignoring",
              entry_point.name,
              provider,
          )
      else:
        # Module import triggers decorator execution
        logging.debug(
            "Loaded provider module/object from entry point: %s",
            entry_point.name,
        )
    except Exception as exc:
      logging.warning(
          "Failed to load third-party provider %s: %s", entry_point.value, exc
      )


__all__ = ["registry", "load_plugins_once", "load_builtins_once"]
