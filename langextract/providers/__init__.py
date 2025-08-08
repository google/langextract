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

# Track whether plugins have been loaded
_PLUGINS_LOADED = False


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

  _PLUGINS_LOADED = True

  try:
    entry_points_group = metadata.entry_points(group="langextract.providers")
  except Exception as exc:
    logging.debug("No third-party provider entry points found: %s", exc)
    return

  for entry_point in entry_points_group:
    try:
      provider = entry_point.load()

      if isinstance(provider, type):
        registry.register(entry_point.name)(provider)
        logging.info(
            "Registered third-party provider from entry point: %s",
            entry_point.name,
        )
      else:
        logging.debug(
            "Loaded provider module from entry point: %s", entry_point.name
        )
    except Exception as exc:
      logging.warning(
          "Failed to load third-party provider %s: %s", entry_point.value, exc
      )


# pylint: disable=wrong-import-position
from langextract.providers import gemini  # noqa: F401
from langextract.providers import ollama  # noqa: F401

try:
  from langextract.providers import openai  # noqa: F401
except ImportError:
  pass

try:
  from langextract.providers import azure_openai  # noqa: F401
except ImportError:
  pass

__all__ = ["registry", "load_plugins_once"]
