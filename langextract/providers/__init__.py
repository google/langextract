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
import warnings

from absl import logging

from langextract.providers.builtin_registry import BUILTIN_PROVIDERS
from langextract.providers.router import register
from langextract.providers.router import register_lazy

__all__ = [
    'gemini',
    'openai',
    'ollama',
    'router',
    'schemas',
    'load_plugins_once',
    'load_builtins_once',
]

# Track provider loading for lazy initialization
_PLUGINS_LOADED = False
_BUILTINS_LOADED = False


def load_builtins_once() -> None:
  """Load built-in providers to register their patterns.

  Idempotent function that ensures provider patterns are available
  for model resolution. Uses lazy registration to ensure providers
  can be re-registered after registry.clear() even if their modules
  are already in sys.modules.
  """
  global _BUILTINS_LOADED  # pylint: disable=global-statement

  if _BUILTINS_LOADED:
    return

  # Register built-ins lazily so they can be re-registered after a registry.clear()
  # even if their modules were already imported earlier in the test run.
  for config in BUILTIN_PROVIDERS:
    register_lazy(
        *config['patterns'],
        target=config['target'],
        priority=config['priority'],
    )

  _BUILTINS_LOADED = True


def load_plugins_once() -> None:
  """Load provider plugins from installed packages.

  Discovers and loads langextract provider plugins using entry points.
  This function is idempotent - multiple calls have no effect.
  """
  global _PLUGINS_LOADED  # pylint: disable=global-statement
  if _PLUGINS_LOADED:
    return

  # Check if plugin loading is disabled
  if os.environ.get('LANGEXTRACT_DISABLE_PLUGINS', '').lower() in (
      '1',
      'true',
      'yes',
  ):
    logging.info('Plugin loading disabled via LANGEXTRACT_DISABLE_PLUGINS')
    _PLUGINS_LOADED = True
    return

  # Load built-in providers first
  load_builtins_once()

  try:
    # Get entry points using the metadata API
    eps = metadata.entry_points()

    # Try different APIs based on what's available
    if hasattr(eps, 'select'):
      # Python 3.10+ API
      provider_eps = eps.select(group='langextract.providers')
    elif hasattr(eps, 'get'):
      # Python 3.9 API
      provider_eps = eps.get('langextract.providers', [])
    else:
      # Fallback for older versions
      provider_eps = [
          ep
          for ep in eps
          if getattr(ep, 'group', None) == 'langextract.providers'
      ]

    for entry_point in provider_eps:
      try:
        # Load the entry point
        provider_class = entry_point.load()
        logging.info('Loaded provider plugin: %s', entry_point.name)

        # Register if it has pattern information
        if hasattr(provider_class, 'get_model_patterns'):
          patterns = provider_class.get_model_patterns()
          for pattern in patterns:
            register(
                pattern,
                priority=getattr(
                    provider_class,
                    'pattern_priority',
                    20,  # Default plugin priority
                ),
            )(provider_class)
          logging.info(
              'Registered %d patterns for %s', len(patterns), entry_point.name
          )
      except Exception as e:
        logging.warning(
            'Failed to load provider plugin %s: %s', entry_point.name, e
        )

  except Exception as e:
    logging.warning('Error discovering provider plugins: %s', e)

  _PLUGINS_LOADED = True


def _reset_for_testing() -> None:
  """Reset plugin loading state for testing. Should only be used in tests."""
  global _PLUGINS_LOADED, _BUILTINS_LOADED  # pylint: disable=global-statement
  _PLUGINS_LOADED = False
  _BUILTINS_LOADED = False


def __getattr__(name: str):
  """Lazy loading for submodules."""
  if name == 'router':
    return importlib.import_module('langextract.providers.router')
  elif name == 'registry':  # Backward compat
    warnings.warn(
        'providers.registry is deprecated, use providers.router instead',
        FutureWarning,
        stacklevel=2,
    )
    return importlib.import_module('langextract.providers.router')
  elif name == 'schemas':
    return importlib.import_module('langextract.providers.schemas')
  elif name == '_PLUGINS_LOADED':
    return _PLUGINS_LOADED
  elif name == '_BUILTINS_LOADED':
    return _BUILTINS_LOADED
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
