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

"""Unified logging system for LangExtract.

This module provides a centralized logging configuration that integrates with
the Config system. All modules should use get_logger(__name__) to obtain
their logger instances.

Thread Safety Note:
    The config() context manager uses Python's contextvars module for
    thread-safe and async-safe configuration scoping within the same
    execution context. However, note that:

    1. contextvars are NOT automatically inherited by new threads in Python.
       If you use threading.Thread without explicit context propagation,
       the context configuration will not be propagated.

    2. For multi-threaded code:
       - Use configure() for global configuration (affects all threads)
       - Or use contextvars.copy_context() to explicitly propagate context
       - Or pass Config objects directly to functions that need them

    Example with explicit context propagation:
        import threading
        import contextvars
        import langextract as lx

        def worker():
            with lx.config(log_level="DEBUG"):
                # This code will have DEBUG logging
                pass

        with lx.config(log_level="DEBUG"):
            ctx = contextvars.copy_context()
            t = threading.Thread(target=lambda: ctx.run(worker))
            t.start()
            t.join()
"""
from __future__ import annotations

import contextvars
import logging
from typing import Optional

from langextract._config import Config

_LOGGER_CACHE: dict[str, logging.Logger] = {}

_ROOT_LOGGER_NAME = "langextract"

_config_var: contextvars.ContextVar[Optional[Config]] = contextvars.ContextVar(
    "langextract_config", default=None
)


def _ensure_root_logger() -> logging.Logger:
  """Ensure the root langextract logger is properly initialized."""
  root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
  
  if not root_logger.handlers:
    root_logger.addHandler(logging.NullHandler())
  
  root_logger.setLevel(logging.WARNING)
  root_logger.propagate = False
  
  return root_logger


def get_logger(name: str) -> logging.Logger:
  """Get a logger for the given module name.
  
  The logger name will be prefixed with "langextract." to ensure proper
  namespace isolation. The logger's level is determined by the current
  configuration (context-local, global, or default).
  
  Args:
    name: The module name, typically __name__.
    
  Returns:
    A configured logging.Logger instance.
  """
  if name.startswith(_ROOT_LOGGER_NAME + ".") or name == _ROOT_LOGGER_NAME:
    full_name = name
  else:
    full_name = f"{_ROOT_LOGGER_NAME}.{name}"
  
  if full_name in _LOGGER_CACHE:
    return _LOGGER_CACHE[full_name]
  
  _ensure_root_logger()
  
  logger = logging.getLogger(full_name)
  logger.propagate = True
  
  _LOGGER_CACHE[full_name] = logger
  
  return logger


def _get_effective_config() -> Config:
  """Get the current effective configuration.
  
  Checks context-local configuration first, then falls back to global.
  
  Returns:
    The current effective Config instance.
  """
  context_config = _config_var.get()
  if context_config is not None:
    return context_config
  
  from langextract._config import get_global_config
  return get_global_config()


def _apply_config_to_loggers(config: Config) -> None:
  """Apply log level from config to all langextract loggers.
  
  Args:
    config: The Config instance containing log_level.
  """
  root_logger = _ensure_root_logger()
  
  level_name = config.log_level.upper()
  level = getattr(logging, level_name, logging.WARNING)
  
  root_logger.setLevel(level)
  
  for logger in _LOGGER_CACHE.values():
    logger.setLevel(level)


def configure(**kwargs) -> None:
  """Configure global settings for LangExtract.
  
  This function updates the global configuration that affects all loggers
  and other configurable components.
  
  Args:
    **kwargs: Configuration options to update. Valid keys include:
      - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
      - request_timeout: Timeout for API requests in seconds
      - max_retries: Maximum number of retries for failed requests
      - default_model: Default model to use for extraction
      - default_max_tokens: Default maximum tokens for generation
      - cache_enabled: Whether to enable caching
      - cache_dir: Directory to use for caching
  """
  from langextract._config import get_global_config, set_global_config
  
  current = get_global_config()
  updated = current.model_copy(update=kwargs)
  set_global_config(updated)
  _apply_config_to_loggers(updated)


class _ConfigContext:
  """Context manager for temporary configuration overrides.
  
  Uses contextvars to ensure thread-safe and async-safe configuration
  scoping.
  """
  
  def __init__(self, **kwargs):
    self._kwargs = kwargs
    self._token: Optional[contextvars.Token] = None
    self._prev_config: Optional[Config] = None
  
  def __enter__(self):
    base = _get_effective_config()
    temp_config = base.model_copy(update=self._kwargs)
    
    self._prev_config = _config_var.get()
    self._token = _config_var.set(temp_config)
    
    _apply_config_to_loggers(temp_config)
    
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    if self._token is not None:
      _config_var.reset(self._token)
    
    active_config = _get_effective_config()
    _apply_config_to_loggers(active_config)
    
    return False


def config(**kwargs):
  """Create a context manager for temporary configuration overrides.

  This allows you to temporarily change settings (like log level) within
  a specific code block, after which the previous settings are restored.

  Example:
    with langextract.config(log_level="DEBUG"):
      # Debug logging enabled here
      result = langextract.extract(...)
    # Back to previous log level

  Thread Safety Note:
    This context manager uses contextvars, which are thread-local but NOT
    automatically inherited by new threads. If you need to propagate the
    configuration to a new thread, use contextvars.copy_context().

    Example with explicit context propagation:
        import threading
        import contextvars

        with lx.config(log_level="DEBUG"):
            ctx = contextvars.copy_context()
            t = threading.Thread(target=lambda: ctx.run(my_function))
            t.start()
            t.join()

  Args:
    **kwargs: Configuration options to override temporarily.

  Returns:
    A context manager that applies the configuration on enter and
    restores the previous configuration on exit.
  """
  return _ConfigContext(**kwargs)


def get_context_config() -> Optional[Config]:
  """Get the current context-local configuration, if any.
  
  Returns:
    The Config instance set by the current context manager, or None.
  """
  return _config_var.get()


_ensure_root_logger()
