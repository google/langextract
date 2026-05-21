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

"""Configuration system for LangExtract.

This module provides the Config class and global configuration management
with support for:
- Constructor parameters (highest priority)
- Environment variables (LANGEXTRACT_* prefix)
- Default values (lowest priority)

Thread Safety Note:
    - The global configuration (_global_config) is shared across all threads.
      Use configure() to modify it safely.

    - Context-local configuration (via contextvars in _logging.py) is
      thread-local but NOT automatically inherited by new threads.

    For multi-threaded code:
    1. Use configure() for global settings (affects all threads)
    2. Use contextvars.copy_context() to propagate context-local config
    3. Or pass Config objects directly to worker functions
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Optional


ENV_PREFIX = "LANGEXTRACT_"


@dataclass
class Config:
  """Configuration for LangExtract.
  
  Supports three levels of configuration, from highest to lowest priority:
  1. Constructor parameters passed explicitly
  2. Environment variables with LANGEXTRACT_ prefix
  3. Default values defined in the dataclass
  
  Environment variable mapping:
  - LANGEXTRACT_LOG_LEVEL -> log_level
  - LANGEXTRACT_REQUEST_TIMEOUT -> request_timeout
  - LANGEXTRACT_MAX_RETRIES -> max_retries
  - LANGEXTRACT_DEFAULT_MODEL -> default_model
  - LANGEXTRACT_DEFAULT_MAX_TOKENS -> default_max_tokens
  - LANGEXTRACT_CACHE_ENABLED -> cache_enabled
  - LANGEXTRACT_CACHE_DIR -> cache_dir
  """
  
  log_level: str = field(default="WARNING")
  request_timeout: float = field(default=60.0)
  max_retries: int = field(default=3)
  default_model: Optional[str] = field(default=None)
  default_max_tokens: Optional[int] = field(default=None)
  cache_enabled: bool = field(default=True)
  cache_dir: Optional[str] = field(default=None)
  progress_enabled: bool = field(default=True)
  
  def __post_init__(self):
    """Apply environment variable overrides for fields not explicitly set."""
    env_overrides = self._parse_env_vars()
    
    for field_name, env_value in env_overrides.items():
      if self._is_default_value(field_name):
        self._apply_env_value(field_name, env_value)
    
    self._validate()
  
  @classmethod
  def _parse_env_vars(cls) -> dict[str, str]:
    """Parse all relevant environment variables.
    
    Returns:
      A dict mapping field names to their string values from environment.
    """
    result = {}
    field_to_env = cls._get_field_env_mapping()
    
    for field_name, env_var in field_to_env.items():
      if env_var in os.environ:
        result[field_name] = os.environ[env_var]
    
    return result
  
  @staticmethod
  def _get_field_env_mapping() -> dict[str, str]:
    """Get the mapping from field names to environment variable names."""
    return {
        "log_level": f"{ENV_PREFIX}LOG_LEVEL",
        "request_timeout": f"{ENV_PREFIX}REQUEST_TIMEOUT",
        "max_retries": f"{ENV_PREFIX}MAX_RETRIES",
        "default_model": f"{ENV_PREFIX}DEFAULT_MODEL",
        "default_max_tokens": f"{ENV_PREFIX}DEFAULT_MAX_TOKENS",
        "cache_enabled": f"{ENV_PREFIX}CACHE_ENABLED",
        "cache_dir": f"{ENV_PREFIX}CACHE_DIR",
        "progress_enabled": f"{ENV_PREFIX}PROGRESS_ENABLED",
    }
  
  def _is_default_value(self, field_name: str) -> bool:
    """Check if a field is still at its default value.
    
    This is used to determine if environment variables should override it.
    """
    default_value = self._get_field_default(field_name)
    return getattr(self, field_name) == default_value
  
  @classmethod
  def _get_field_default(cls, field_name: str) -> Any:
    """Get the default value for a field from dataclass metadata.
    
    This avoids creating a new Config instance (which would cause recursion).
    """
    for f in dataclasses.fields(cls):
      if f.name == field_name:
        if f.default is not dataclasses.MISSING:
          return f.default
        elif f.default_factory is not dataclasses.MISSING:
          return f.default_factory()
        else:
          return None
    return None
  
  def _apply_env_value(self, field_name: str, env_value: str) -> None:
    """Apply an environment variable value to a field with proper type conversion."""
    if field_name == "log_level":
      self.log_level = env_value
    elif field_name == "request_timeout":
      self.request_timeout = float(env_value)
    elif field_name == "max_retries":
      self.max_retries = int(env_value)
    elif field_name == "default_model":
      self.default_model = env_value if env_value else None
    elif field_name == "default_max_tokens":
      self.default_max_tokens = int(env_value) if env_value else None
    elif field_name == "cache_enabled":
      self.cache_enabled = env_value.lower() in ("1", "true", "yes", "on")
    elif field_name == "cache_dir":
      self.cache_dir = env_value if env_value else None
    elif field_name == "progress_enabled":
      self.progress_enabled = env_value.lower() not in ("0", "false", "no", "off")
  
  def _validate(self) -> None:
    """Validate the configuration values."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if self.log_level.upper() not in valid_levels:
      raise ValueError(
          f"Invalid log_level: {self.log_level}. "
          f"Must be one of: {valid_levels}"
      )
    
    if self.request_timeout <= 0:
      raise ValueError(
          f"request_timeout must be positive, got {self.request_timeout}"
      )
    
    if self.max_retries < 0:
      raise ValueError(
          f"max_retries must be non-negative, got {self.max_retries}"
      )
    
    if self.default_max_tokens is not None and self.default_max_tokens <= 0:
      raise ValueError(
          f"default_max_tokens must be positive, got {self.default_max_tokens}"
      )
  
  def model_copy(self, update: dict | None = None) -> Config:
    """Create a copy of this Config with optional updates.
    
    Args:
      update: Dictionary of field values to update in the new instance.
      
    Returns:
      A new Config instance with the updated values.
    """
    import copy
    new_config = copy.deepcopy(self)
    
    if update:
      for key, value in update.items():
        if hasattr(new_config, key):
          setattr(new_config, key, value)
    
    new_config._validate()
    return new_config


_global_config: Config | None = None


def get_global_config() -> Config:
  """Get the global configuration.
  
  If no global config has been set, creates a new one using defaults
  and environment variables.
  
  Returns:
    The global Config instance.
  """
  global _global_config
  if _global_config is None:
    _global_config = Config()
  return _global_config


def set_global_config(config: Config) -> None:
  """Set the global configuration.
  
  Args:
    config: The Config instance to use as the global configuration.
  """
  global _global_config
  _global_config = config



