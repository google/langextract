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

"""Tests for the logging and configuration system."""

import logging
import logging.handlers
import os
import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized


class ConfigTest(parameterized.TestCase):
  """Tests for the Config class."""

  def setUp(self):
    super().setUp()
    self._saved_env = {}
    for key in list(os.environ.keys()):
      if key.startswith("LANGEXTRACT_"):
        self._saved_env[key] = os.environ.pop(key)

  def tearDown(self):
    for key, value in self._saved_env.items():
      os.environ[key] = value
    super().tearDown()

  def test_default_values(self):
    """Test that Config has correct default values."""
    from langextract._config import Config

    config = Config()
    self.assertEqual(config.log_level, "WARNING")
    self.assertEqual(config.request_timeout, 60.0)
    self.assertEqual(config.max_retries, 3)
    self.assertIsNone(config.default_model)
    self.assertIsNone(config.default_max_tokens)
    self.assertTrue(config.cache_enabled)
    self.assertIsNone(config.cache_dir)

  def test_constructor_parameters_override_defaults(self):
    """Test that constructor parameters take highest priority."""
    from langextract._config import Config

    config = Config(
        log_level="DEBUG",
        request_timeout=30.0,
        max_retries=5,
        default_model="gemini-2.5-flash",
        default_max_tokens=8192,
        cache_enabled=False,
        cache_dir="/tmp/cache",
    )
    self.assertEqual(config.log_level, "DEBUG")
    self.assertEqual(config.request_timeout, 30.0)
    self.assertEqual(config.max_retries, 5)
    self.assertEqual(config.default_model, "gemini-2.5-flash")
    self.assertEqual(config.default_max_tokens, 8192)
    self.assertFalse(config.cache_enabled)
    self.assertEqual(config.cache_dir, "/tmp/cache")

  def test_environment_variables_override_defaults(self):
    """Test that environment variables override defaults when not explicitly set."""
    from langextract._config import Config

    os.environ["LANGEXTRACT_LOG_LEVEL"] = "INFO"
    os.environ["LANGEXTRACT_REQUEST_TIMEOUT"] = "45.5"
    os.environ["LANGEXTRACT_MAX_RETRIES"] = "10"
    os.environ["LANGEXTRACT_DEFAULT_MODEL"] = "test-model"
    os.environ["LANGEXTRACT_DEFAULT_MAX_TOKENS"] = "4096"
    os.environ["LANGEXTRACT_CACHE_ENABLED"] = "false"
    os.environ["LANGEXTRACT_CACHE_DIR"] = "/env/cache"

    config = Config()
    self.assertEqual(config.log_level, "INFO")
    self.assertEqual(config.request_timeout, 45.5)
    self.assertEqual(config.max_retries, 10)
    self.assertEqual(config.default_model, "test-model")
    self.assertEqual(config.default_max_tokens, 4096)
    self.assertFalse(config.cache_enabled)
    self.assertEqual(config.cache_dir, "/env/cache")

  def test_constructor_overrides_environment(self):
    """Test that constructor parameters override environment variables."""
    from langextract._config import Config

    os.environ["LANGEXTRACT_LOG_LEVEL"] = "INFO"

    config = Config(log_level="DEBUG")
    self.assertEqual(config.log_level, "DEBUG")

  def test_model_copy(self):
    """Test that model_copy creates a copy with updates."""
    from langextract._config import Config

    original = Config(log_level="WARNING", request_timeout=60.0)
    copied = original.model_copy({"log_level": "DEBUG", "max_retries": 10})

    self.assertEqual(original.log_level, "WARNING")
    self.assertEqual(original.request_timeout, 60.0)
    self.assertEqual(original.max_retries, 3)

    self.assertEqual(copied.log_level, "DEBUG")
    self.assertEqual(copied.request_timeout, 60.0)
    self.assertEqual(copied.max_retries, 10)

  @parameterized.named_parameters(
      dict(testcase_name="debug", level="DEBUG"),
      dict(testcase_name="info", level="INFO"),
      dict(testcase_name="warning", level="WARNING"),
      dict(testcase_name="error", level="ERROR"),
      dict(testcase_name="critical", level="CRITICAL"),
  )
  def test_valid_log_levels(self, level):
    """Test that valid log levels are accepted."""
    from langextract._config import Config

    config = Config(log_level=level)
    self.assertEqual(config.log_level, level)

  def test_invalid_log_level_raises(self):
    """Test that invalid log levels raise ValueError."""
    from langextract._config import Config

    with self.assertRaises(ValueError):
      Config(log_level="INVALID")


class GlobalConfigTest(absltest.TestCase):
  """Tests for global configuration management."""

  def setUp(self):
    super().setUp()
    import langextract._config as config_module

    self._original_global = config_module._global_config
    config_module._global_config = None

    self._saved_env = {}
    for key in list(os.environ.keys()):
      if key.startswith("LANGEXTRACT_"):
        self._saved_env[key] = os.environ.pop(key)

  def tearDown(self):
    import langextract._config as config_module

    config_module._global_config = self._original_global

    for key, value in self._saved_env.items():
      os.environ[key] = value
    super().tearDown()

  def test_get_global_config_creates_default(self):
    """Test that get_global_config creates a default config if none exists."""
    from langextract._config import get_global_config, Config

    config = get_global_config()
    self.assertIsInstance(config, Config)
    self.assertEqual(config.log_level, "WARNING")

  def test_set_global_config(self):
    """Test that set_global_config updates the global config."""
    from langextract._config import get_global_config, set_global_config, Config

    new_config = Config(log_level="DEBUG")
    set_global_config(new_config)

    config = get_global_config()
    self.assertIs(config, new_config)
    self.assertEqual(config.log_level, "DEBUG")


class LoggingTest(parameterized.TestCase):
  """Tests for the logging system."""

  def setUp(self):
    super().setUp()
    import langextract._config as config_module
    import langextract._logging as logging_module

    self._original_global = config_module._global_config
    self._original_logger_cache = logging_module._LOGGER_CACHE.copy()

    config_module._global_config = None
    logging_module._LOGGER_CACHE.clear()

    self._saved_env = {}
    for key in list(os.environ.keys()):
      if key.startswith("LANGEXTRACT_"):
        self._saved_env[key] = os.environ.pop(key)

  def tearDown(self):
    import langextract._config as config_module
    import langextract._logging as logging_module

    config_module._global_config = self._original_global
    logging_module._LOGGER_CACHE.clear()
    logging_module._LOGGER_CACHE.update(self._original_logger_cache)

    for key, value in self._saved_env.items():
      os.environ[key] = value
    super().tearDown()

  def test_get_logger_default_level(self):
    """Test that loggers have WARNING level by default."""
    from langextract._logging import get_logger

    logger = get_logger("test_module")
    root_logger = logging.getLogger("langextract")

    self.assertEqual(logger.name, "langextract.test_module")
    self.assertEqual(root_logger.level, logging.WARNING)

  def test_get_logger_with_module_name(self):
    """Test that get_logger properly handles __name__ style module names."""
    from langextract._logging import get_logger

    logger1 = get_logger("langextract.resolver")
    self.assertEqual(logger1.name, "langextract.resolver")

    logger2 = get_logger("resolver")
    self.assertEqual(logger2.name, "langextract.resolver")

    logger3 = get_logger("langextract")
    self.assertEqual(logger3.name, "langextract")

  def test_logger_caching(self):
    """Test that get_logger returns the same logger for the same name."""
    from langextract._logging import get_logger

    logger1 = get_logger("test_module")
    logger2 = get_logger("test_module")

    self.assertIs(logger1, logger2)

  def test_configure_changes_log_level(self):
    """Test that configure() changes the log level."""
    from langextract._logging import get_logger, configure

    root_logger = logging.getLogger("langextract")
    self.assertEqual(root_logger.level, logging.WARNING)

    configure(log_level="DEBUG")
    self.assertEqual(root_logger.level, logging.DEBUG)

    configure(log_level="ERROR")
    self.assertEqual(root_logger.level, logging.ERROR)

  def test_configure_updates_existing_loggers(self):
    """Test that configure() updates all cached loggers."""
    from langextract._logging import get_logger, configure

    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    configure(log_level="DEBUG")

    self.assertEqual(logger1.level, logging.DEBUG)
    self.assertEqual(logger2.level, logging.DEBUG)


class ConfigContextTest(parameterized.TestCase):
  """Tests for the config context manager."""

  def setUp(self):
    super().setUp()
    import langextract._config as config_module
    import langextract._logging as logging_module

    self._original_global = config_module._global_config
    self._original_logger_cache = logging_module._LOGGER_CACHE.copy()

    config_module._global_config = None
    logging_module._LOGGER_CACHE.clear()

    self._saved_env = {}
    for key in list(os.environ.keys()):
      if key.startswith("LANGEXTRACT_"):
        self._saved_env[key] = os.environ.pop(key)

  def tearDown(self):
    import langextract._config as config_module
    import langextract._logging as logging_module

    config_module._global_config = self._original_global
    logging_module._LOGGER_CACHE.clear()
    logging_module._LOGGER_CACHE.update(self._original_logger_cache)

    for key, value in self._saved_env.items():
      os.environ[key] = value
    super().tearDown()

  def test_context_manager_temporarily_changes_level(self):
    """Test that config context manager temporarily changes log level."""
    from langextract._logging import config, get_context_config

    root_logger = logging.getLogger("langextract")
    self.assertEqual(root_logger.level, logging.WARNING)
    self.assertIsNone(get_context_config())

    with config(log_level="DEBUG"):
      self.assertEqual(root_logger.level, logging.DEBUG)
      self.assertIsNotNone(get_context_config())
      self.assertEqual(get_context_config().log_level, "DEBUG")

    self.assertEqual(root_logger.level, logging.WARNING)
    self.assertIsNone(get_context_config())

  def test_context_manager_nested(self):
    """Test that nested context managers work correctly."""
    from langextract._logging import config

    root_logger = logging.getLogger("langextract")
    self.assertEqual(root_logger.level, logging.WARNING)

    with config(log_level="INFO"):
      self.assertEqual(root_logger.level, logging.INFO)

      with config(log_level="DEBUG"):
        self.assertEqual(root_logger.level, logging.DEBUG)

      self.assertEqual(root_logger.level, logging.INFO)

    self.assertEqual(root_logger.level, logging.WARNING)

  def test_context_manager_restores_after_exception(self):
    """Test that context manager restores config after exception."""
    from langextract._logging import config

    root_logger = logging.getLogger("langextract")
    self.assertEqual(root_logger.level, logging.WARNING)

    with self.assertRaises(ValueError):
      with config(log_level="DEBUG"):
        self.assertEqual(root_logger.level, logging.DEBUG)
        raise ValueError("test error")

    self.assertEqual(root_logger.level, logging.WARNING)

  def test_context_manager_thread_isolation(self):
    """Test that contextvar configuration is thread-local.
    
    Note: Logger level modification is global (standard Python logging
    doesn't support context-local levels). However, the contextvar
    configuration storage is thread-local, which allows for concurrent
    use cases where each thread manages its own configuration.
    """
    from langextract._logging import config, get_context_config

    results = {}

    def thread_func(thread_id, level, results_dict):
      with config(log_level=level, request_timeout=float(thread_id) * 10):
        time.sleep(0.01)
        ctx_config = get_context_config()
        results_dict[thread_id] = {
            "log_level": ctx_config.log_level,
            "request_timeout": ctx_config.request_timeout,
        }

    thread1 = threading.Thread(
        target=thread_func, args=(1, "DEBUG", results)
    )
    thread2 = threading.Thread(
        target=thread_func, args=(2, "ERROR", results)
    )

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    self.assertEqual(results[1]["log_level"], "DEBUG")
    self.assertEqual(results[1]["request_timeout"], 10.0)
    self.assertEqual(results[2]["log_level"], "ERROR")
    self.assertEqual(results[2]["request_timeout"], 20.0)

    self.assertIsNone(get_context_config())


class IntegrationTest(parameterized.TestCase):
  """Integration tests for logging and config system."""

  def setUp(self):
    super().setUp()
    import langextract._config as config_module
    import langextract._logging as logging_module

    self._original_global = config_module._global_config
    self._original_logger_cache = logging_module._LOGGER_CACHE.copy()

    config_module._global_config = None
    logging_module._LOGGER_CACHE.clear()

    self._saved_env = {}
    for key in list(os.environ.keys()):
      if key.startswith("LANGEXTRACT_"):
        self._saved_env[key] = os.environ.pop(key)

  def tearDown(self):
    import langextract._config as config_module
    import langextract._logging as logging_module

    config_module._global_config = self._original_global
    logging_module._LOGGER_CACHE.clear()
    logging_module._LOGGER_CACHE.update(self._original_logger_cache)

    for key, value in self._saved_env.items():
      os.environ[key] = value
    super().tearDown()

  def test_logger_emits_messages_at_correct_level(self):
    """Test that logger only emits messages at or above its level."""
    from langextract._logging import configure, get_logger

    logger = get_logger("test_emission")
    handler = logging.handlers.MemoryHandler(capacity=100)
    logger.addHandler(handler)
    logger.propagate = False

    configure(log_level="WARNING")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")

    self.assertEqual(len(handler.buffer), 1)
    self.assertEqual(handler.buffer[0].getMessage(), "warning message")

    handler.buffer.clear()
    configure(log_level="DEBUG")
    logger.debug("debug message")
    logger.info("info message")

    self.assertEqual(len(handler.buffer), 2)

  def test_configure_multiple_options(self):
    """Test that configure can update multiple options at once."""
    from langextract._config import get_global_config
    from langextract._logging import configure

    configure(
        log_level="DEBUG",
        request_timeout=120.0,
        max_retries=0,
        cache_enabled=False,
    )

    config = get_global_config()
    self.assertEqual(config.log_level, "DEBUG")
    self.assertEqual(config.request_timeout, 120.0)
    self.assertEqual(config.max_retries, 0)
    self.assertFalse(config.cache_enabled)

  def test_context_manager_multiple_options(self):
    """Test that config context manager can set multiple options."""
    from langextract._config import get_global_config
    from langextract._logging import config, get_context_config

    with config(
        log_level="DEBUG",
        request_timeout=30.0,
        max_retries=5,
    ):
      ctx_config = get_context_config()
      self.assertEqual(ctx_config.log_level, "DEBUG")
      self.assertEqual(ctx_config.request_timeout, 30.0)
      self.assertEqual(ctx_config.max_retries, 5)

    global_config = get_global_config()
    self.assertEqual(global_config.log_level, "WARNING")
    self.assertEqual(global_config.request_timeout, 60.0)
    self.assertEqual(global_config.max_retries, 3)


class ProgressConfigTest(parameterized.TestCase):
  """Tests for progress_enabled configuration."""

  def setUp(self):
    super().setUp()
    import langextract._config as config_module
    import langextract._logging as logging_module

    self._original_global = config_module._global_config
    self._original_logger_cache = logging_module._LOGGER_CACHE.copy()

    config_module._global_config = None
    logging_module._LOGGER_CACHE.clear()

    self._saved_env = {}
    for key in list(os.environ.keys()):
      if key.startswith("LANGEXTRACT_"):
        self._saved_env[key] = os.environ.pop(key)

  def tearDown(self):
    import langextract._config as config_module
    import langextract._logging as logging_module

    config_module._global_config = self._original_global
    logging_module._LOGGER_CACHE.clear()
    logging_module._LOGGER_CACHE.update(self._original_logger_cache)

    for key, value in self._saved_env.items():
      os.environ[key] = value
    super().tearDown()

  def test_progress_enabled_default(self):
    """Test that progress_enabled is True by default."""
    from langextract._config import Config

    config = Config()
    self.assertTrue(config.progress_enabled)

  def test_progress_enabled_constructor(self):
    """Test that progress_enabled can be set via constructor."""
    from langextract._config import Config

    config = Config(progress_enabled=False)
    self.assertFalse(config.progress_enabled)

  def test_progress_enabled_environment_variable(self):
    """Test that progress_enabled can be set via environment variable."""
    from langextract._config import Config

    os.environ["LANGEXTRACT_PROGRESS_ENABLED"] = "0"
    config = Config()
    self.assertFalse(config.progress_enabled)

  def test_progress_enabled_environment_variable_true(self):
    """Test that progress_enabled can be set to True via environment variable."""
    from langextract._config import Config

    os.environ["LANGEXTRACT_PROGRESS_ENABLED"] = "1"
    config = Config()
    self.assertTrue(config.progress_enabled)

  def test_progress_module_checks_config(self):
    """Test that progress module checks the config."""
    from langextract._logging import configure
    from langextract.progress import _is_progress_enabled

    configure(progress_enabled=False)
    self.assertFalse(_is_progress_enabled())

    configure(progress_enabled=True)
    self.assertTrue(_is_progress_enabled())


class AbslVerbosityCompatibilityTest(parameterized.TestCase):
  """Tests for absl verbosity compatibility.

  These tests verify that langextract.configure(log_level='DEBUG')
  works correctly even when absl flags are in use.
  """

  def setUp(self):
    super().setUp()
    import langextract._config as config_module
    import langextract._logging as logging_module

    self._original_global = config_module._global_config
    self._original_logger_cache = logging_module._LOGGER_CACHE.copy()

    config_module._global_config = None
    logging_module._LOGGER_CACHE.clear()

    self._saved_env = {}
    for key in list(os.environ.keys()):
      if key.startswith("LANGEXTRACT_"):
        self._saved_env[key] = os.environ.pop(key)

  def tearDown(self):
    import langextract._config as config_module
    import langextract._logging as logging_module

    config_module._global_config = self._original_global
    logging_module._LOGGER_CACHE.clear()
    logging_module._LOGGER_CACHE.update(self._original_logger_cache)

    for key, value in self._saved_env.items():
      os.environ[key] = value
    super().tearDown()

  def test_configure_works_without_absl_flags_initialized(self):
    """Test that configure works when absl flags haven't been initialized."""
    from langextract._logging import configure, get_logger

    logger = get_logger("test_absl")
    root_logger = logging.getLogger("langextract")

    configure(log_level="DEBUG")
    self.assertEqual(root_logger.level, logging.DEBUG)

  def test_configure_overrides_effective_log_level(self):
    """Test that configure directly sets the effective log level.

    Priority: langextract.configure() > environment variables > defaults.
    """
    from langextract._logging import configure, get_logger

    logger = get_logger("test_override")
    handler = logging.handlers.MemoryHandler(capacity=100)
    logger.addHandler(handler)
    logger.propagate = False

    configure(log_level="WARNING")
    logger.debug("should not appear")
    self.assertEqual(len(handler.buffer), 0)

    configure(log_level="DEBUG")
    logger.debug("should appear")
    self.assertEqual(len(handler.buffer), 1)
    self.assertEqual(handler.buffer[0].getMessage(), "should appear")

  def test_configure_sets_all_cached_loggers(self):
    """Test that configure updates all cached loggers."""
    from langextract._logging import configure, get_logger

    logger1 = get_logger("logger1")
    logger2 = get_logger("logger2")
    logger3 = get_logger("logger3")

    configure(log_level="ERROR")

    self.assertEqual(logger1.level, logging.ERROR)
    self.assertEqual(logger2.level, logging.ERROR)
    self.assertEqual(logger3.level, logging.ERROR)

    configure(log_level="DEBUG")

    self.assertEqual(logger1.level, logging.DEBUG)
    self.assertEqual(logger2.level, logging.DEBUG)
    self.assertEqual(logger3.level, logging.DEBUG)


class ProductionWorkflowTest(parameterized.TestCase):
  """Integration tests for production workflows from README.

  These tests verify the three configuration methods from README:
  1. Using configure() (global setting)
  2. Using config() context manager (temporary setting)
  3. Using environment variables
  """

  def setUp(self):
    super().setUp()
    import langextract._config as config_module
    import langextract._logging as logging_module

    self._original_global = config_module._global_config
    self._original_logger_cache = logging_module._LOGGER_CACHE.copy()

    config_module._global_config = None
    logging_module._LOGGER_CACHE.clear()

    self._saved_env = {}
    for key in list(os.environ.keys()):
      if key.startswith("LANGEXTRACT_"):
        self._saved_env[key] = os.environ.pop(key)

  def tearDown(self):
    import langextract._config as config_module
    import langextract._logging as logging_module

    config_module._global_config = self._original_global
    logging_module._LOGGER_CACHE.clear()
    logging_module._LOGGER_CACHE.update(self._original_logger_cache)

    for key, value in self._saved_env.items():
      os.environ[key] = value
    super().tearDown()

  def test_workflow_1_configure_global(self):
    """Test workflow 1: Using configure() for global setting.

    From README:
    import langextract as lx
    lx.configure(log_level="DEBUG")
    """
    from langextract._logging import configure, get_logger

    resolver_logger = get_logger("resolver")
    providers_logger = get_logger("providers")
    annotation_logger = get_logger("annotation")
    chunking_logger = get_logger("chunking")

    root_logger = logging.getLogger("langextract")
    self.assertEqual(root_logger.level, logging.WARNING)

    configure(log_level="DEBUG")

    self.assertEqual(root_logger.level, logging.DEBUG)
    self.assertEqual(resolver_logger.level, logging.DEBUG)
    self.assertEqual(providers_logger.level, logging.DEBUG)
    self.assertEqual(annotation_logger.level, logging.DEBUG)
    self.assertEqual(chunking_logger.level, logging.DEBUG)

  def test_workflow_2_context_manager_temporary(self):
    """Test workflow 2: Using config() context manager for temporary setting.

    From README:
    with lx.config(log_level="DEBUG"):
        result = lx.extract(...)  # Debug logs enabled here
    # Back to previous log level
    """
    from langextract._config import get_global_config
    from langextract._logging import config, get_logger

    resolver_logger = get_logger("resolver")
    root_logger = logging.getLogger("langextract")

    self.assertEqual(root_logger.level, logging.WARNING)
    global_config = get_global_config()
    self.assertEqual(global_config.log_level, "WARNING")

    with config(log_level="DEBUG"):
      self.assertEqual(root_logger.level, logging.DEBUG)

    self.assertEqual(root_logger.level, logging.WARNING)

  def test_workflow_3_environment_variables(self):
    """Test workflow 3: Using environment variables.

    From README:
    export LANGEXTRACT_LOG_LEVEL="DEBUG"
    export LANGEXTRACT_REQUEST_TIMEOUT="120.0"
    export LANGEXTRACT_MAX_RETRIES="5"
    """
    import langextract._config as config_module

    config_module._global_config = None

    os.environ["LANGEXTRACT_LOG_LEVEL"] = "DEBUG"
    os.environ["LANGEXTRACT_REQUEST_TIMEOUT"] = "120.0"
    os.environ["LANGEXTRACT_MAX_RETRIES"] = "5"
    os.environ["LANGEXTRACT_CACHE_ENABLED"] = "false"

    from langextract._config import get_global_config

    global_config = get_global_config()
    self.assertEqual(global_config.log_level, "DEBUG")
    self.assertEqual(global_config.request_timeout, 120.0)
    self.assertEqual(global_config.max_retries, 5)
    self.assertFalse(global_config.cache_enabled)

  def test_all_modules_use_consistent_logger(self):
    """Test that all key modules use the unified logger.

    Verify that resolver, providers, annotation, and chunking modules
    all use get_logger() with proper naming.
    """
    from langextract._logging import get_logger

    module_loggers = [
        ("langextract.resolver", get_logger("resolver")),
        ("langextract.providers", get_logger("providers")),
        ("langextract.annotation", get_logger("annotation")),
        ("langextract.chunking", get_logger("chunking")),
        ("langextract.progress", get_logger("progress")),
    ]

    for expected_name, logger in module_loggers:
      self.assertEqual(logger.name, expected_name)
      self.assertTrue(logger.name.startswith("langextract."))

  def test_configuration_priority_order(self):
    """Test configuration priority order.

    Priority (highest to lowest):
    1. Explicit configure() call
    2. Environment variables
    3. Default values
    """
    import langextract._config as config_module

    os.environ["LANGEXTRACT_LOG_LEVEL"] = "ERROR"
    os.environ["LANGEXTRACT_REQUEST_TIMEOUT"] = "30.0"

    config_module._global_config = None

    from langextract._config import get_global_config
    from langextract._logging import configure

    env_config = get_global_config()
    self.assertEqual(env_config.log_level, "ERROR")
    self.assertEqual(env_config.request_timeout, 30.0)

    configure(log_level="DEBUG", request_timeout=120.0)

    explicit_config = get_global_config()
    self.assertEqual(explicit_config.log_level, "DEBUG")
    self.assertEqual(explicit_config.request_timeout, 120.0)


if __name__ == "__main__":
  absltest.main()
