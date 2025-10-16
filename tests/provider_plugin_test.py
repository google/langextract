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

"""Tests for provider plugin system.

Note: This file contains test helper classes that intentionally have
few public methods. The too-few-public-methods warnings are expected.
"""

from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types as builtin_types
from unittest import mock
import uuid

from absl.testing import absltest
import pytest

import langextract as lx
from langextract.core import base_model
from langextract.core import types


def _create_mock_entry_points(entry_points_list):
  """Create a mock EntryPoints object for testing.

  Args:
    entry_points_list: List of entry points to return for langextract.providers.

  Returns:
    A mock object that behaves like importlib.metadata.EntryPoints.
  """

  class MockEntryPoints:  # pylint: disable=too-few-public-methods
    """Mock EntryPoints that implements select() method."""

    def select(self, group=None):
      if group == "langextract.providers":
        return entry_points_list
      return []

  return MockEntryPoints()


class PluginSmokeTest(absltest.TestCase):
  """Basic smoke tests for plugin loading functionality."""

  def setUp(self):
    super().setUp()
    lx.providers.registry.clear()
    # Always reset both flags to ensure clean state
    lx.providers._reset_for_testing()
    # Register cleanup
    self.addCleanup(lx.providers.registry.clear)
    self.addCleanup(lx.providers._reset_for_testing)

  def test_plugin_discovery_and_usage(self):
    """Test plugin discovery via entry points.

    Entry points can return a class or module. Registration happens via
    the @register decorator in both cases.
    """

    def _ep_load():
      @lx.providers.registry.register(r"^plugin-model")
      class PluginProvider(base_model.BaseLanguageModel):  # pylint: disable=too-few-public-methods

        def __init__(self, model_id=None, **kwargs):
          super().__init__()
          self.model_id = model_id

        def infer(self, batch_prompts, **kwargs):
          return [[types.ScoredOutput(score=1.0, output="ok")]]

      return PluginProvider

    ep = builtin_types.SimpleNamespace(
        name="plugin_provider",
        group="langextract.providers",
        value="my_pkg:PluginProvider",
        load=_ep_load,
    )

    with mock.patch.object(
        metadata, "entry_points", return_value=_create_mock_entry_points([ep])
    ):
      lx.providers.load_plugins_once()

    resolved_cls = lx.providers.registry.resolve("plugin-model-123")
    self.assertEqual(
        resolved_cls.__name__,
        "PluginProvider",
        "Provider should be resolvable after plugin load",
    )

    cfg = lx.factory.ModelConfig(model_id="plugin-model-123")
    model = lx.factory.create_model(cfg)

    out = model.infer(["hi"])[0][0].output
    self.assertEqual(out, "ok", "Provider should return expected output")

  def test_plugin_disabled_by_env_var(self):
    """Test that LANGEXTRACT_DISABLE_PLUGINS=1 prevents plugin loading."""

    with mock.patch.dict("os.environ", {"LANGEXTRACT_DISABLE_PLUGINS": "1"}):
      with mock.patch.object(metadata, "entry_points") as mock_ep:
        lx.providers.load_plugins_once()
        mock_ep.assert_not_called()

  def test_handles_import_errors_gracefully(self):
    """Test that import errors during plugin loading don't crash."""

    def _bad_load():
      raise ImportError("Plugin not found")

    bad_ep = builtin_types.SimpleNamespace(
        name="bad_plugin",
        group="langextract.providers",
        value="bad_pkg:BadProvider",
        load=_bad_load,
    )

    with mock.patch.object(
        metadata,
        "entry_points",
        return_value=_create_mock_entry_points([bad_ep]),
    ):
      lx.providers.load_plugins_once()

      providers = lx.providers.registry.list_providers()
      self.assertIsInstance(
          providers,
          list,
          "Registry should remain functional after import error",
      )
      # Built-in providers should still be loaded even if plugin fails
      self.assertGreater(
          len(providers),
          0,
          "Built-in providers should still be available after plugin failure",
      )

  def test_load_plugins_once_is_idempotent(self):
    """Test that load_plugins_once only discovers once."""

    def _ep_load():
      @lx.providers.registry.register(r"^plugin-model")
      class Plugin(base_model.BaseLanguageModel):  # pylint: disable=too-few-public-methods

        def infer(self, *a, **k):
          return [[types.ScoredOutput(score=1.0, output="ok")]]

      return Plugin

    ep = builtin_types.SimpleNamespace(
        name="plugin_provider",
        group="langextract.providers",
        value="pkg:Plugin",
        load=_ep_load,
    )

    with mock.patch.object(
        metadata, "entry_points", return_value=_create_mock_entry_points([ep])
    ) as m:
      lx.providers.load_plugins_once()
      lx.providers.load_plugins_once()  # should be a no-op
      self.assertEqual(m.call_count, 1, "Discovery should happen only once")

  def test_non_subclass_entry_point_does_not_crash(self):
    """Test that non-BaseLanguageModel classes don't crash the system."""

    class NotAProvider:  # pylint: disable=too-few-public-methods
      """Dummy class to test non-provider handling."""

    bad_ep = builtin_types.SimpleNamespace(
        name="bad",
        group="langextract.providers",
        value="bad:NotAProvider",
        load=lambda: NotAProvider,
    )

    with mock.patch.object(
        metadata,
        "entry_points",
        return_value=_create_mock_entry_points([bad_ep]),
    ):
      lx.providers.load_plugins_once()
      # The system should remain functional even if a bad provider is loaded
      # Trying to use it would fail, but discovery shouldn't crash
      providers = lx.providers.registry.list_providers()
      self.assertIsInstance(
          providers,
          list,
          "Registry should remain functional with bad provider",
      )
      with self.assertRaisesRegex(
          lx.exceptions.InferenceConfigError, "No provider registered"
      ):
        lx.providers.registry.resolve("bad")

  def test_plugin_priority_override_core_provider(self):
    """Plugin with higher priority should override core provider on conflicts."""

    lx.providers.registry.clear()
    lx.providers._PLUGINS_LOADED = False

    def _ep_load():
      @lx.providers.registry.register(r"^gemini", priority=50)
      class OverrideGemini(base_model.BaseLanguageModel):  # pylint: disable=too-few-public-methods

        def infer(self, batch_prompts, **kwargs):
          return [[types.ScoredOutput(score=1.0, output="override")]]

      return OverrideGemini

    ep = builtin_types.SimpleNamespace(
        name="override_gemini",
        group="langextract.providers",
        value="pkg:OverrideGemini",
        load=_ep_load,
    )

    with mock.patch.object(
        metadata, "entry_points", return_value=_create_mock_entry_points([ep])
    ):
      lx.providers.load_plugins_once()

    # Core gemini registers with priority 10 in providers.gemini
    # Our plugin registered with priority 50; it should win.
    resolved = lx.providers.registry.resolve("gemini-2.5-flash")
    self.assertEqual(resolved.__name__, "OverrideGemini")

  def test_resolve_provider_for_plugin(self):
    """resolve_provider should find plugin by class name and name-insensitive."""

    lx.providers.registry.clear()
    lx.providers._PLUGINS_LOADED = False

    def _ep_load():
      @lx.providers.registry.register(r"^plugin-resolve")
      class ResolveMePlease(base_model.BaseLanguageModel):  # pylint: disable=too-few-public-methods

        def infer(self, batch_prompts, **kwargs):
          return [[types.ScoredOutput(score=1.0, output="ok")]]

      return ResolveMePlease

    ep = builtin_types.SimpleNamespace(
        name="resolver_plugin",
        group="langextract.providers",
        value="pkg:ResolveMePlease",
        load=_ep_load,
    )

    with mock.patch.object(
        metadata, "entry_points", return_value=_create_mock_entry_points([ep])
    ):
      lx.providers.load_plugins_once()

    cls_by_exact = lx.providers.registry.resolve_provider("ResolveMePlease")
    self.assertEqual(cls_by_exact.__name__, "ResolveMePlease")

    cls_by_partial = lx.providers.registry.resolve_provider("resolveme")
    self.assertEqual(cls_by_partial.__name__, "ResolveMePlease")

  def test_plugin_with_custom_schema(self):
    """Test that a plugin can provide its own schema implementation."""

    class TestPluginSchema(lx.schema.BaseSchema):
      """Test schema implementation."""

      def __init__(self, config):
        self._config = config

      @classmethod
      def from_examples(cls, examples_data, attribute_suffix="_attributes"):
        return cls({"generated": True, "count": len(examples_data)})

      def to_provider_config(self):
        return {"custom_schema": self._config}

      @property
      def requires_raw_output(self):
        return True

    def _ep_load():
      @lx.providers.registry.register(r"^custom-schema-test")
      class SchemaTestProvider(base_model.BaseLanguageModel):

        def __init__(self, model_id=None, **kwargs):
          super().__init__()
          self.model_id = model_id
          self.schema_config = kwargs.get("custom_schema")

        @classmethod
        def get_schema_class(cls):
          return TestPluginSchema

        def infer(self, batch_prompts, **kwargs):
          output = (
              f"Schema={self.schema_config}"
              if self.schema_config
              else "No schema"
          )
          return [[types.ScoredOutput(score=1.0, output=output)]]

      return SchemaTestProvider

    ep = builtin_types.SimpleNamespace(
        name="schema_test",
        group="langextract.providers",
        value="test:SchemaTestProvider",
        load=_ep_load,
    )

    with mock.patch.object(
        metadata, "entry_points", return_value=_create_mock_entry_points([ep])
    ):
      lx.providers.load_plugins_once()

    provider_cls = lx.providers.registry.resolve("custom-schema-test-v1")
    self.assertEqual(
        provider_cls.get_schema_class().__name__,
        "TestPluginSchema",
        "Plugin should provide custom schema class",
    )

    examples = [
        lx.data.ExampleData(
            text="Test",
            extractions=[
                lx.data.Extraction(
                    extraction_class="test",
                    extraction_text="test text",
                )
            ],
        )
    ]

    config = lx.factory.ModelConfig(model_id="custom-schema-test-v1")
    model = lx.factory._create_model_with_schema(
        config=config,
        examples=examples,
        use_schema_constraints=True,
        fence_output=None,
    )

    self.assertIsNotNone(
        model.schema_config,
        "Model should have schema config applied",
    )
    self.assertTrue(
        model.schema_config["generated"],
        "Schema should be generated from examples",
    )
    self.assertFalse(
        model.requires_fence_output,
        "Schema outputs raw JSON, no fences needed",
    )


class PluginE2ETest(absltest.TestCase):
  """End-to-end test with actual pip installation.

  This test is expensive and only runs when explicitly requested
  via tox -e plugin-e2e or in CI when provider files change.
  """

  def test_plugin_with_schema_e2e(self):
    """Test that a plugin with custom schema works end-to-end with extract()."""

    class TestPluginSchema(lx.schema.BaseSchema):
      """Test schema implementation."""

      def __init__(self, config):
        self._config = config

      @classmethod
      def from_examples(cls, examples_data, attribute_suffix="_attributes"):
        return cls({"generated": True, "count": len(examples_data)})

      def to_provider_config(self):
        return {"custom_schema": self._config}

      @property
      def requires_raw_output(self):
        return True

    def _ep_load():
      @lx.providers.registry.register(r"^e2e-schema-test")
      class SchemaE2EProvider(base_model.BaseLanguageModel):

        def __init__(self, model_id=None, **kwargs):
          super().__init__()
          self.model_id = model_id
          self.schema_config = kwargs.get("custom_schema")

        @classmethod
        def get_schema_class(cls):
          return TestPluginSchema

        def infer(self, batch_prompts, **kwargs):
          # Return a mock extraction that includes schema info
          if self.schema_config:
            output = (
                '{"extractions": [{"entity": "test", '
                '"entity_attributes": {"schema": "applied"}}]}'
            )
          else:
            output = '{"extractions": []}'
          return [[types.ScoredOutput(score=1.0, output=output)]]

      return SchemaE2EProvider

    ep = builtin_types.SimpleNamespace(
        name="schema_e2e",
        group="langextract.providers",
        value="test:SchemaE2EProvider",
        load=_ep_load,
    )

    # Clear and set up registry
    lx.providers.registry.clear()
    lx.providers._PLUGINS_LOADED = False
    self.addCleanup(lx.providers.registry.clear)
    self.addCleanup(setattr, lx.providers, "_PLUGINS_LOADED", False)

    with mock.patch.object(
        metadata, "entry_points", return_value=_create_mock_entry_points([ep])
    ):
      lx.providers.load_plugins_once()

      # Test with extract() using schema constraints
      examples = [
          lx.data.ExampleData(
              text="Find entities",
              extractions=[
                  lx.data.Extraction(
                      extraction_class="entity",
                      extraction_text="example",
                      attributes={"type": "test"},
                  )
              ],
          )
      ]

      result = lx.extract(
          text_or_documents="Test text for extraction",
          prompt_description="Extract entities",
          examples=examples,
          model_id="e2e-schema-test-v1",
          use_schema_constraints=True,
          fence_output=False,  # Schema supports strict mode
      )

      # Verify we got results
      self.assertIsInstance(result, lx.data.AnnotatedDocument)
      self.assertIsNotNone(result.extractions)
      self.assertGreater(len(result.extractions), 0)

      # Verify the schema was applied by checking the extraction
      extraction = result.extractions[0]
      self.assertEqual(extraction.extraction_class, "entity")
      self.assertIn("schema", extraction.attributes)
      self.assertEqual(extraction.attributes["schema"], "applied")

  @pytest.mark.requires_pip
  @pytest.mark.integration
  def test_pip_install_discovery_and_cleanup(self):
    """Test complete plugin lifecycle: install, discovery, usage, uninstall.

    This test:
    1. Creates a Python package with a provider plugin
    2. Installs it via pip
    3. Verifies the plugin is discovered and usable
    4. Uninstalls and verifies cleanup
    """

    # Skip in Bazel environment where pip operations don't work
    if os.environ.get("TEST_TMPDIR") or os.environ.get(
        "BUILD_WORKING_DIRECTORY"
    ):
      self.skipTest("pip install tests don't work in Bazel sandbox")

    # Also skip if pip is not available
    try:
      subprocess.run(
          [sys.executable, "-m", "pip", "--version"],
          capture_output=True,
          check=True,
      )
    except (subprocess.CalledProcessError, FileNotFoundError):
      self.skipTest("pip not available in test environment")

    with tempfile.TemporaryDirectory() as tmpdir:
      pkg_name = f"test_langextract_plugin_{uuid.uuid4().hex[:8]}"
      pkg_dir = Path(tmpdir) / pkg_name
      pkg_dir.mkdir()

      (pkg_dir / pkg_name).mkdir()
      (pkg_dir / pkg_name / "__init__.py").write_text("")

      (pkg_dir / pkg_name / "provider.py").write_text(textwrap.dedent("""
        import langextract as lx
        from langextract.core import base_model
        from langextract.core import types

        USED_BY_EXTRACT = False

        class TestPipSchema(lx.schema.BaseSchema):
            '''Test schema for pip provider.'''

            def __init__(self, config):
                self._config = config

            @classmethod
            def from_examples(cls, examples_data, attribute_suffix="_attributes"):
                return cls({"pip_schema": True, "examples": len(examples_data)})

            def to_provider_config(self):
                return {"schema_config": self._config}

            @property
            def requires_raw_output(self):
                return True

        @lx.providers.registry.register(r'^test-pip-model', priority=50)
        class TestPipProvider(base_model.BaseLanguageModel):
            def __init__(self, model_id, **kwargs):
                super().__init__()
                self.model_id = model_id
                self.schema_config = kwargs.get("schema_config", {})

            @classmethod
            def get_schema_class(cls):
                return TestPipSchema

            def infer(self, batch_prompts, **kwargs):
                global USED_BY_EXTRACT
                USED_BY_EXTRACT = True
                schema_info = "with_schema" if self.schema_config else "no_schema"
                return [[types.ScoredOutput(score=1.0, output=f"pip test response: {schema_info}")]]
      """))

      (pkg_dir / "pyproject.toml").write_text(textwrap.dedent(f"""
        [build-system]
        requires = ["setuptools>=61.0"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "{pkg_name}"
        version = "0.0.1"
        description = "Test plugin for langextract"

        [project.entry-points."langextract.providers"]
        test_provider = "{pkg_name}.provider:TestPipProvider"
      """))

      pip_env = {
          **os.environ,
          "PIP_NO_INPUT": "1",
          "PIP_DISABLE_PIP_VERSION_CHECK": "1",
      }
      result = subprocess.run(
          [
              sys.executable,
              "-m",
              "pip",
              "install",
              "-e",
              str(pkg_dir),
              "--no-deps",
              "-q",
          ],
          check=True,
          capture_output=True,
          text=True,
          env=pip_env,
      )

      self.assertEqual(result.returncode, 0, "pip install failed")
      self.assertNotIn(
          "ERROR",
          result.stderr.upper(),
          f"pip install had errors: {result.stderr}",
      )

      try:
        test_script = Path(tmpdir) / "test_plugin.py"
        test_script.write_text(textwrap.dedent(f"""
          import langextract as lx
          import sys

          lx.providers.load_plugins_once()

          # Test 1: Basic usage without schema
          cfg = lx.factory.ModelConfig(model_id="test-pip-model-123")
          model = lx.factory.create_model(cfg)
          result = model.infer(["test prompt"])
          assert "no_schema" in result[0][0].output, f"Got: {{result[0][0].output}}"

          # Test 2: With schema constraints
          examples = [
              lx.data.ExampleData(
                  text="test",
                  extractions=[
                      lx.data.Extraction(
                          extraction_class="test",
                          extraction_text="test",
                      )
                  ],
              )
          ]

          cfg2 = lx.factory.ModelConfig(model_id="test-pip-model-456")
          model2 = lx.factory._create_model_with_schema(
              config=cfg2,
              examples=examples,
              use_schema_constraints=True,
              fence_output=None,
          )
          result2 = model2.infer(["test prompt"])
          assert "with_schema" in result2[0][0].output, f"Got: {{result2[0][0].output}}"
          assert model2.requires_fence_output == False, "Schema outputs raw JSON, should not need fences"

          # Test 3: Verify schema class is available
          provider_cls = lx.providers.registry.resolve("test-pip-model-xyz")
          assert provider_cls.__name__ == "TestPipProvider", "Plugin should be resolvable"
          schema_cls = provider_cls.get_schema_class()
          assert schema_cls.__name__ == "TestPipSchema", f"Schema class should be TestPipSchema, got {{schema_cls.__name__}}"

          from {pkg_name}.provider import USED_BY_EXTRACT
          assert USED_BY_EXTRACT, "Provider infer() was not called"

          print("SUCCESS: Plugin test with schema passed")
        """))

        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertIn(
            "SUCCESS",
            result.stdout,
            f"Test failed. stdout: {result.stdout}, stderr: {result.stderr}",
        )

      finally:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg_name],
            check=False,
            capture_output=True,
            env=pip_env,
        )

        lx.providers.registry.clear()
        lx.providers._PLUGINS_LOADED = False
        lx.providers.load_plugins_once()

        with self.assertRaisesRegex(
            lx.exceptions.InferenceConfigError,
            "No provider registered for model_id='test-pip-model",
        ):
          lx.providers.registry.resolve("test-pip-model-789")


if __name__ == "__main__":
  absltest.main()
