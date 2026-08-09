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

"""Regression tests for the provider plugin generator."""

import ast
import contextlib
import importlib.metadata
import importlib.util
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from unittest import mock
import zipfile

try:
  import tomllib
except ModuleNotFoundError:  # Python < 3.11
  import tomli as tomllib

from absl.testing import absltest
from absl.testing import parameterized
import pytest

from langextract import data
import langextract as lx
from langextract.core import schema as core_schema

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts"
    / "create_provider_plugin.py"
)
_REPO_ROOT = _SCRIPT_PATH.parent.parent


def _load_generator_module():
  """Loads scripts/create_provider_plugin.py as a module."""
  spec = importlib.util.spec_from_file_location(
      "create_provider_plugin", _SCRIPT_PATH
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class _CreateProviderPluginTestBase(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._generator = _load_generator_module()

  def _make_tempdir(self):
    """Returns a fresh temp dir cleaned up at test end (pytest-safe)."""
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
    return tmp_dir

  def _generate_full_plugin(
      self, base_dir, provider_name, package_name, patterns, with_schema
  ):
    """Generates a complete plugin into base_dir (no install)."""
    base_dir = pathlib.Path(base_dir)
    (base_dir / f"langextract_{package_name}").mkdir(parents=True)
    with contextlib.redirect_stdout(io.StringIO()):
      self._generator.create_pyproject_toml(
          base_dir, provider_name, package_name
      )
      self._generator.create_provider(
          base_dir=base_dir,
          provider_name=provider_name,
          package_name=package_name,
          patterns=patterns,
          with_schema=with_schema,
      )
      if with_schema:
        self._generator.create_schema(base_dir, provider_name, package_name)
      self._generator.create_test_script(
          base_dir=base_dir,
          provider_name=provider_name,
          package_name=package_name,
          patterns=patterns,
          with_schema=with_schema,
      )
    return base_dir

  def _generate_provider_source(self, provider_name, package_name, with_schema):
    """Runs create_provider in a temp dir and returns provider.py text."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      base_dir = pathlib.Path(tmp_dir)
      (base_dir / f"langextract_{package_name}").mkdir()
      with contextlib.redirect_stdout(io.StringIO()):
        self._generator.create_provider(
            base_dir=base_dir,
            provider_name=provider_name,
            package_name=package_name,
            patterns=[f"^{package_name}"],
            with_schema=with_schema,
        )
      provider_path = base_dir / f"langextract_{package_name}" / "provider.py"
      return provider_path.read_text(encoding="utf-8")

  def _generate_readme_source(self, provider_name, package_name, patterns):
    """Runs create_readme in a temp dir and returns README.md text."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      base_dir = pathlib.Path(tmp_dir)
      with contextlib.redirect_stdout(io.StringIO()):
        self._generator.create_readme(
            base_dir=base_dir,
            provider_name=provider_name,
            package_name=package_name,
            patterns=patterns,
        )
      return (base_dir / "README.md").read_text(encoding="utf-8")

  def _import_generated_package(self, base_dir, package_name):
    """Puts base_dir on sys.path and imports the generated package.

    Importing a generated provider runs its registry.register decorator,
    so the process-global provider registry is cleared before the import
    and again on cleanup (mirroring provider_plugin_test.py) — the
    registration must not leak into later tests.
    """
    lx.providers.registry.clear()
    lx.providers._reset_for_testing()  # pylint: disable=protected-access
    self.addCleanup(lx.providers.registry.clear)
    self.addCleanup(
        lx.providers._reset_for_testing  # pylint: disable=protected-access
    )
    base_dir = str(base_dir)
    module_prefix = f"langextract_{package_name}"
    sys.path.insert(0, base_dir)
    self.addCleanup(sys.path.remove, base_dir)

    def _purge_modules():
      for name in list(sys.modules):
        if name == module_prefix or name.startswith(module_prefix + "."):
          del sys.modules[name]

    _purge_modules()
    self.addCleanup(_purge_modules)
    return importlib.import_module(f"{module_prefix}.provider")

  def _run_generated_test_script(self, base_dir, extra_path=None, cwd=None):
    """Runs the generated test_plugin.py from a neutral cwd."""
    return subprocess.run(
        [sys.executable, str(pathlib.Path(base_dir) / "test_plugin.py")],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd or self._make_tempdir(),
        env=self._subprocess_env(extra_path),
    )

  def _make_fake_install(self, base_dir, package_name, entry_point_value):
    """Simulates an installed plugin: package copy + dist-info metadata."""
    site_dir = pathlib.Path(self._make_tempdir())
    module_name = f"langextract_{package_name}"
    shutil.copytree(
        pathlib.Path(base_dir) / module_name, site_dir / module_name
    )
    dist_info = site_dir / f"{module_name}-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: langextract-{package_name}\n"
        "Version: 0.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        f"[langextract.providers]\n{package_name} = {entry_point_value}\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    return site_dir

  def _make_dir_alias(self, target):
    """Creates a symlink alias of target, skipping if unsupported."""
    alias = pathlib.Path(self._make_tempdir()) / "alias"
    try:
      alias.symlink_to(pathlib.Path(target), target_is_directory=True)
    except OSError as e:
      self.skipTest(f"symlinks unavailable on this platform: {e}")
    return alias

  def _mutate_installed_provider(self, site_dir, package_name, old, new):
    """Rewrites the fake-installed provider.py, asserting the target exists."""
    provider_path = (
        pathlib.Path(site_dir) / f"langextract_{package_name}" / "provider.py"
    )
    source = provider_path.read_text(encoding="utf-8")
    self.assertEqual(source.count(old), 1, msg=f"mutation target: {old!r}")
    provider_path.write_text(source.replace(old, new), encoding="utf-8")

  def _subprocess_env(self, extra_path=None):
    """Builds an env whose PYTHONPATH covers extra_path plus the repo root."""
    env = dict(os.environ)
    path_entries = [str(p) for p in (extra_path or [])] + [str(_REPO_ROOT)]
    if env.get("PYTHONPATH"):
      path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    return env

  def _run_generator_cli(self, work_dir, *cli_args):
    """Runs the generator CLI in work_dir and asserts it succeeded."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *cli_args, "--no-install"],
        capture_output=True,
        text=True,
        check=False,
        cwd=work_dir,
    )
    self.assertEqual(
        result.returncode,
        0,
        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    return result

  def _readme_usage_model_id(self, readme_source):
    """Returns the model_id literal from the README usage snippet, if any."""
    if "```python\n" not in readme_source:
      return None
    snippet = readme_source.split("```python\n")[1].split("```")[0]
    for node in ast.walk(ast.parse(snippet)):
      if (
          isinstance(node, ast.keyword)
          and node.arg == "model_id"
          and isinstance(node.value, ast.Constant)
          and isinstance(node.value.value, str)
      ):
        return node.value.value
    return None

  def _run_install_with_hermetic_backend(
      self,
      env,
      present_specs=(),
      broken_specs=(),
      setuptools_version=None,
      pip_result=None,
      configured_no_index=False,
      langextract_available=True,
  ):
    """Runs install_and_test with a fully mocked backend-probe surface.

    Both importlib.util.find_spec and the setuptools version probe are
    replaced — nothing is inherited from the interpreter running the
    tests, so each test constructs its own premise.

    Args:
      env: The complete environment for the call (os.environ is replaced).
      present_specs: Module names that resolve to a (fake) spec; names not
        listed anywhere resolve to None.
      broken_specs: Module names whose probe raises, simulating broken
        import machinery.
      setuptools_version: Version string the metadata probe reports; None
        raises PackageNotFoundError; an Exception instance is raised as-is.
      pip_result: Result mock every subprocess.run call returns; defaults
        to a clean success with empty streams.
      configured_no_index: Whether the isolated pip config enables no-index.
      langextract_available: Whether the LangExtract runtime is importable.

    Returns:
      A (success, mock_run, output) tuple.
    """
    base_dir = pathlib.Path(self._make_tempdir())
    fake_result = pip_result or mock.Mock(returncode=0, stdout="", stderr="")
    present = set(present_specs)
    broken = set(broken_specs)
    simulated_names = {
        "langextract",
        "setuptools",
        "setuptools.build_meta",
        *present,
        *broken,
    }
    real_find_spec = self._generator.importlib.util.find_spec

    def _fake_find_spec(name, *args, **kwargs):
      if name in broken:
        raise RuntimeError(f"simulated broken import machinery: {name}")
      if name in simulated_names:
        if name == "langextract":
          return (
              mock.Mock(name="spec:langextract")
              if langextract_available
              else None
          )
        return mock.Mock(name=f"spec:{name}") if name in present else None
      return real_find_spec(name, *args, **kwargs)

    def _fake_metadata_version(distribution_name):
      if setuptools_version is None:
        raise importlib.metadata.PackageNotFoundError(distribution_name)
      if isinstance(setuptools_version, Exception):
        raise setuptools_version
      return setuptools_version

    with mock.patch.object(
        self._generator,
        "_pip_config_no_index_enabled",
        autospec=True,
        return_value=configured_no_index,
    ):
      with mock.patch.object(
          self._generator.importlib.util,
          "find_spec",
          autospec=True,
          side_effect=_fake_find_spec,
      ):
        self.assertIsNotNone(
            self._generator.importlib.util.find_spec("json"),
            msg="unrelated module probes must use the real import machinery",
        )
        with mock.patch.object(
            self._generator.importlib.metadata,
            "version",
            autospec=True,
            side_effect=_fake_metadata_version,
        ):
          with mock.patch.object(
              self._generator.subprocess,
              "run",
              autospec=True,
              return_value=fake_result,
          ) as mock_run:
            with mock.patch.dict(os.environ, env, clear=True):
              with contextlib.redirect_stdout(io.StringIO()) as out:
                success = self._generator.install_and_test(base_dir)
    return success, mock_run, out.getvalue()

  _USABLE_BACKEND_SPECS = ("setuptools", "setuptools.build_meta")

  def _assert_offline_safe_install(self, success, mock_run, output):
    """Asserts the local no-build-isolation editable-install path ran."""
    self.assertTrue(
        success,
        msg=f"offline-capable install must succeed; output:\n{output}",
    )
    install_cmd = mock_run.call_args_list[0].args[0]
    self.assertIn(
        "--no-build-isolation",
        install_cmd,
        msg=(
            "with a usable local backend the offline-safe"
            " no-build-isolation path must be taken"
        ),
    )
    self.assertIn("--no-deps", install_cmd)

  def _assert_fails_before_pip_with_upgrade_hint(
      self, success, mock_run, output
  ):
    """Asserts the early actionable failure path ran instead of pip."""
    self.assertFalse(
        success,
        msg=(
            "an unusable local backend in no-index mode cannot build —"
            " install_and_test must report failure before invoking pip;"
            f" output:\n{output}"
        ),
    )
    mock_run.assert_not_called()
    self.assertIn(
        "pip install 'setuptools>=77.0'",
        output,
        msg=(
            "the prerequisite message must carry the shell-safe exact"
            " upgrade command (quoted so >= survives the shell)"
        ),
    )

  def _assert_online_isolated_install(self, success, mock_run, output):
    """Asserts pip build isolation ran and the test step still followed."""
    self.assertTrue(success, msg=f"output:\n{output}")
    self.assertLen(
        mock_run.call_args_list,
        2,
        msg=(
            "install_and_test must run the pip install and then proceed"
            " to the generated test step"
        ),
    )
    install_cmd = mock_run.call_args_list[0].args[0]
    self.assertNotIn(
        "--no-build-isolation",
        install_cmd,
        msg=(
            "without a usable local backend the editable install must let"
            " pip provision the declared build requirements via isolation"
        ),
    )
    self.assertIn(
        "--no-deps",
        install_cmd,
        msg="runtime deps still come from the dev environment",
    )
    test_cmd = mock_run.call_args_list[1].args[0]
    self.assertEqual(
        test_cmd[-1],
        "test_plugin.py",
        msg="the generated test step must still run after the install",
    )


class GeneratedPluginTest(_CreateProviderPluginTestBase):

  def test_provider_without_schema_is_valid_python(self):
    source = self._generate_provider_source(
        "FocusedPlain", "focusedplain", with_schema=False
    )

    compile(source, "provider.py", "exec")
    self.assertNotIn(".schema import", source)
    self.assertStartsWith(
        source,
        '"""Provider implementation for FocusedPlain."""',
        msg=f"module docstring must start at column 0: {source[:60]!r}",
    )

  def test_provider_with_schema_is_valid_python(self):
    source = self._generate_provider_source(
        "FocusedSchema", "focusedschema", with_schema=True
    )

    compile(source, "provider.py", "exec")
    self.assertIn(
        "\nfrom langextract_focusedschema.schema import FocusedSchemaSchema\n",
        source,
        msg="schema import must land at column 0 after dedent",
    )
    self.assertStartsWith(
        source,
        '"""Provider implementation for FocusedSchema."""',
        msg=f"module docstring must start at column 0: {source[:60]!r}",
    )

  def test_readme_heading_is_at_column_zero(self):
    source = self._generate_readme_source(
        "FocusedReadme", "focusedreadme", ["^focusedreadme", "^alternate"]
    )

    self.assertStartsWith(source, "# LangExtract FocusedReadme Provider\n")
    self.assertIn(
        "\n- `focusedreadme*`: Models matching pattern ^focusedreadme\n"
        "- `alternate*`: Models matching pattern ^alternate\n",
        source,
    )

  def test_generated_schema_supports_current_base_schema_lifecycle(self):
    """Generated schema must instantiate and apply against BaseSchema."""
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "RtLifecycle",
        "rtlifecycle",
        ["^rtlifecycle"],
        with_schema=True,
    )

    provider_module = self._import_generated_package(base_dir, "rtlifecycle")
    schema_class = provider_module.RtLifecycleLanguageModel.get_schema_class()
    self.assertEqual(schema_class.__name__, "RtLifecycleSchema")

    examples = [
        data.ExampleData(
            text="Test text",
            extractions=[
                data.Extraction(
                    extraction_class="entity",
                    extraction_text="test",
                    attributes={"type": "example"},
                )
            ],
        )
    ]
    schema_instance = schema_class.from_examples(examples)
    self.assertIsInstance(schema_instance, core_schema.BaseSchema)
    self.assertFalse(schema_instance.requires_raw_output)
    provider_config = schema_instance.to_provider_config()
    self.assertIn("response_schema", provider_config)
    self.assertTrue(provider_config["structured_output"])

    provider = provider_module.RtLifecycleLanguageModel(
        model_id="rtlifecycle-test"
    )
    provider.apply_schema(schema_instance)
    self.assertEqual(provider.response_schema, schema_instance.schema_dict)
    self.assertTrue(provider.structured_output)
    provider.apply_schema(None)
    self.assertIsNone(provider.response_schema)
    self.assertFalse(provider.structured_output)

    results = list(provider.infer(["hello"]))
    self.assertLen(results, 1)
    self.assertStartsWith(results[0][0].output, "Mock response for: hello")

  @parameterized.named_parameters(
      dict(testcase_name="without_schema", with_schema=False),
      dict(testcase_name="with_schema", with_schema=True),
  )
  def test_generated_python_is_formatted(self, with_schema):
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "Formatted",
        "formatted",
        ["^formatted", "^alternate"],
        with_schema=with_schema,
    )
    generated_files = [
        base_dir / "langextract_formatted" / "__init__.py",
        base_dir / "langextract_formatted" / "provider.py",
        base_dir / "test_plugin.py",
    ]
    if with_schema:
      generated_files.append(base_dir / "langextract_formatted" / "schema.py")

    pyink_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyink",
            "--check",
            "--config",
            str(_REPO_ROOT / "pyproject.toml"),
            *map(str, generated_files),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        pyink_result.returncode,
        0,
        msg=f"stdout:\n{pyink_result.stdout}\nstderr:\n{pyink_result.stderr}",
    )

    isort_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isort",
            "--check-only",
            "--settings-path",
            str(_REPO_ROOT / "pyproject.toml"),
            *map(str, generated_files),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(
        isort_result.returncode,
        0,
        msg=f"stdout:\n{isort_result.stdout}\nstderr:\n{isort_result.stderr}",
    )

  def test_generated_test_script_passes_against_installed_plugin(self):
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "EpGreen",
        "epgreen",
        ["^epgreen"],
        with_schema=True,
    )
    site_dir = self._make_fake_install(
        base_dir, "epgreen", "langextract_epgreen.provider:EpGreenLanguageModel"
    )

    result = self._run_generated_test_script(base_dir, extra_path=[site_dir])

    self.assertEqual(
        result.returncode,
        0,
        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    self.assertIn("✅ All checks passed!", result.stdout)

  def test_generated_test_script_exits_nonzero_on_failing_check(self):
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "EpBroken",
        "epbroken",
        ["^epbroken"],
        with_schema=True,
    )
    site_dir = self._make_fake_install(
        base_dir, "epbroken", "langextract_epbroken.provider:WrongClass"
    )

    result = self._run_generated_test_script(base_dir, extra_path=[site_dir])

    self.assertNotEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}")
    self.assertIn("check(s) failed", result.stdout)
    self.assertNotIn("✅ All checks passed!", result.stdout)

  def test_generated_test_script_fails_when_plugin_not_installed(self):
    """The source tree next to test_plugin.py must not mask a bad install."""
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "EpAbsent",
        "epabsent",
        ["^epabsent"],
        with_schema=False,
    )

    result = self._run_generated_test_script(base_dir)

    self.assertEqual(result.returncode, 1, msg=f"stdout:\n{result.stdout}")
    self.assertIn("Plugin not installed", result.stdout)

  def test_generated_test_script_distinguishes_broken_installed_plugin(self):
    """An installed plugin whose import breaks is not 'not installed'.

    A missing SDK dependency inside an installed plugin raises
    ImportError too — the script must say the plugin import failed and
    surface the real exception, still exiting nonzero.
    """
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "BrokenDep",
        "brokendep",
        ["^brokendep"],
        with_schema=False,
    )
    site_dir = self._make_fake_install(
        base_dir,
        "brokendep",
        "langextract_brokendep.provider:BrokenDepLanguageModel",
    )
    self._mutate_installed_provider(
        site_dir,
        "brokendep",
        "import os",
        "import os\nimport missing_sdk_dependency_xyz",
    )

    result = self._run_generated_test_script(base_dir, extra_path=[site_dir])

    self.assertNotEqual(result.returncode, 0)
    self.assertNotIn(
        "Plugin not installed",
        result.stdout,
        msg=(
            "an installed-but-broken plugin must not be misdiagnosed as"
            f" absent; stdout:\n{result.stdout}"
        ),
    )
    self.assertIn(
        "failed to import",
        result.stdout,
        msg="the script must say the plugin import itself failed",
    )
    self.assertIn(
        "missing_sdk_dependency_xyz",
        result.stdout,
        msg="the real underlying exception must be surfaced",
    )

  def test_generated_import_helper_isolates_global_registry(self):
    """The in-process import's registration must not outlive the test."""
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "RegIso",
        "regiso",
        ["^regiso"],
        with_schema=False,
    )

    provider_module = self._import_generated_package(base_dir, "regiso")
    self.assertIs(
        lx.providers.registry.resolve("regiso-model"),
        provider_module.RegIsoLanguageModel,
    )

    self.doCleanups()

    with self.assertRaisesRegex(
        lx.exceptions.InferenceConfigError,
        "No provider registered",
        msg=(
            "the generated provider registration must not survive the"
            " helper's cleanup — the global registry stays isolated"
        ),
    ):
      lx.providers.registry.resolve("regiso-model")

  def test_uninstalled_plugin_fails_even_through_symlinked_script_dir(self):
    """A symlink alias of the script dir must not defeat sys.path filtering.

    Mirrors path-alias layouts (such as /var vs /private/var temp dirs)
    where the script directory and a sys.path entry are different spellings
    of the same directory: whichever spelling is imported through, the
    adjacent uninstalled source must not satisfy the import.
    """
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "AliasGd",
        "aliasgd",
        ["^aliasgd"],
        with_schema=False,
    )
    alias = self._make_dir_alias(base_dir)

    # Direction 1: script invoked via its real path, alias on sys.path.
    via_real = self._run_generated_test_script(base_dir, extra_path=[alias])
    # Direction 2: script invoked via the alias, real dir on sys.path.
    via_alias = self._run_generated_test_script(alias, extra_path=[base_dir])

    for label, result in (("real", via_real), ("alias", via_alias)):
      self.assertEqual(
          result.returncode,
          1,
          msg=f"[{label}] stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
      )
      self.assertIn("Plugin not installed", result.stdout, msg=f"[{label}]")

  def test_generated_test_script_fails_on_result_count_mismatch(self):
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "CntChk",
        "cntchk",
        ["^cntchk"],
        with_schema=False,
    )
    site_dir = self._make_fake_install(
        base_dir, "cntchk", "langextract_cntchk.provider:CntChkLanguageModel"
    )
    self._mutate_installed_provider(
        site_dir,
        "cntchk",
        "for prompt in batch_prompts:",
        "for prompt in batch_prompts[:1]:",
    )

    result = self._run_generated_test_script(base_dir, extra_path=[site_dir])

    self.assertNotEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}")
    self.assertIn("Expected 2 results, got 1", result.stdout)
    self.assertNotIn("✅ All checks passed!", result.stdout)

  def test_generated_test_script_fails_on_empty_inference_output(self):
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "OutChk",
        "outchk",
        ["^outchk"],
        with_schema=False,
    )
    site_dir = self._make_fake_install(
        base_dir, "outchk", "langextract_outchk.provider:OutChkLanguageModel"
    )
    self._mutate_installed_provider(
        site_dir,
        "outchk",
        'result = f"Mock response for: {prompt[:50]}..."',
        'result = ""',
    )

    result = self._run_generated_test_script(base_dir, extra_path=[site_dir])

    self.assertNotEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}")
    self.assertIn("missing or empty output", result.stdout)
    self.assertNotIn("✅ All checks passed!", result.stdout)

  def test_readme_usage_snippet_is_valid_python_for_quote_patterns(self):
    source = self._generate_readme_source(
        "QuoteDoc", "quotedoc", ['^x"y', "^a'b", '^t"""q']
    )

    snippet = source.split("```python\n")[1].split("```")[0]
    compile(snippet, "readme_usage.py", "exec")
    self.assertIn("model_id=" + repr('x"y-model'), snippet)


class GeneratorCliTest(_CreateProviderPluginTestBase):

  @parameterized.named_parameters(
      dict(testcase_name="space_in_name", cli_args=["My Provider"]),
      dict(testcase_name="leading_digit", cli_args=["9Lives"]),
      dict(testcase_name="quote_in_name", cli_args=['Bad"Quote']),
      dict(testcase_name="empty_name", cli_args=[""]),
      dict(
          testcase_name="hyphen_package",
          cli_args=["Good", "--package-name", "my-prov"],
      ),
      dict(
          testcase_name="digit_package",
          cli_args=["Good", "--package-name", "9pkg"],
      ),
  )
  def test_cli_rejects_invalid_names_without_writing(self, cli_args):
    work_dir = self._make_tempdir()

    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *cli_args, "--no-install"],
        capture_output=True,
        text=True,
        check=False,
        cwd=work_dir,
    )

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("ERROR", result.stdout + result.stderr)
    self.assertEmpty(
        os.listdir(work_dir), msg="rejected input must not write files"
    )

  @parameterized.named_parameters(
      dict(testcase_name="normal", extra_args=()),
      dict(testcase_name="force", extra_args=("--force",)),
  )
  def test_cli_fails_cleanly_when_target_is_plain_file(self, extra_args):
    """A plain file named langextract-<pkg> must fail cleanly, untouched."""
    work_dir = self._make_tempdir()
    collision = pathlib.Path(work_dir) / "langextract-filecol"
    collision.write_text("precious bytes", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "FileCol",
            *extra_args,
            "--no-install",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=work_dir,
    )

    self.assertNotEqual(result.returncode, 0)
    combined = result.stdout + result.stderr
    self.assertIn(
        "is not a directory",
        combined,
        msg=f"need a concise actionable message, got:\n{combined}",
    )
    self.assertNotIn(
        "Traceback",
        combined,
        msg="the collision must be diagnosed, not dumped as a traceback",
    )
    self.assertEqual(
        collision.read_text(encoding="utf-8"),
        "precious bytes",
        msg="the colliding file must never be deleted or overwritten",
    )
    self.assertEqual(
        os.listdir(work_dir),
        ["langextract-filecol"],
        msg="nothing else may be written after the failed pre-check",
    )

  @parameterized.named_parameters(
      dict(testcase_name="normal", extra_args=()),
      dict(testcase_name="force", extra_args=("--force",)),
  )
  def test_cli_fails_cleanly_on_dangling_symlink_target(self, extra_args):
    """A dangling symlink named langextract-<pkg> must fail cleanly."""
    work_dir = self._make_tempdir()
    link = pathlib.Path(work_dir) / "langextract-symcol"
    try:
      link.symlink_to("missing-target-dir")
    except OSError as e:
      self.skipTest(f"symlinks unavailable on this platform: {e}")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "SymCol",
            *extra_args,
            "--no-install",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=work_dir,
    )

    self.assertNotEqual(result.returncode, 0)
    combined = result.stdout + result.stderr
    self.assertIn(
        "symlink",
        combined,
        msg=f"need a concise actionable message, got:\n{combined}",
    )
    self.assertNotIn(
        "Traceback",
        combined,
        msg="the symlink must be diagnosed, not dumped as a traceback",
    )
    self.assertTrue(
        os.path.lexists(link),
        msg="the symlink itself must be preserved",
    )
    self.assertEqual(
        os.readlink(link),
        "missing-target-dir",
        msg="the link target must not be rewritten",
    )
    self.assertEqual(
        os.listdir(work_dir),
        ["langextract-symcol"],
        msg="nothing may be written after the failed pre-check",
    )

  def test_cli_force_refuses_resolvable_outer_symlink(self):
    """--force must not write through a plugin-directory symlink."""
    work_dir = self._make_tempdir()
    target_dir = pathlib.Path(self._make_tempdir())
    sentinel = target_dir / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    link = pathlib.Path(work_dir) / "langextract-outerlink"
    try:
      link.symlink_to(target_dir, target_is_directory=True)
    except OSError as e:
      self.skipTest(f"symlinks unavailable on this platform: {e}")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "OuterLink",
            "--force",
            "--no-install",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=work_dir,
    )

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("symlink", result.stdout + result.stderr)
    self.assertTrue(link.is_symlink())
    self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")
    self.assertEqual(os.listdir(target_dir), ["sentinel.txt"])

  @parameterized.named_parameters(
      dict(testcase_name="dangling", link_target="missing-inner-target"),
      dict(testcase_name="self_loop", link_target="langextract_symincol"),
  )
  def test_cli_force_fails_cleanly_on_inner_symlink_collision(
      self, link_target
  ):
    """--force onto a dir whose inner package path is a broken symlink.

    The inner-path analogue of the outer dangling-symlink collision: a
    dangling (or self-looping) langextract_<pkg> symlink inside an
    existing base directory must be refused before any write, with the
    link and its target left untouched.
    """
    work_dir = self._make_tempdir()
    base_dir = pathlib.Path(work_dir) / "langextract-symincol"
    base_dir.mkdir()
    inner_link = base_dir / "langextract_symincol"
    try:
      inner_link.symlink_to(link_target)
    except OSError as e:
      self.skipTest(f"symlinks unavailable on this platform: {e}")
    sentinel = base_dir / "pyproject.toml"
    sentinel.write_text("# original untouched", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "SymInCol",
            "--force",
            "--no-install",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=work_dir,
    )

    self.assertNotEqual(result.returncode, 0)
    combined = result.stdout + result.stderr
    self.assertIn(
        "symlink",
        combined,
        msg=f"need a concise actionable message, got:\n{combined}",
    )
    self.assertNotIn(
        "Traceback",
        combined,
        msg="the collision must be diagnosed, not dumped as a traceback",
    )
    self.assertTrue(
        os.path.lexists(inner_link),
        msg="the inner symlink itself must be preserved",
    )
    self.assertEqual(
        os.readlink(inner_link),
        link_target,
        msg="the link target must not be rewritten",
    )
    self.assertEqual(
        sentinel.read_text(encoding="utf-8"),
        "# original untouched",
        msg="no partial generator rewrites may happen before the check",
    )
    self.assertCountEqual(
        os.listdir(base_dir),
        ["langextract_symincol", "pyproject.toml"],
        msg="nothing may be added to the existing directory",
    )

  def test_cli_force_refuses_resolvable_inner_symlink(self):
    """--force must not write through a generated-package symlink."""
    work_dir = self._make_tempdir()
    base_dir = pathlib.Path(work_dir) / "langextract-innerlink"
    base_dir.mkdir()
    base_sentinel = base_dir / "pyproject.toml"
    base_sentinel.write_text("# original", encoding="utf-8")
    target_dir = pathlib.Path(self._make_tempdir())
    target_sentinel = target_dir / "sentinel.txt"
    target_sentinel.write_text("untouched", encoding="utf-8")
    inner_link = base_dir / "langextract_innerlink"
    try:
      inner_link.symlink_to(target_dir, target_is_directory=True)
    except OSError as e:
      self.skipTest(f"symlinks unavailable on this platform: {e}")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "InnerLink",
            "--force",
            "--no-install",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=work_dir,
    )

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("symlink", result.stdout + result.stderr)
    self.assertTrue(inner_link.is_symlink())
    self.assertEqual(base_sentinel.read_text(encoding="utf-8"), "# original")
    self.assertEqual(target_sentinel.read_text(encoding="utf-8"), "untouched")
    self.assertEqual(os.listdir(target_dir), ["sentinel.txt"])

  def test_cli_force_fails_cleanly_when_inner_package_path_is_file(self):
    """--force onto a dir whose inner package path is a plain file."""
    work_dir = self._make_tempdir()
    base_dir = pathlib.Path(work_dir) / "langextract-innercol"
    base_dir.mkdir()
    inner = base_dir / "langextract_innercol"
    inner.write_text("precious inner bytes", encoding="utf-8")
    sentinel = base_dir / "pyproject.toml"
    sentinel.write_text("# original untouched", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "InnerCol",
            "--force",
            "--no-install",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=work_dir,
    )

    self.assertNotEqual(result.returncode, 0)
    combined = result.stdout + result.stderr
    self.assertIn(
        "is not a directory",
        combined,
        msg=f"need a concise actionable message, got:\n{combined}",
    )
    self.assertNotIn(
        "Traceback",
        combined,
        msg="the collision must be diagnosed, not dumped as a traceback",
    )
    self.assertEqual(
        inner.read_text(encoding="utf-8"),
        "precious inner bytes",
        msg="the inner file must never be deleted or overwritten",
    )
    self.assertEqual(
        sentinel.read_text(encoding="utf-8"),
        "# original untouched",
        msg="no partial generator rewrites may happen before the check",
    )
    self.assertCountEqual(
        os.listdir(base_dir),
        ["langextract_innercol", "pyproject.toml"],
        msg="nothing may be added to the existing directory",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="pyproject",
          relative_parts=("pyproject.toml",),
          with_schema=False,
      ),
      dict(
          testcase_name="provider",
          relative_parts=("langextract_filelink", "provider.py"),
          with_schema=False,
      ),
      dict(
          testcase_name="package_init",
          relative_parts=("langextract_filelink", "__init__.py"),
          with_schema=False,
      ),
      dict(
          testcase_name="schema",
          relative_parts=("langextract_filelink", "schema.py"),
          with_schema=True,
      ),
      dict(
          testcase_name="test_script",
          relative_parts=("test_plugin.py",),
          with_schema=False,
      ),
      dict(
          testcase_name="readme",
          relative_parts=("README.md",),
          with_schema=False,
      ),
      dict(
          testcase_name="gitignore",
          relative_parts=(".gitignore",),
          with_schema=False,
      ),
      dict(
          testcase_name="license",
          relative_parts=("LICENSE",),
          with_schema=False,
      ),
  )
  def test_cli_force_refuses_file_symlink_before_any_write(
      self, relative_parts, with_schema
  ):
    work_dir = pathlib.Path(self._make_tempdir())
    base_dir = work_dir / "langextract-filelink"
    package_dir = base_dir / "langextract_filelink"
    package_dir.mkdir(parents=True)
    victim = work_dir / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    link = base_dir.joinpath(*relative_parts)
    try:
      link.symlink_to(victim)
    except OSError as e:
      self.skipTest(f"symlinks unavailable on this platform: {e}")
    cli_args = ["--with-schema"] if with_schema else []

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "FileLink",
            "--force",
            *cli_args,
            "--no-install",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=work_dir,
    )

    self.assertEqual(result.returncode, 1)
    combined = result.stdout + result.stderr
    self.assertIn("symlink", combined)
    self.assertNotIn("Traceback", combined)
    self.assertTrue(link.is_symlink())
    self.assertEqual(victim.read_text(encoding="utf-8"), "untouched")
    output_paths = [
        base_dir / "pyproject.toml",
        package_dir / "provider.py",
        package_dir / "__init__.py",
        package_dir / "schema.py",
        base_dir / "test_plugin.py",
        base_dir / "README.md",
        base_dir / ".gitignore",
        base_dir / "LICENSE",
    ]
    for path in output_paths:
      if path != link:
        self.assertFalse(
            os.path.lexists(path),
            msg=f"preflight must reject before writing {path}",
        )

  @parameterized.named_parameters(
      dict(testcase_name="native_no_follow", use_no_follow=True),
      dict(testcase_name="atomic_fallback", use_no_follow=False),
  )
  def test_write_refuses_symlink_created_after_preflight(self, use_no_follow):
    platform_no_follow = getattr(os, "O_NOFOLLOW", 0)
    if use_no_follow and not platform_no_follow:
      self.skipTest("O_NOFOLLOW is unavailable on this platform")
    work_dir = pathlib.Path(self._make_tempdir())
    destination = work_dir / "provider.py"
    victim = work_dir / "victim.py"
    victim.write_text("untouched", encoding="utf-8")
    real_ensure = self._generator._ensure_safe_output_path

    def _swap_destination(path):
      real_ensure(path)
      path.symlink_to(victim)

    no_follow = platform_no_follow if use_no_follow else 0
    with mock.patch.object(
        self._generator.os, "O_NOFOLLOW", no_follow, create=True
    ):
      with mock.patch.object(
          self._generator,
          "_ensure_safe_output_path",
          autospec=True,
          side_effect=_swap_destination,
      ):
        with self.assertRaises(SystemExit):
          self._generator._write_generated_file(destination, "generated")

    self.assertTrue(destination.is_symlink())
    self.assertEqual(victim.read_text(encoding="utf-8"), "untouched")
    self.assertEmpty(list(work_dir.glob(".provider.py.*.tmp")))

  def test_hostile_but_valid_patterns_generate_valid_python(self):
    hostile_patterns = ["^a'b", '^x"y', '^t"""q', r"^d\d+"]
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "Hostile",
        "hostile",
        hostile_patterns,
        with_schema=False,
    )

    provider_source = (
        base_dir / "langextract_hostile" / "provider.py"
    ).read_text(encoding="utf-8")
    test_source = (base_dir / "test_plugin.py").read_text(encoding="utf-8")
    compile(provider_source, "provider.py", "exec")
    compile(test_source, "test_plugin.py", "exec")
    provider_tree = ast.parse(provider_source)
    register_call = next(
        node
        for node in ast.walk(provider_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "register"
    )
    self.assertEqual(
        [ast.literal_eval(arg) for arg in register_call.args], hostile_patterns
    )

    site_dir = self._make_fake_install(
        base_dir, "hostile", "langextract_hostile.provider:HostileLanguageModel"
    )
    result = self._run_generated_test_script(base_dir, extra_path=[site_dir])
    self.assertEqual(
        result.returncode,
        0,
        msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    self.assertIn("✅ All checks passed!", result.stdout)


class PluginInstallationTest(_CreateProviderPluginTestBase):

  @parameterized.named_parameters(
      dict(testcase_name="global", section="global"),
      dict(testcase_name="install", section="install"),
  )
  def test_pip_no_index_detects_config_file(self, section):
    config_path = pathlib.Path(self._make_tempdir()) / "pip.conf"
    config_path.write_text(f"[{section}]\nno-index = true\n", encoding="utf-8")

    with mock.patch.dict(
        os.environ, {"PIP_CONFIG_FILE": str(config_path)}, clear=True
    ):
      self.assertTrue(self._generator._pip_no_index_enabled())

  def test_pip_no_index_environment_overrides_config_file(self):
    config_path = pathlib.Path(self._make_tempdir()) / "pip.conf"
    config_path.write_text("[global]\nno-index = true\n", encoding="utf-8")

    with mock.patch.dict(
        os.environ,
        {"PIP_CONFIG_FILE": str(config_path), "PIP_NO_INDEX": "0"},
        clear=True,
    ):
      self.assertFalse(self._generator._pip_no_index_enabled())

  def test_install_and_test_preserves_cwd_and_scopes_subprocess_cwd(self):
    base_dir = pathlib.Path(self._make_tempdir())
    cwd_before = os.getcwd()
    fake_result = mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(
        self._generator,
        "_local_backend_problem",
        autospec=True,
        return_value=None,
    ):
      with mock.patch.object(
          self._generator.subprocess,
          "run",
          autospec=True,
          return_value=fake_result,
      ) as mock_run:
        with contextlib.redirect_stdout(io.StringIO()):
          success = self._generator.install_and_test(base_dir)

    self.assertTrue(success)
    self.assertEqual(os.getcwd(), cwd_before)
    for call in mock_run.call_args_list:
      self.assertEqual(call.kwargs.get("cwd"), base_dir)

  def test_install_and_test_fails_fast_without_langextract(self):
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={}, langextract_available=False
    )

    self.assertFalse(success)
    mock_run.assert_not_called()
    self.assertIn("LangExtract is not installed", output)
    self.assertIn("pip install langextract", output)

  def test_generated_test_script_fails_on_whitespace_only_output(self):
    """Whitespace-only inference output must fail like empty output does."""
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "WsChk",
        "wschk",
        ["^wschk"],
        with_schema=False,
    )
    site_dir = self._make_fake_install(
        base_dir, "wschk", "langextract_wschk.provider:WsChkLanguageModel"
    )
    self._mutate_installed_provider(
        site_dir,
        "wschk",
        'result = f"Mock response for: {prompt[:50]}..."',
        'result = " \\n\\t "',
    )

    result = self._run_generated_test_script(base_dir, extra_path=[site_dir])

    self.assertNotEqual(
        result.returncode,
        0,
        msg=(
            "whitespace-only inference output must exit nonzero; stdout:\n"
            f"{result.stdout}"
        ),
    )
    self.assertNotIn("✅ All checks passed!", result.stdout)

  @parameterized.named_parameters(
      dict(
          testcase_name="plain",
          provider_name="StrictPlain",
          package_name="strictplain",
          with_schema=False,
      ),
      dict(
          testcase_name="with_schema",
          provider_name="StrictSchema",
          package_name="strictschema",
          with_schema=True,
      ),
  )
  def test_generated_plugin_completes_strict_public_extract_call(
      self, provider_name, package_name, with_schema
  ):
    """An installed generated provider must survive a strict lx.extract."""
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        provider_name,
        package_name,
        [f"^{package_name}"],
        with_schema=with_schema,
    )
    site_dir = self._make_fake_install(
        base_dir,
        package_name,
        f"langextract_{package_name}.provider:{provider_name}LanguageModel",
    )
    driver = pathlib.Path(self._make_tempdir()) / "run_extract.py"
    driver.write_text(
        textwrap.dedent(f"""\
            import langextract as lx
            from langextract import data
            from langextract import factory

            lx.providers.load_plugins_once()
            config = factory.ModelConfig(
                model_id="{package_name}-model",
                provider="{provider_name}LanguageModel",
            )
            model = factory.create_model(config)
            result = lx.extract(
                text_or_documents="Alice met Bob in Paris.",
                prompt_description="Extract person names.",
                examples=[
                    data.ExampleData(
                        text="Carol went home.",
                        extractions=[
                            data.Extraction(
                                extraction_class="person",
                                extraction_text="Carol",
                            )
                        ],
                    )
                ],
                model=model,
                resolver_params={{"suppress_parse_errors": False}},
                show_progress=False,
            )
            print("EXTRACT_COMPLETED", type(result).__name__)
        """),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        check=False,
        cwd=self._make_tempdir(),
        env=self._subprocess_env(extra_path=[site_dir]),
    )

    self.assertEqual(
        result.returncode,
        0,
        msg=(
            "generated provider output must complete a public lx.extract"
            " call with parse errors unsuppressed; stdout:\n"
            f"{result.stdout}\nstderr:\n{result.stderr}"
        ),
    )
    self.assertIn("EXTRACT_COMPLETED", result.stdout)

  @pytest.mark.requires_pip
  def test_install_and_test_editable_install_is_offline_safe(self):
    """install_and_test must not depend on index access to install."""
    base_dir = pathlib.Path(self._make_tempdir()) / "langextract-offlineinst"
    self._generate_full_plugin(
        base_dir,
        "OfflineInst",
        "offlineinst",
        ["^offlineinst"],
        with_schema=False,
    )
    self.addCleanup(
        subprocess.run,
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "langextract-offlineinst",
        ],
        capture_output=True,
        check=False,
    )

    with mock.patch.dict(os.environ, {"PIP_NO_INDEX": "1"}):
      with contextlib.redirect_stdout(io.StringIO()) as out:
        success = self._generator.install_and_test(base_dir)

    self.assertTrue(
        success,
        msg=(
            "editable install must succeed with PIP_NO_INDEX=1 (no"
            " build-isolation downloads); output:\n"
            f"{out.getvalue()}"
        ),
    )

  def test_install_and_test_offline_succeeds_with_external_wheel(self):
    """Compatible setuptools + external wheel dist: offline-safe path."""
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={"PIP_NO_INDEX": "1"},
        present_specs=self._USABLE_BACKEND_SPECS + ("wheel",),
        setuptools_version="84.0.0",
    )

    self._assert_offline_safe_install(success, mock_run, output)

  def test_install_and_test_offline_succeeds_without_wheel(self):
    """Compatible setuptools vendors bdist_wheel: no external wheel needed.

    setuptools meets the generated build-system minimum and supplies
    setuptools._vendor.wheel; the wheel distribution itself is absent.
    The offline-safe no-build-isolation path must be taken and succeed.
    """
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={"PIP_NO_INDEX": "1"},
        present_specs=(
            self._USABLE_BACKEND_SPECS + ("setuptools._vendor.wheel",)
        ),
        setuptools_version="77.0.3",
    )

    self._assert_offline_safe_install(success, mock_run, output)

  def test_install_and_test_fails_fast_offline_without_setuptools(self):
    """Offline with no setuptools at all: fail before pip, name it."""
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={"PIP_NO_INDEX": "1"},
    )

    self._assert_fails_before_pip_with_upgrade_hint(success, mock_run, output)
    self.assertIn(
        "setuptools is not installed",
        output,
        msg="the message must state the actual problem: nothing installed",
    )

  def test_install_and_test_fails_fast_with_configured_no_index(self):
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={},
        configured_no_index=True,
    )

    self._assert_fails_before_pip_with_upgrade_hint(success, mock_run, output)
    self.assertIn(
        "setuptools is not installed",
        output,
        msg="the message must state the actual problem: nothing installed",
    )

  def test_install_and_test_online_isolates_build_without_setuptools(self):
    """Online with no setuptools: use build isolation, keep --no-deps."""
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={},
    )

    self._assert_online_isolated_install(success, mock_run, output)

  @parameterized.named_parameters(
      dict(testcase_name="with_external_wheel", extra_specs=("wheel",)),
      dict(testcase_name="without_wheel", extra_specs=()),
  )
  def test_install_and_test_fails_fast_offline_with_outdated_setuptools(
      self, extra_specs
  ):
    """Installed-but-inadequate setuptools: fail before pip, say upgrade.

    setuptools 60.10.0 imports fine but predates the PEP 660 editable
    hooks (setuptools 64.0.0) and the SPDX license support (77.0.0) the
    generated project needs. Whether or not the external wheel dist is
    installed, the diagnosis is the same backend inadequacy — never a
    misleading raw bdist_wheel/build_editable backend traceback.
    """
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={"PIP_NO_INDEX": "1"},
        present_specs=self._USABLE_BACKEND_SPECS + extra_specs,
        setuptools_version="60.10.0",
    )

    self._assert_fails_before_pip_with_upgrade_hint(success, mock_run, output)
    self.assertIn(
        "60.10.0",
        output,
        msg="the message must name the inadequate installed version",
    )
    self.assertNotIn(
        "bdist_wheel",
        output,
        msg="no raw backend error may leak into the diagnosis",
    )
    self.assertNotIn(
        "pip install wheel",
        output,
        msg=(
            "the wheel distribution is not the problem — the diagnosis"
            " must be the outdated setuptools backend"
        ),
    )

  def test_install_and_test_online_isolates_build_with_outdated_setuptools(
      self,
  ):
    """Outdated local setuptools online: pip build isolation takes over."""
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={},
        present_specs=self._USABLE_BACKEND_SPECS + ("wheel",),
        setuptools_version="60.10.0",
    )

    self._assert_online_isolated_install(success, mock_run, output)

  def test_install_failure_surfaces_stdout_diagnostics(self):
    """pip can put the actionable error on stdout — it must be shown."""
    failed_install = mock.Mock(
        returncode=1,
        stdout="ERROR: resolution-impossible detail only on stdout",
        stderr="",
    )

    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={"PIP_NO_INDEX": "1"},
        present_specs=self._USABLE_BACKEND_SPECS + ("wheel",),
        setuptools_version="84.0.0",
        pip_result=failed_install,
    )

    self.assertFalse(success)
    self.assertLen(
        mock_run.call_args_list,
        1,
        msg="a failed install must not proceed to the generated test step",
    )
    self.assertIn(
        "resolution-impossible detail only on stdout",
        output,
        msg=(
            "install-failure reporting must include pip's stdout — some"
            " pip diagnostics never reach stderr"
        ),
    )

  def test_install_failure_shows_both_streams_without_duplication(self):
    """Distinct streams both appear; identical streams appear once."""
    distinct = mock.Mock(
        returncode=1,
        stdout="stdout-only-clue",
        stderr="stderr-only-clue",
    )
    identical = mock.Mock(
        returncode=1,
        stdout="the-same-diagnostic",
        stderr="the-same-diagnostic",
    )

    _, _, distinct_output = self._run_install_with_hermetic_backend(
        env={"PIP_NO_INDEX": "1"},
        present_specs=self._USABLE_BACKEND_SPECS + ("wheel",),
        setuptools_version="84.0.0",
        pip_result=distinct,
    )
    _, _, identical_output = self._run_install_with_hermetic_backend(
        env={"PIP_NO_INDEX": "1"},
        present_specs=self._USABLE_BACKEND_SPECS + ("wheel",),
        setuptools_version="84.0.0",
        pip_result=identical,
    )

    self.assertIn("stdout-only-clue", distinct_output)
    self.assertIn("stderr-only-clue", distinct_output)
    self.assertEqual(
        identical_output.count("the-same-diagnostic"),
        1,
        msg="identical stdout/stderr must not be printed twice",
    )

  def test_install_and_test_fails_fast_offline_just_below_minimum(self):
    """Boundary: setuptools 76.9.9 sits just below the 77.0 floor.

    Pins the preflight floor to the generated build-system requirement
    (`setuptools>=77.0`): lowering the version constant without changing
    the generated template must trip this test.
    """
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={"PIP_NO_INDEX": "1"},
        present_specs=self._USABLE_BACKEND_SPECS + ("wheel",),
        setuptools_version="76.9.9",
    )

    self._assert_fails_before_pip_with_upgrade_hint(success, mock_run, output)
    self.assertIn(
        "76.9.9",
        output,
        msg="the message must name the just-below-minimum installed version",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="backend_import_broken",
          probe_kwargs={
              "present_specs": ("setuptools",),
              "broken_specs": ("setuptools.build_meta",),
              "setuptools_version": "84.0.0",
          },
      ),
      dict(
          testcase_name="version_metadata_missing",
          probe_kwargs={
              "present_specs": ("setuptools", "setuptools.build_meta"),
              "setuptools_version": None,
          },
      ),
      dict(
          testcase_name="version_probe_raises",
          probe_kwargs={
              "present_specs": ("setuptools", "setuptools.build_meta"),
              "setuptools_version": RuntimeError("corrupt metadata"),
          },
      ),
      dict(
          testcase_name="version_unparsable",
          probe_kwargs={
              "present_specs": ("setuptools", "setuptools.build_meta"),
              "setuptools_version": "not-a-version",
          },
      ),
  )
  def test_install_and_test_treats_backend_probe_errors_as_unavailable(
      self, probe_kwargs
  ):
    """Any find_spec/import/version-probe failure means no local backend.

    Offline, each of these must fail before pip with the actionable
    upgrade command — never leak a probe traceback or run a build that
    cannot succeed.
    """
    success, mock_run, output = self._run_install_with_hermetic_backend(
        env={"PIP_NO_INDEX": "1"},
        **probe_kwargs,
    )

    self._assert_fails_before_pip_with_upgrade_hint(success, mock_run, output)
    self.assertNotIn(
        "Traceback",
        output,
        msg="probe failures must be diagnosed, not dumped as tracebacks",
    )


class RegenerationAndPackagingTest(_CreateProviderPluginTestBase):

  def test_force_regenerate_without_schema_removes_stale_schema_recoverably(
      self,
  ):
    """--force from schema mode to non-schema mode must not keep schema.py."""
    work_dir = self._make_tempdir()
    self._run_generator_cli(work_dir, "SchemaFlip", "--with-schema")
    base_dir = pathlib.Path(work_dir) / "langextract-schemaflip"
    schema_path = base_dir / "langextract_schemaflip" / "schema.py"
    self.assertTrue(schema_path.exists(), msg="schema-mode run must write it")

    # Two schema-mode -> non-schema-mode round trips: each non-schema run
    # must retire the fresh schema.py without overwriting older backups.
    self._run_generator_cli(work_dir, "SchemaFlip", "--force")
    self._run_generator_cli(work_dir, "SchemaFlip", "--with-schema", "--force")
    self._run_generator_cli(work_dir, "SchemaFlip", "--force")

    self.assertFalse(
        schema_path.exists(),
        msg="non-schema regeneration must not leave schema.py behind",
    )
    # Probe importability in a SUBPROCESS: importing the package would run
    # its provider-registering __init__, and that must never touch this
    # process's global provider registry.
    probe = subprocess.run(
        [sys.executable, "-c", "import langextract_schemaflip.schema"],
        capture_output=True,
        text=True,
        check=False,
        cwd=self._make_tempdir(),
        env=self._subprocess_env(extra_path=[base_dir]),
    )
    self.assertNotEqual(
        probe.returncode,
        0,
        msg=(
            "stale schema.py from the schema-mode run must not be"
            " importable (or packageable) after --force regeneration"
            " without --with-schema; stdout:\n"
            f"{probe.stdout}\nstderr:\n{probe.stderr}"
        ),
    )
    self.assertIn(
        "ModuleNotFoundError",
        probe.stderr,
        msg="the probe must fail on the missing schema module itself",
    )

    with self.assertRaisesRegex(
        lx.exceptions.InferenceConfigError,
        "No provider registered",
        msg=(
            "the stale-schema importability check must not leak a live"
            " ^schemaflip registration into the process-global provider"
            " registry"
        ),
    ):
      lx.providers.registry.resolve("schemaflip-model")

    backups = [
        path
        for path in base_dir.rglob("*")
        if path.is_file()
        and path.suffix != ".pyc"
        and "class SchemaFlipSchema"
        in path.read_text(encoding="utf-8", errors="ignore")
    ]
    self.assertLen(
        backups,
        2,
        msg=(
            "cleanup must be recoverable and never overwrite: each of the"
            " two retired schema.py files must survive as its own backup"
            " carrying the original schema content, got"
            f" {[p.name for p in backups]}"
        ),
    )
    for path in backups:
      self.assertEndsWith(
          path.name,
          ".bak",
          msg=(
              f"backup {path.name!r} must end in .bak so the generated"
              " .gitignore's *.bak rule covers it"
          ),
      )

  def test_force_regenerate_retires_dangling_schema_symlink(self):
    work_dir = pathlib.Path(self._make_tempdir())
    self._run_generator_cli(work_dir, "DanglingSchema")
    package_dir = (
        work_dir / "langextract-danglingschema" / "langextract_danglingschema"
    )
    schema_path = package_dir / "schema.py"
    first_backup = package_dir / "schema.py.bak"
    try:
      schema_path.symlink_to("missing-schema-target.py")
      first_backup.symlink_to("older-missing-target.py")
    except OSError as e:
      self.skipTest(f"symlinks unavailable on this platform: {e}")

    self._run_generator_cli(work_dir, "DanglingSchema", "--force")

    self.assertFalse(os.path.lexists(schema_path))
    self.assertTrue(first_backup.is_symlink())
    self.assertEqual(os.readlink(first_backup), "older-missing-target.py")
    next_backup = package_dir / "schema.py.1.bak"
    self.assertTrue(next_backup.is_symlink())
    self.assertEqual(os.readlink(next_backup), "missing-schema-target.py")

  def test_readme_example_model_id_matches_first_pattern_when_derivable(self):
    source = self._generate_readme_source("DerivDoc", "derivdoc", [r"^d\d+"])

    literal = self._readme_usage_model_id(source)

    self.assertIsNotNone(
        literal,
        msg="derivable pattern must keep a runnable README usage example",
    )
    self.assertTrue(
        re.search(r"^d\d+", literal),
        msg=(
            f"README example model_id {literal!r} must genuinely match the"
            r" first pattern '^d\d+'"
        ),
    )

  def test_readme_has_no_misleading_example_for_underivable_pattern(self):
    source = self._generate_readme_source("LookDoc", "lookdoc", ["^(?=x)y"])

    literal = self._readme_usage_model_id(source)

    self.assertIsNone(
        literal,
        msg=(
            f"README presents {literal!r} as a runnable model_id, but no"
            " sample can be derived for pattern '^(?=x)y' — underivable"
            " patterns must omit the runnable usage example"
        ),
    )

  def test_readme_usage_uses_current_extract_keyword(self):
    source = self._generate_readme_source("KwDoc", "kwdoc", ["^kwdoc"])

    self.assertIn("```python\n", source)
    snippet = source.split("```python\n")[1].split("```")[0]
    extract_calls = [
        node
        for node in ast.walk(ast.parse(snippet))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "extract"
    ]
    self.assertLen(extract_calls, 1)
    keywords = [kw.arg for kw in extract_calls[0].keywords]
    self.assertIn(
        "text_or_documents",
        keywords,
        msg="README usage must pass the source text via text_or_documents=",
    )
    self.assertNotIn(
        "text",
        keywords,
        msg=(
            "lx.extract has no 'text' parameter — the README example must"
            " not use a legacy/invalid keyword"
        ),
    )

  def test_skipped_pattern_forbids_unqualified_all_checks_passed(self):
    """A pattern with no derivable sample must not report a clean pass."""
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "LookPat",
        "lookpat",
        ["^(?=x)y"],
        with_schema=False,
    )
    site_dir = self._make_fake_install(
        base_dir, "lookpat", "langextract_lookpat.provider:LookPatLanguageModel"
    )

    result = self._run_generated_test_script(base_dir, extra_path=[site_dir])

    self.assertNotEqual(
        result.returncode,
        0,
        msg=(
            "a skipped (underivable) pattern leaves registration"
            " unverified, so the generated script must exit nonzero —"
            " install_and_test relies on that exit code to mark the"
            f" plugin failed; stdout:\n{result.stdout}"
        ),
    )
    self.assertNotIn(
        "✅ All checks passed!",
        result.stdout,
        msg=(
            "a skipped (underivable) pattern must suppress the"
            " unqualified all-checks-passed banner"
        ),
    )

  def test_generated_license_file_matches_declared_license(self):
    base_dir = pathlib.Path(self._make_tempdir())
    with contextlib.redirect_stdout(io.StringIO()):
      self._generator.create_pyproject_toml(base_dir, "LicChk", "licchk")
      self._generator.create_readme(base_dir, "LicChk", "licchk", ["^licchk"])
      self._generator.create_license(base_dir)

    pyproject = tomllib.loads(
        (base_dir / "pyproject.toml").read_text(encoding="utf-8")
    )
    readme = (base_dir / "README.md").read_text(encoding="utf-8")
    license_text = (base_dir / "LICENSE").read_text(encoding="utf-8")

    license_field = pyproject.get("project", {}).get("license")
    if isinstance(license_field, dict):
      declared_license = license_field.get("text")
    else:
      declared_license = license_field
    self.assertEqual(
        declared_license,
        "Apache-2.0",
        msg="generated pyproject.toml must declare the Apache-2.0 license",
    )
    self.assertEqual(
        pyproject.get("project", {}).get("license-files"),
        ["LICENSE"],
        msg=(
            "generated pyproject.toml must ship the LICENSE file via"
            " project.license-files (PEP 639)"
        ),
    )
    self.assertIn(
        "Apache License 2.0",
        readme,
        msg="generated README must declare the Apache License 2.0",
    )
    self.assertNotIn(
        "TODO",
        license_text,
        msg=(
            "pyproject/README declare Apache-2.0 but LICENSE is a TODO"
            " placeholder — generated licensing must be consistent"
        ),
    )
    self.assertIn(
        "Version 2.0",
        license_text,
        msg="LICENSE must carry the Apache-2.0 license text it declares",
    )
    self.assertFalse(
        license_text.startswith("\n"),
        msg=(
            "LICENSE must start with the license text itself, not a blank"
            " line — the canonical Apache-2.0 file opens with the centered"
            " 'Apache License' heading"
        ),
    )

  @parameterized.named_parameters(
      dict(testcase_name="without_schema", with_schema=False),
      dict(testcase_name="with_schema", with_schema=True),
  )
  def test_generated_test_script_numbering_is_contiguous(self, with_schema):
    base_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "NumChk",
        "numchk",
        ["^numchk"],
        with_schema=with_schema,
    )
    source = (base_dir / "test_plugin.py").read_text(encoding="utf-8")

    numbers = [int(n) for n in re.findall(r'print\("(?:\\n)?(\d+)\. ', source)]

    self.assertNotEmpty(numbers)
    self.assertEqual(
        numbers,
        list(range(1, len(numbers) + 1)),
        msg=f"checklist numbering must be contiguous, got {numbers}",
    )

  def test_replace_placeholder_line_replaces_single_occurrence(self):
    content = "a\n__TOKEN__\nb\n"

    replaced = self._generator._replace_placeholder_line(
        content, "__TOKEN__", "x\n"
    )
    removed = self._generator._replace_placeholder_line(
        content, "__TOKEN__", ""
    )

    self.assertEqual(replaced, "a\nx\nb\n")
    self.assertEqual(removed, "a\nb\n")

  @parameterized.named_parameters(
      dict(testcase_name="missing", content="a\nb\n"),
      dict(
          testcase_name="duplicated",
          content="__TOKEN__\nmid\n__TOKEN__\n",
      ),
  )
  def test_replace_placeholder_line_rejects_non_unique(self, content):
    with self.assertRaises(ValueError):
      self._generator._replace_placeholder_line(content, "__TOKEN__", "x\n")

  def test_generated_schema_has_no_dead_extraction_types_computation(self):
    base_dir = pathlib.Path(self._make_tempdir())
    (base_dir / "langextract_deadcomp").mkdir(parents=True)
    with contextlib.redirect_stdout(io.StringIO()):
      self._generator.create_schema(base_dir, "DeadComp", "deadcomp")
    schema_path = base_dir / "langextract_deadcomp" / "schema.py"
    source = schema_path.read_text(encoding="utf-8")

    if "extraction_types" not in source:
      return  # The computation was removed; there is nothing dead to flag.

    spec = importlib.util.spec_from_file_location(
        "deadcomp_generated_schema", schema_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rich_examples = [
        data.ExampleData(
            text="Test text",
            extractions=[
                data.Extraction(
                    extraction_class="person",
                    extraction_text="Carol",
                    attributes={"role": "engineer"},
                ),
                data.Extraction(
                    extraction_class="city",
                    extraction_text="Paris",
                    attributes={"country": "France"},
                ),
            ],
        )
    ]

    rich = module.DeadCompSchema.from_examples(rich_examples)
    empty = module.DeadCompSchema.from_examples([])

    self.assertNotEqual(
        rich.schema_dict,
        empty.schema_dict,
        msg=(
            "from_examples computes extraction_types but the schema is"
            " example-independent — the computation is dead code; use it"
            " or remove it"
        ),
    )

  def test_generated_provider_api_key_annotation_allows_none(self):
    source = self._generate_provider_source(
        "AnnChk", "annchk", with_schema=False
    )

    tree = ast.parse(source)
    init = next(
        node
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name.endswith("LanguageModel")
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    arg_names = [a.arg for a in init.args.args]
    idx = arg_names.index("api_key")
    default_offset = len(init.args.args) - len(init.args.defaults)
    default = init.args.defaults[idx - default_offset]
    annotation = init.args.args[idx].annotation

    self.assertIsInstance(default, ast.Constant)
    self.assertIsNone(default.value)
    self.assertIsNotNone(annotation)
    self.assertFalse(
        isinstance(annotation, ast.Name) and annotation.id == "str",
        msg=(
            "api_key defaults to None but is annotated plain 'str' —"
            " under Python 3.10+ conventions the annotation must be"
            " 'str | None' (or Optional[str])"
        ),
    )

  @pytest.mark.requires_pip
  def test_pep660_editable_install_passes_and_adjacent_source_still_fails(
      self,
  ):
    """The supported real editable-install path must pass the test script."""
    base_dir = pathlib.Path(self._make_tempdir()) / "langextract-pepreal"
    self._generate_full_plugin(
        base_dir,
        "PepReal",
        "pepreal",
        ["^pepreal"],
        with_schema=False,
    )
    install_env = dict(os.environ)
    install_env["PIP_NO_INDEX"] = "1"
    self.addCleanup(
        subprocess.run,
        [sys.executable, "-m", "pip", "uninstall", "-y", "langextract-pepreal"],
        capture_output=True,
        check=False,
    )
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "--no-build-isolation",
            "--no-deps",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=base_dir,
        env=install_env,
    )
    self.assertEqual(
        install.returncode,
        0,
        msg=f"stdout:\n{install.stdout}\nstderr:\n{install.stderr}",
    )

    installed = self._run_generated_test_script(base_dir)
    self.assertEqual(
        installed.returncode,
        0,
        msg=(
            "generated test script must pass under a real PEP-660 editable"
            f" install; stdout:\n{installed.stdout}\n"
            f"stderr:\n{installed.stderr}"
        ),
    )
    self.assertIn("✅ All checks passed!", installed.stdout)

    ghost_dir = self._generate_full_plugin(
        self._make_tempdir(),
        "PepGhost",
        "pepghost",
        ["^pepghost"],
        with_schema=False,
    )
    ghost = self._run_generated_test_script(ghost_dir)
    self.assertEqual(
        ghost.returncode,
        1,
        msg=f"stdout:\n{ghost.stdout}\nstderr:\n{ghost.stderr}",
    )
    self.assertIn("Plugin not installed", ghost.stdout)

  @pytest.mark.requires_pip
  def test_generated_wheel_and_sdist_carry_license_without_deprecation(self):
    """Real wheel/sdist builds: LICENSE + entry point in, deprecation out.

    Builds the generated project with the local backend (verbose so
    backend warnings surface), asserts no project.license deprecation is
    emitted, inspects the artifacts, then installs the wheel and runs a
    strict lx.extract from a neutral cwd.
    """
    base_dir = pathlib.Path(self._make_tempdir()) / "langextract-wheelchk"
    self._generate_full_plugin(
        base_dir,
        "WheelChk",
        "wheelchk",
        ["^wheelchk"],
        with_schema=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
      self._generator.create_readme(
          base_dir, "WheelChk", "wheelchk", ["^wheelchk"]
      )
      self._generator.create_license(base_dir)
    dist_dir = pathlib.Path(self._make_tempdir())
    offline_env = dict(os.environ)
    offline_env["PIP_NO_INDEX"] = "1"

    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(base_dir),
            "--no-deps",
            "--no-build-isolation",
            "-v",
            "-w",
            str(dist_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=self._make_tempdir(),
        env=offline_env,
    )
    self.assertEqual(
        build.returncode,
        0,
        msg=f"stdout:\n{build.stdout}\nstderr:\n{build.stderr}",
    )
    build_output = build.stdout + build.stderr
    self.assertNotIn(
        "`project.license` as a TOML table is deprecated",
        build_output,
        msg=(
            "the generated pyproject must use the SPDX string form so"
            " setuptools emits no 2027-deadline license deprecation"
        ),
    )

    wheels = list(dist_dir.glob("*.whl"))
    self.assertLen(wheels, 1, msg=f"dist contents: {list(dist_dir.iterdir())}")
    with zipfile.ZipFile(wheels[0]) as wheel_file:
      names = wheel_file.namelist()
      license_names = [n for n in names if n.endswith("licenses/LICENSE")]
      self.assertLen(
          license_names,
          1,
          msg=f"wheel must ship the LICENSE, contents: {names}",
      )
      license_text = wheel_file.read(license_names[0]).decode("utf-8")
      self.assertStartsWith(
          license_text, "                                 Apache License"
      )
      entry_point_names = [n for n in names if n.endswith("entry_points.txt")]
      self.assertLen(entry_point_names, 1)
      entry_points_text = wheel_file.read(entry_point_names[0]).decode("utf-8")
      self.assertIn("[langextract.providers]", entry_points_text)
      self.assertIn(
          "wheelchk = langextract_wheelchk.provider:WheelChkLanguageModel",
          entry_points_text,
      )
      metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
      self.assertLen(metadata_names, 1)
      metadata_text = wheel_file.read(metadata_names[0]).decode("utf-8")
      self.assertIn("License-Expression: Apache-2.0", metadata_text)

    sdist_out = pathlib.Path(self._make_tempdir())
    sdist_build = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from setuptools.build_meta import build_sdist;"
                f" build_sdist({str(sdist_out)!r})"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=base_dir,
        env=offline_env,
    )
    self.assertEqual(
        sdist_build.returncode,
        0,
        msg=f"stdout:\n{sdist_build.stdout}\nstderr:\n{sdist_build.stderr}",
    )
    self.assertNotIn(
        "`project.license` as a TOML table is deprecated",
        sdist_build.stdout + sdist_build.stderr,
    )
    sdists = list(sdist_out.glob("*.tar.gz"))
    self.assertLen(sdists, 1)
    with tarfile.open(sdists[0]) as sdist_file:
      sdist_names = sdist_file.getnames()
    self.assertTrue(
        any(n.endswith("/LICENSE") for n in sdist_names),
        msg=f"sdist must ship the LICENSE, contents: {sdist_names}",
    )
    self.assertTrue(
        any(n.endswith("/pyproject.toml") for n in sdist_names),
        msg="sdist must carry pyproject.toml (the entry-point declaration)",
    )

    self.addCleanup(
        subprocess.run,
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "langextract-wheelchk",
        ],
        capture_output=True,
        check=False,
    )
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(wheels[0]),
            "--no-deps",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=self._make_tempdir(),
        env=offline_env,
    )
    self.assertEqual(
        install.returncode,
        0,
        msg=f"stdout:\n{install.stdout}\nstderr:\n{install.stderr}",
    )

    driver = pathlib.Path(self._make_tempdir()) / "run_extract.py"
    driver.write_text(
        textwrap.dedent("""\
            import langextract as lx
            from langextract import data
            from langextract import factory

            lx.providers.load_plugins_once()
            config = factory.ModelConfig(
                model_id="wheelchk-model", provider="WheelChkLanguageModel"
            )
            model = factory.create_model(config)
            result = lx.extract(
                text_or_documents="Alice met Bob in Paris.",
                prompt_description="Extract person names.",
                examples=[
                    data.ExampleData(
                        text="Carol went home.",
                        extractions=[
                            data.Extraction(
                                extraction_class="person",
                                extraction_text="Carol",
                            )
                        ],
                    )
                ],
                model=model,
                resolver_params={"suppress_parse_errors": False},
                show_progress=False,
            )
            print("EXTRACT_COMPLETED", type(result).__name__)
        """),
        encoding="utf-8",
    )
    extract = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        check=False,
        cwd=self._make_tempdir(),
        env=self._subprocess_env(),
    )
    self.assertEqual(
        extract.returncode,
        0,
        msg=f"stdout:\n{extract.stdout}\nstderr:\n{extract.stderr}",
    )
    self.assertIn("EXTRACT_COMPLETED", extract.stdout)


if __name__ == "__main__":
  absltest.main()
