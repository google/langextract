#!/usr/bin/env python3
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

"""Checks that mock.patch calls in tests say what they replace the target with.

A bare `mock.patch("pkg.func")` installs a MagicMock that accepts any call and
any attribute, so a test keeps passing after the real signature changes or
when an attribute name is misspelled. Every `mock.patch(...)` and
`mock.patch.object(...)` call must therefore pass one of `autospec`, `spec`,
`spec_set`, `new` (keyword or positional) or `new_callable`.

Files that predate this check are listed in `_LEGACY_UNSPECCED` with their
current number of unspecced calls. A file fails when its count differs from
the recorded one: above means a new unspecced call, below means the entry
should be lowered so the improvement cannot regress.

Usage:
  python scripts/check_mock_patch_spec.py tests
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import pathlib
import sys

_SPEC_KEYWORDS = frozenset(
    {"autospec", "spec", "spec_set", "new", "new_callable"}
)

# Positional index of `new` in patch(target, new) and
# patch.object(target, attribute, new).
_NEW_POSITION_IN_PATCH = 1
_NEW_POSITION_IN_PATCH_OBJECT = 2

# Unspecced calls that predate this check, by file name. Lower a number when
# you fix calls in that file, and remove the entry when it reaches zero.
_LEGACY_UNSPECCED: Mapping[str, int] = {
    "extract_schema_integration_test.py": 17,
    "factory_schema_test.py": 11,
    "gemini_retry_test.py": 8,
    "inference_test.py": 27,
    "init_test.py": 22,
    "io_test.py": 3,
    "openai_batch_test.py": 6,
    "provider_plugin_test.py": 9,
    "provider_schema_test.py": 4,
    "test_gemini_batch_api.py": 1,
    "test_kwargs_passthrough.py": 9,
}

_FIX_HINT = (
    "Pass autospec=True (preferred), or spec=, spec_set=, new= or"
    " new_callable= when autospec does not apply."
)


def _is_patch(node: ast.expr) -> bool:
  """Returns True for `patch`, `mock.patch` and `unittest.mock.patch`."""
  if isinstance(node, ast.Name):
    return node.id == "patch"
  return isinstance(node, ast.Attribute) and node.attr == "patch"


def _new_position(call: ast.Call) -> int | None:
  """Returns where `new` sits positionally, or None if not a patch call."""
  func = call.func
  if _is_patch(func):
    return _NEW_POSITION_IN_PATCH
  if (
      isinstance(func, ast.Attribute)
      and func.attr == "object"
      and _is_patch(func.value)
  ):
    return _NEW_POSITION_IN_PATCH_OBJECT
  return None


def _is_off(value: ast.expr) -> bool:
  """Returns True for a literal False or None, as in `autospec=False`."""
  return isinstance(value, ast.Constant) and value.value in (False, None)


def _declares_replacement(call: ast.Call, new_position: int) -> bool:
  """Returns True if the call says what the target is replaced with."""
  if len(call.args) > new_position:
    return True
  for keyword in call.keywords:
    if keyword.arg == "new":
      return True
    if keyword.arg in _SPEC_KEYWORDS and not _is_off(keyword.value):
      return True
  return False


def find_unspecced_patches(source: str) -> list[int]:
  """Finds patch calls that do not declare a replacement.

  Args:
    source: Python source code to inspect.

  Returns:
    Line numbers of the offending calls, in ascending order.
  """
  lines = []
  for node in ast.walk(ast.parse(source)):
    if not isinstance(node, ast.Call):
      continue
    new_position = _new_position(node)
    if new_position is None:
      continue
    if not _declares_replacement(node, new_position):
      lines.append(node.lineno)
  return sorted(lines)


def check_paths(
    paths: Sequence[pathlib.Path], legacy: Mapping[str, int]
) -> list[str]:
  """Checks every Python file under the given paths.

  Args:
    paths: Files or directories to check.
    legacy: Allowed number of unspecced calls per file name.

  Returns:
    One message per failing file. Empty when everything passes.
  """
  files = []
  for path in paths:
    files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])

  errors = []
  for file in files:
    lines = find_unspecced_patches(file.read_text(encoding="utf-8"))
    allowed = legacy.get(file.name, 0)
    if len(lines) > allowed:
      locations = ", ".join(str(line) for line in lines)
      errors.append(
          f"{file}: {len(lines)} mock.patch call(s) without a spec, {allowed}"
          f" allowed. {_FIX_HINT} Calls are on lines: {locations}."
      )
    elif len(lines) < allowed:
      errors.append(
          f"{file}: {len(lines)} mock.patch call(s) without a spec, but"
          f" {allowed} recorded. Lower its entry in _LEGACY_UNSPECCED in"
          f" {pathlib.Path(__file__).name} to {len(lines)}."
      )
  return errors


def main(argv: Sequence[str]) -> int:
  if not argv:
    print(__doc__, file=sys.stderr)
    return 2
  errors = check_paths([pathlib.Path(arg) for arg in argv], _LEGACY_UNSPECCED)
  for error in errors:
    print(error, file=sys.stderr)
  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
