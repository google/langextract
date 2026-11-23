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

Files that predate this check are listed in `_LEGACY_UNSPECCED` with the
number of unspecced calls they had when it was added. A file fails when it has
more than that.

Usage:
  python scripts/check_mock_patch_spec.py tests
"""

import ast
from collections.abc import Mapping, Sequence
import pathlib
import sys

_SPEC_KEYWORDS = frozenset(
    {"autospec", "spec", "spec_set", "new", "new_callable"}
)

# Positional index of `new`, by the name the call is spelled with. Matching
# the full name keeps unrelated calls such as `requests.patch(url)` out.
_NEW_POSITION: Mapping[str, int] = {
    "patch": 1,
    "mock.patch": 1,
    "unittest.mock.patch": 1,
    "patch.object": 2,
    "mock.patch.object": 2,
    "unittest.mock.patch.object": 2,
}

# Unspecced calls that predate this check, by file name. Do not raise a
# number. Lower it when you fix calls in that file.
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
    new_position = _NEW_POSITION.get(ast.unparse(node.func))
    if new_position is None:
      continue
    has_positional_new = len(node.args) > new_position
    has_spec_keyword = any(k.arg in _SPEC_KEYWORDS for k in node.keywords)
    if not has_positional_new and not has_spec_keyword:
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
    if len(lines) <= allowed:
      continue
    locations = ", ".join(str(line) for line in lines)
    message = (
        f"{file}: {len(lines)} mock.patch or mock.patch.object call(s) pass"
        " none of autospec, spec, spec_set, new or new_callable. Add"
        " autospec=True to the call you added, or spec=/new= when autospec"
        f" does not apply. Unspecced calls are on lines: {locations}."
    )
    if allowed:
      message += (
          f" {allowed} of these predate the check and are tolerated; do not"
          " raise that number."
      )
    errors.append(message)
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
