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


def find_unspecced_patches(
    source: str | bytes, filename: str = "<unknown>"
) -> list[int]:
  """Finds patch calls that do not declare a replacement.

  Args:
    source: Python source code to inspect.
    filename: Name reported if the source does not parse.

  Returns:
    Line numbers of the offending calls, in ascending order.
  """
  lines = []
  for node in ast.walk(ast.parse(source, filename=filename)):
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


def check_paths(paths: Sequence[pathlib.Path]) -> list[str]:
  """Checks every Python file under the given paths.

  Args:
    paths: Files or directories to check.

  Returns:
    One message per failing file. Empty when everything passes.
  """
  errors = []
  for path in paths:
    files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
    for file in files:
      lines = find_unspecced_patches(file.read_bytes(), filename=str(file))
      if not lines:
        continue
      locations = ", ".join(str(line) for line in lines)
      errors.append(
          f"{file}: {len(lines)} mock.patch or mock.patch.object call(s) pass"
          " none of autospec, spec, spec_set, new or new_callable. Add"
          " autospec=True, or spec=/new= when autospec does not apply."
          f" Unspecced calls are on lines: {locations}."
      )
  return errors


def main(argv: Sequence[str]) -> int:
  if not argv:
    print(__doc__, file=sys.stderr)
    return 2
  paths = [pathlib.Path(arg) for arg in argv]
  missing = [str(path) for path in paths if not path.exists()]
  if missing:
    print(f"No such file or directory: {', '.join(missing)}", file=sys.stderr)
    return 2
  errors = check_paths(paths)
  for error in errors:
    print(error, file=sys.stderr)
  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
