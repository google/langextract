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

# Unspecced calls that predate this check, keyed by the checked directory's
# name plus the path inside it. Do not raise a number. Lower it when you fix
# calls in that file.
_LEGACY_UNSPECCED: Mapping[str, int] = {
    "tests/extract_schema_integration_test.py": 17,
    "tests/factory_schema_test.py": 11,
    "tests/gemini_retry_test.py": 8,
    "tests/inference_test.py": 27,
    "tests/init_test.py": 22,
    "tests/io_test.py": 3,
    "tests/openai_batch_test.py": 6,
    "tests/provider_plugin_test.py": 9,
    "tests/provider_schema_test.py": 4,
    "tests/test_gemini_batch_api.py": 1,
    "tests/test_kwargs_passthrough.py": 9,
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


def check_paths(
    paths: Sequence[pathlib.Path], legacy: Mapping[str, int]
) -> list[str]:
  """Checks every Python file under the given paths.

  Args:
    paths: Files or directories to check.
    legacy: Allowed number of unspecced calls per file, keyed by the checked
      directory's name plus the path inside it, such as "tests/io_test.py".

  Returns:
    One message per failing file. Empty when everything passes.
  """
  errors = []
  for path in paths:
    # Resolved so the key is the same however the path was spelled.
    path = path.resolve()
    root = path if path.is_dir() else path.parent
    for file in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
      key = (root.name / file.relative_to(root)).as_posix()
      lines = find_unspecced_patches(file.read_bytes(), filename=str(file))
      allowed = legacy.get(key, 0)
      if len(lines) <= allowed:
        continue
      locations = ", ".join(str(line) for line in lines)
      message = (
          f"{key}: {len(lines)} mock.patch or mock.patch.object call(s) pass"
          " none of autospec, spec, spec_set, new or new_callable. Add"
          " autospec=True to the call you added, or spec=/new= when autospec"
          f" does not apply. Unspecced calls are on lines: {locations}."
      )
      if allowed:
        message += (
            f" {allowed} of these predate the check and are tolerated; do not"
            " raise that number in _LEGACY_UNSPECCED in"
            " scripts/check_mock_patch_spec.py."
        )
      errors.append(message)
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
  errors = check_paths(paths, _LEGACY_UNSPECCED)
  for error in errors:
    print(error, file=sys.stderr)
  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
