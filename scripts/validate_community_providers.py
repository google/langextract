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

#!/usr/bin/env python3
"""Validation for COMMUNITY_PROVIDERS.md plugin registry table."""

import os
from pathlib import Path
import re
import re as regex_module
import sys
from typing import Dict, List, Tuple

HEADER_ANCHOR = '| Plugin Name | PyPI Package |'
END_MARKER = '<!-- ADD NEW PLUGINS ABOVE THIS LINE -->'

# GitHub username/org and repo patterns
GH_NAME = r'[-a-zA-Z0-9]+'  # usernames/orgs allow hyphens
GH_REPO = r'[-a-zA-Z0-9._]+'  # repos allow ., _
GH_USER_LINK = rf'\[@{GH_NAME}\]\(https://github\.com/{GH_NAME}\)'
GH_MULTI_USER = rf'^{GH_USER_LINK}(,\s*{GH_USER_LINK})*$'

# Markdown link to a GitHub repo
GH_REPO_LINK = rf'^\[[^\]]+\]\(https://github\.com/{GH_NAME}/{GH_REPO}\)$'

# Issue link must point to LangExtract repository (issues only)
LANGEXTRACT_ISSUE_LINK = (
    r'^\[[^\]]+\]\(https://github\.com/google/langextract/issues/\d+\)$'
)

# PEP 503-ish normalized name (loose): lowercase letters/digits with - _ . separators
PYPI_NORMALIZED = r'`[a-z0-9]([\-_.]?[a-z0-9]+)*`'

MIN_DESC_LEN = 10


def normalize_pypi(name: str) -> str:
  """PEP 503 normalization for PyPI package names."""
  return regex_module.sub(r'[-_.]+', '-', name.strip().lower())


def find_table_bounds(lines: List[str]) -> Tuple[int, int]:
  start = end = -1
  for i, line in enumerate(lines):
    if HEADER_ANCHOR in line:
      start = i
    elif start >= 0 and END_MARKER in line:
      end = i
      break
  return start, end


def parse_row(line: str) -> List[str]:
  # assumes caller trimmed line
  parts = [c.strip() for c in line.split('|')[1:-1]]
  return parts


def validate(filepath: Path) -> bool:
  errors: List[str] = []
  warnings: List[str] = []

  content = filepath.read_text(encoding='utf-8')
  lines = content.splitlines()

  start, end = find_table_bounds(lines)
  if start < 0:
    errors.append('Could not find plugin registry table header.')
    print_report(errors, warnings)
    return False
  if end < 0:
    errors.append(
        'Could not find end marker: <!-- ADD NEW PLUGINS ABOVE THIS LINE -->.'
    )
    print_report(errors, warnings)
    return False

  rows: List[Dict] = []
  seen_names = set()
  seen_pkgs = set()

  for i in range(start + 2, end):
    raw = lines[i].strip()
    if not raw:
      continue

    if not raw.startswith('|') or not raw.endswith('|'):
      errors.append(
          f"Line {i+1}: Not a valid table row (must start and end with '|')."
      )
      continue

    cols = parse_row(raw)
    if len(cols) != 6:
      errors.append(f'Line {i+1}: Expected 6 columns, found {len(cols)}.')
      continue

    plugin, pypi, maint, repo, desc, issue_link = cols

    # Basic presence checks
    if not plugin:
      errors.append(f'Line {i+1}: Plugin Name is required.')

    if not re.fullmatch(PYPI_NORMALIZED, pypi):
      errors.append(
          f'Line {i+1}: PyPI package must be backticked and normalized (e.g.,'
          ' `langextract-provider-foo`).'
      )
    elif pypi and not pypi.strip('`').lower().startswith('langextract-'):
      errors.append(
          f'Line {i+1}: PyPI package should start with `langextract-` for'
          ' discoverability.'
      )

    if not re.fullmatch(GH_MULTI_USER, maint):
      errors.append(
          f'Line {i+1}: Maintainer must be one or more GitHub handles as links '
          '(e.g., [@alice](https://github.com/alice) or comma-separated).'
      )

    if not re.fullmatch(GH_REPO_LINK, repo):
      errors.append(
          f'Line {i+1}: GitHub Repo must be a Markdown link to a GitHub'
          ' repository.'
      )

    if not desc or len(desc) < MIN_DESC_LEN:
      errors.append(
          f'Line {i+1}: Description must be at least {MIN_DESC_LEN} characters.'
      )

    # Issue link is required and must point to LangExtract repo
    if not issue_link:
      errors.append(f'Line {i+1}: Issue Link is required.')
    elif not re.fullmatch(LANGEXTRACT_ISSUE_LINK, issue_link):
      errors.append(
          f'Line {i+1}: Issue Link must point to a LangExtract issue (e.g.,'
          ' [#123](https://github.com/google/langextract/issues/123)).'
      )

    rows.append({
        'line': i + 1,
        'plugin': plugin,
        'pypi': pypi.strip('`').lower() if pypi else '',
    })

  # Duplicate checks (case-insensitive and PEP 503 normalized)
  for r in rows:
    pn_key = r['plugin'].strip().casefold()
    pk_key = normalize_pypi(r['pypi']) if r['pypi'] else None

    if pn_key in seen_names:
      errors.append(f"Line {r['line']}: Duplicate Plugin Name '{r['plugin']}'.")
    seen_names.add(pn_key)

    if pk_key and pk_key in seen_pkgs:
      errors.append(f"Line {r['line']}: Duplicate PyPI Package '{r['pypi']}'.")
    if pk_key:
      seen_pkgs.add(pk_key)

  # Required alphabetical sorting check
  sorted_by_name = sorted(rows, key=lambda r: r['plugin'].casefold())
  if [r['plugin'] for r in rows] != [r['plugin'] for r in sorted_by_name]:
    errors.append('Registry rows must be alphabetically sorted by Plugin Name.')

  # Guardrail: discourage leaving only the example entry
  if len(rows) == 1 and rows[0]['plugin'].lower().startswith('example'):
    warnings.append(
        'The registry currently contains only the example row. Add real'
        ' providers above the marker.'
    )

  print_report(errors, warnings)
  return not errors


def print_report(errors: List[str], warnings: List[str]) -> None:
  if errors:
    print('❌ Validation failed:')
    for e in errors:
      print(f'  • {e}')
  if warnings:
    print('⚠️  Warnings:')
    for w in warnings:
      print(f'  • {w}')
  if not errors and not warnings:
    print('✅ Table format validation passed!')


def validate_input_path(input_str: str) -> Path:
  """Validate and sanitize user-provided path to prevent traversal attacks.
  
  Args:
    input_str: The raw input string from command line arguments
    
  Returns:
    A validated Path object safe to use for file operations
    
  Raises:
    ValueError: If the path contains invalid characters or attempts traversal
  """
  import string
  
  # Strictly validate the input string
  if not input_str:
    raise ValueError('Path cannot be empty')
  
  # Check for null bytes and other dangerous characters
  if '\x00' in input_str or '\n' in input_str or '\r' in input_str:
    raise ValueError('Path contains invalid characters')
  
  # Check for suspicious path patterns that indicate traversal attempts
  if '..' in input_str or input_str.startswith('/') or input_str.startswith('~'):
    raise ValueError('Path traversal detected')
  
  # Only allow alphanumeric, hyphen, underscore, dot, and forward slash in paths
  allowed_chars = set(string.ascii_letters + string.digits + '-_./')
  if not all(c in allowed_chars for c in input_str):
    raise ValueError('Path contains invalid characters')
  
  # Create the path
  path = Path(input_str)
  
  # Ensure path doesn't contain suspicious components
  for part in path.parts:
    if part == '..' or part == '.':
      raise ValueError('Invalid path component')
    if part.startswith('-') or part.startswith('/'):
      raise ValueError('Invalid path component')
  
  return path


if __name__ == '__main__':
  # Set default path
  path = Path('COMMUNITY_PROVIDERS.md')
  
  # Validate and use command line argument if provided
  if len(sys.argv) > 1:
    user_input = sys.argv[1]
    # Immediately validate the user input before any processing
    if not user_input or not isinstance(user_input, str):
      print(f'❌ Error: Invalid path provided: input must be a non-empty string')
      sys.exit(1)
    
    # Block dangerous patterns
    if '..' in user_input or user_input.startswith('/') or user_input.startswith('~'):
      print(f'❌ Error: Invalid path provided: path traversal detected')
      sys.exit(1)
    
    # Block special characters that could be used in attacks
    dangerous_chars = ['|', '&', ';', '$', '`', '\n', '\r', '\x00', '*', '?', '(', ')', '<', '>', '{', '}', '[', ']']
    if any(char in user_input for char in dangerous_chars):
      print(f'❌ Error: Invalid path provided: contains dangerous characters')
      sys.exit(1)
    
    # Create path from validated input
    path = Path(user_input)
  
  # Verify file exists
  if not path.exists():
    print(f'❌ Error: File not found: {path}')
    sys.exit(1)
  
  # Validate the file
  ok = validate(path)
  sys.exit(0 if ok else 1)
