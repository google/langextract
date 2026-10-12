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

"""Checks a pull request for changes that need a security review.

The check reads the diff and the pull request title, description, and branch
name as plain text. It never imports, installs, or runs code from the pull
request. CI runs this file from the base branch, so a pull request cannot
change how it is checked.

Findings have three severities:
  high: credentials or repository privileges could be exposed.
  medium: a maintainer should take a look.
  low: informational.

Patterns are written so that this file does not match its own rules (for
example, dots in host names are regex-escaped).

Usage:
  python pr_security_check.py --base BASE_SHA --head HEAD_SHA \
      [--title-file F] [--body-file F] [--branch NAME] [--maintainer] \
      [--json-out F] [--summary-out F] [--brief] [--fail-on-high]

Exit status is 0, or 3 when --fail-on-high is set and a high finding exists.
"""

from __future__ import annotations

import argparse
import codecs
from collections.abc import Iterable, Sequence
import dataclasses
import json
import pathlib
import re
import subprocess
import sys
import unicodedata

HIGH = "high"
MEDIUM = "medium"
LOW = "low"
_SEVERITY_ORDER = {HIGH: 0, MEDIUM: 1, LOW: 2}
EXIT_HIGH_FINDINGS = 3

# Request-logging and callback services that are not a dependency of this
# library. Dots are escaped so this source does not match itself.
_CALLBACK_HOST_RE = re.compile(
    r"(?i)\b(?:[a-z0-9-]+\.)*(?:"
    r"webhook\.site|oastify\.com|burpcollaborator\.net|interact\.sh|"
    r"oast\.(?:fun|live|me|online|pro|site)|pipedream\.net|"
    r"requestbin\.(?:com|net)|requestcatcher\.com|beeceptor\.com|"
    r"ngrok(?:-free)?\.(?:io|app|dev)|dnslog\.cn|ceye\.io|"
    r"canarytokens\.(?:com|org)|hookbin\.com|webhook\.cool"
    r")\b"
)

# Reading environment variables from Python, JavaScript, or Actions.
_ENV_ACCESS_RE = re.compile(
    r"\b(?:os\.)?(?:environ|getenv)\b|process\.env\b|\$\{\{\s*secrets\."
)
# Shell variable references; only meaningful outside Python sources.
_SHELL_ENV_RE = re.compile(r"\$\{?[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET)\b")

# Copying the whole environment, which includes every credential at once.
_ENV_DUMP_RE = re.compile(
    r"(?m)\benviron\s*\)|\benviron\.(?:items|copy|keys|values)\(|"
    r"\bprint[e]nv\b|\$\(\s*env\b|(?:^|[;&|]\s*)env\s*\|"
)

# Names that usually hold credentials.
_SECRET_NAME_RE = re.compile(
    r"\b[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)\b"
)

# Network access from Python: calls and imports, including `import os, x`.
# Calls are lowercase functions or client constructors, so exception
# constructors such as `httpx.ConnectError(...)` in tests do not count.
_PY_NETWORK_RE = re.compile(
    r"(?m)\burllib\.request\b|\burlopen\(|"
    r"\b(?:requests|httpx|aiohttp|urllib3)\."
    r"(?:[a-z_]\w*|(?:Async)?Client|(?:Client)?Session|PoolManager)\(|"
    r"\bhttp\.client\b|\bsocket\.(?:socket|create_)|"
    r"^\s*import\s+[^#\n]*\b(?:requests|httpx|aiohttp|urllib3|socket)\b|"
    r"^\s*from\s+(?:requests|httpx|aiohttp|urllib3|socket|http\.client)\b"
)
_IMPORT_LINE_RE = re.compile(r"^\s*(?:import|from)\s")
# Network access from the shell. In Python sources these words only count
# on a line that also starts a process, since they otherwise sit in strings.
_SHELL_NETWORK_RE = re.compile(
    r"\b(?:curl|wget|nslookup|Invoke-WebRequest)\b|\bnc\s+-"
)
_LOCAL_TARGET_RE = re.compile(r"\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0)\b")
# URLs to literal public IPv4 addresses (private and loopback excluded).
_RAW_IP_URL_RE = re.compile(
    r"\b[a-z]+://(?!(?:127|10|0)\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)"
    r"\d{1,3}(?:\.\d{1,3}){3}\b"
)
_ENCODING_RE = re.compile(
    r"\bb(?:64|ase64)\b|b64(?:en|de)code|base64\.|zlib\.decompress|"
    r"bytes\.fromhex|codecs\.decode|binascii\.|marshal\.loads"
)
# Builtin calls only; attribute calls such as `re.compile(` are excluded.
_DYNAMIC_EXEC_RE = re.compile(
    r"(?:\bbuiltins\.|(?<![\w.]))(?:exec|eval|compile)\s*\(|"
    r"\bmarshal\.loads\(|__import__\("
)
_PROCESS_RE = re.compile(r"\bsubprocess\.|\bos\.(?:system|popen)\(")
_PYTHON_SUFFIXES = frozenset({".py", ".pyi"})

# Command substitution in PR text that would run if a shell expanded it.
_COMMAND_SUBSTITUTION_RE = re.compile(
    r"\$\(\s*(?:/[\w./-]*/)?(?:curl|wget|nc|ncat|nslookup|dig|bash|sh|"
    r"python3?|base64|whoami|id|e[n]v|cat|hostname)\b|\$\{IFS\}"
)
_PIPE_TO_SHELL_RE = re.compile(r"\|\s*(?:ba|z)?sh\b")
_HTML_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
_COMMENT_COMMAND_RE = re.compile(
    r"\b(?:curl|wget|nc|nslookup|base64|print[e]nv)\b|\$\(|`"
)

# Build and test-runner configuration, which CI reads before running tests.
_BUILD_CONFIG_RE = re.compile(
    r"(?:^|/)(?:pyproject\.toml|tox\.ini|setup\.cfg|setup\.py|noxfile\.py|"
    r"requirements[\w.-]*\.txt|conftest\.py|pytest\.ini|\.pre-commit-config"
    r"\.yaml)$"
)
_REMOTE_DEPENDENCY_RE = re.compile(
    r"\bgit\+|@\s*(?:https?|git|ssh)://|--(?:extra-)?index-url|"
    r"--find-links|\bfind-links\s*="
)

# Characters that hide or reorder text. Written as escapes on purpose.
_BIDI_CHARS = frozenset(
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)
_ZERO_WIDTH_CHARS = frozenset("\u200b\u200c\u200d\u2060\ufeff")
_CODE_SUFFIXES = frozenset({
    ".py",
    ".pyi",
    ".sh",
    ".bash",
    ".yml",
    ".yaml",
    ".toml",
    ".cfg",
    ".ini",
    ".js",
    ".ts",
    ".json",
})
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt", ".html", ".ipynb"})
_TEXT_SUFFIXES = _CODE_SUFFIXES | _DOC_SUFFIXES | frozenset({".csv"})
_LOADABLE_BINARY_SUFFIXES = frozenset(
    {".pyc", ".pyo", ".so", ".dylib", ".dll", ".pth", ".whl", ".egg", ".zip"}
)
_PACKAGE_DIRS = ("langextract/", "tests/", "scripts/", ".github/")
_IMPORT_TIME_FILES_RE = re.compile(
    r"(?:^|/)(?:__init__\.py|conftest\.py|setup\.py|sitecustomize\.py)$"
)
_SYMLINK_MODE = "120000"

_MAX_EXCERPT = 120


@dataclasses.dataclass(frozen=True)
class Finding:
  """One result of the check."""

  severity: str
  rule: str
  message: str
  path: str = ""
  line: int = 0
  excerpt: str = ""


@dataclasses.dataclass
class FileDiff:
  """Added content for one file in a diff."""

  path: str
  is_new: bool = False
  is_binary: bool = False
  is_symlink: bool = False
  quoted_path: bool = False
  added: list[tuple[int, str]] = dataclasses.field(default_factory=list)

  @property
  def suffix(self) -> str:
    return pathlib.PurePosixPath(self.path).suffix.lower()

  @property
  def added_text(self) -> str:
    return "\n".join(text for _, text in self.added)

  @property
  def is_doc(self) -> bool:
    return self.suffix in _DOC_SUFFIXES or self.path.startswith("docs/")

  @property
  def runs_in_ci(self) -> bool:
    """Whether CI imports or collects this file without anyone calling it."""
    return bool(_IMPORT_TIME_FILES_RE.search(self.path)) or (
        self.path.startswith("tests/") and self.suffix == ".py"
    )


def normalize(text: str) -> str:
  """Returns NFKC-normalized text, the form Python uses for identifiers.

  Look-alike letters such as mathematical bold normalize to ASCII, so a call
  written with them still runs as the ASCII builtin and should match here.
  """
  return unicodedata.normalize("NFKC", text)


def printable(text: str) -> str:
  """Returns text with hidden characters shown as escapes and no backticks."""
  out = []
  for char in text:
    if char.isprintable() and char not in _ZERO_WIDTH_CHARS:
      out.append(char)
    else:
      out.append(f"\\u{ord(char):04x}")
  result = "".join(out).replace("`", "'")
  return result.replace("<", "&lt;").replace(">", "&gt;")


def safe_excerpt(text: str) -> str:
  """Returns a short, printable excerpt with URLs made non-clickable.

  Args:
    text: The raw text to excerpt.
  """
  excerpt = printable(text).strip().replace("://", "[:]//")
  excerpt = _CALLBACK_HOST_RE.sub(
      lambda m: m.group(0).replace(".", "[.]"), excerpt
  )
  if len(excerpt) > _MAX_EXCERPT:
    excerpt = excerpt[: _MAX_EXCERPT - 3] + "..."
  return excerpt


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_QUOTED_PATH = r'"(?:[^"\\]|\\.)*"'
_DIFF_GIT_RE = re.compile(
    rf"^diff --git (?:{_QUOTED_PATH}|a/.*?) (?P<b>{_QUOTED_PATH}|b/.*)$"
)


def _parse_path(token: str, prefix: str = "") -> tuple[str, bool]:
  """Returns a path from a diff header and whether git quoted it.

  Git C-quotes paths that contain quotes, backslashes, or control characters,
  even with `core.quotePath=false`.

  Args:
    token: The path as it appears in the header, possibly quoted.
    prefix: A leading `b/` to remove, if present.
  """
  quoted = len(token) >= 2 and token[0] == token[-1] == '"'
  if quoted:
    raw = codecs.escape_decode(token[1:-1].encode("utf-8"))[0]
    token = raw.decode("utf-8", errors="replace")
  if prefix and token.startswith(prefix):
    token = token[len(prefix) :]
  return token, quoted


def _diff_git_path(rest: str) -> tuple[str, bool]:
  """Returns the new path from the text after `diff --git `.

  Unrenamed files appear as `a/P b/P`; splitting at the midpoint keeps paths
  that contain " b/" intact. Renames also have `rename to` lines, which
  replace this value.
  """
  if rest.startswith("a/") and (len(rest) - 5) % 2 == 0:
    path = rest[2 : 2 + (len(rest) - 5) // 2]
    if rest == f"a/{path} b/{path}":
      return path, False
  match = _DIFF_GIT_RE.match("diff --git " + rest)
  if match:
    return _parse_path(match.group("b"), "b/")
  return rest, False


def parse_diff(diff_text: str) -> list[FileDiff]:
  """Parses `git diff` output into added lines per file.

  Header lines such as `+++ b/path` are only recognized before the first
  hunk, so added content that happens to start with `++` is still checked.
  Paths come from whichever headers are present (binary files, pure renames,
  and deletions have no `+++ b/` line). Quoted paths are decoded and marked,
  since unusual names can defeat path-based rules.

  Args:
    diff_text: Unified diff text produced with `--no-textconv --no-ext-diff`.

  Returns:
    One FileDiff per file in the diff, in order.
  """
  files: list[FileDiff] = []
  current: FileDiff | None = None
  in_hunk = False
  line_no = 0
  for raw in diff_text.split("\n"):
    if raw.startswith("diff --git "):
      path, quoted = _diff_git_path(raw[len("diff --git ") :])
      current = FileDiff(path=path, quoted_path=quoted)
      files.append(current)
      in_hunk = False
      line_no = 0
      continue
    if current is None:
      continue
    if raw.startswith("@@"):
      match = _HUNK_RE.match(raw)
      line_no = int(match.group(1)) if match else 0
      in_hunk = True
    elif not in_hunk:
      if raw.startswith("new file mode"):
        current.is_new = True
        current.is_symlink = raw.endswith(_SYMLINK_MODE)
      elif raw.startswith("new mode") and raw.endswith(_SYMLINK_MODE):
        current.is_symlink = True
      elif raw.startswith("Binary files ") or raw == "GIT binary patch":
        current.is_binary = True
      elif raw.startswith(("rename to ", "copy to ")):
        current.path, quoted = _parse_path(raw.split(" ", 2)[2])
        current.quoted_path = current.quoted_path or quoted
      elif raw.startswith("+++ ") and raw != "+++ /dev/null":
        current.path, quoted = _parse_path(raw[len("+++ ") :], "b/")
        current.quoted_path = current.quoted_path or quoted
    elif raw.startswith("+"):
      current.added.append((line_no, raw[1:]))
      line_no += 1
    elif raw.startswith(" "):
      line_no += 1
  return files


def _first_line(file: FileDiff, pattern: re.Pattern[str]) -> tuple[int, str]:
  for line_no, text in file.added:
    if pattern.search(normalize(text)):
      return line_no, text
  return file.added[0] if file.added else (0, "")


def _is_network_line(file: FileDiff, line: str) -> bool:
  """Whether a normalized added line reaches the network, for this file type.

  In Python sources, shell tools such as curl only count when the same line
  starts a process; otherwise they are words inside strings or comments.
  """
  if _PY_NETWORK_RE.search(line):
    return True
  if file.suffix in _PYTHON_SUFFIXES:
    return bool(_PROCESS_RE.search(line) and _SHELL_NETWORK_RE.search(line))
  return bool(_SHELL_NETWORK_RE.search(line))


def _network_lines(
    file: FileDiff, include_imports: bool = True
) -> list[tuple[int, str]]:
  """Returns added lines that reach a non-local network destination.

  Args:
    file: The file to inspect.
    include_imports: Whether a bare import of a network library counts.
  """
  return [
      (line_no, text)
      for line_no, text in file.added
      if _is_network_line(file, normalize(text))
      and not _LOCAL_TARGET_RE.search(text)
      and (include_imports or not _IMPORT_LINE_RE.match(text))
  ]


def _reads_env(file: FileDiff, text: str) -> bool:
  if _ENV_ACCESS_RE.search(text):
    return True
  return file.suffix not in _PYTHON_SUFFIXES and bool(
      _SHELL_ENV_RE.search(text)
  )


def _check_credential_egress(file: FileDiff) -> Iterable[Finding]:
  """Flags code that reads credentials and also uses the network."""
  if file.path.startswith(".github/"):
    # Workflows use secrets and curl as a matter of course;
    # _check_configuration covers them instead.
    return
  text = normalize(file.added_text)
  env_dump = bool(_ENV_DUMP_RE.search(text))
  reads_secret = env_dump or (
      _reads_env(file, text) and _SECRET_NAME_RE.search(text)
  )
  network = _network_lines(file)
  if not (reads_secret and network):
    return
  # Provider code legitimately reads an API key and calls an API, so only
  # raise the severity when another strong signal is present.
  signals = []
  if env_dump:
    signals.append("a copy of the whole environment")
  if _CALLBACK_HOST_RE.search(text):
    signals.append("a request-logging service")
  code_lines = [
      normalize(line)
      for _, line in file.added
      if not _IMPORT_LINE_RE.match(line)
  ]
  if any(
      _ENCODING_RE.search(line)
      and (_reads_env(file, line) or _is_network_line(file, line))
      for line in code_lines
  ):
    signals.append("encoding on the same line as the read or the request")
  if file.runs_in_ci:
    signals.append("a file that runs at import or test time")
  line_no, line = network[0]
  if signals and not (file.is_doc and not env_dump):
    yield Finding(
        HIGH,
        "credential-egress",
        "Reads credential-like environment variables and uses the network, "
        "together with "
        + ", ".join(signals)
        + ".",
        file.path,
        line_no,
        safe_excerpt(line),
    )
  else:
    yield Finding(
        MEDIUM,
        "credential-with-network",
        "Reads credential-like environment variables and uses the network; "
        "confirm the destination is the intended provider.",
        file.path,
        line_no,
        safe_excerpt(line),
    )


def _check_hosts(file: FileDiff) -> Iterable[Finding]:
  reported_ip = False
  for line_no, text in file.added:
    if _CALLBACK_HOST_RE.search(text):
      yield Finding(
          HIGH,
          "callback-service",
          "References a request-logging or callback service.",
          file.path,
          line_no,
          safe_excerpt(text),
      )
      return
    if not reported_ip and _RAW_IP_URL_RE.search(text):
      reported_ip = True
      yield Finding(
          MEDIUM,
          "raw-ip-url",
          "Adds a URL to a literal public IP address.",
          file.path,
          line_no,
          safe_excerpt(text),
      )


def _has_zero_width(text: str) -> bool:
  """True for zero-width characters outside emoji sequences and a BOM."""
  text = text.lstrip("\ufeff")
  for i, char in enumerate(text):
    if char not in _ZERO_WIDTH_CHARS:
      continue
    if char == "\u200d" and 0 < i < len(text) - 1:
      before, after = text[i - 1], text[i + 1]
      if not before.isascii() and not after.isascii():
        continue  # Joiner between emoji, as in family or flag sequences.
    return True
  return False


def _check_hidden_characters(file: FileDiff) -> Iterable[Finding]:
  for line_no, text in file.added:
    if "\x00" in text:
      kind = "NUL bytes"
    elif any(char in _BIDI_CHARS for char in text):
      kind = "bidirectional control characters"
    elif file.suffix in _CODE_SUFFIXES and _has_zero_width(text):
      kind = "zero-width characters"
    elif file.suffix == ".py" and _lookalike_identifiers(text):
      kind = "look-alike letters in code"
    else:
      continue
    yield Finding(
        MEDIUM,
        "hidden-characters",
        f"Adds {kind}, which can hide or reorder text in review.",
        file.path,
        line_no,
        safe_excerpt(text),
    )
    return


def _lookalike_identifiers(text: str) -> bool:
  """True when NFKC turns non-ASCII letters into ASCII letters."""
  if text.isascii():
    return False
  normalized = normalize(text)
  if normalized == text:
    return False
  return any(c.isascii() and c.isalpha() for c in set(normalized) - set(text))


def _check_file_shape(file: FileDiff) -> Iterable[Finding]:
  if file.quoted_path or " a/" in file.path or " b/" in file.path:
    yield Finding(
        MEDIUM,
        "unusual-path",
        "File path contains quotes, backslashes, control characters, or "
        "diff markers.",
        file.path,
    )
  if file.is_symlink:
    yield Finding(
        MEDIUM,
        "symlink",
        "Adds a symbolic link, which can point outside the repository.",
        file.path,
    )
  if file.is_binary:
    in_package = file.path.startswith(_PACKAGE_DIRS)
    if file.suffix in _CODE_SUFFIXES:
      yield Finding(
          HIGH,
          "binary-code-file",
          "A code file is treated as binary by git, so its contents do not "
          "appear in the diff.",
          file.path,
      )
    elif file.suffix in _TEXT_SUFFIXES:
      yield Finding(
          MEDIUM,
          "binary-text-file",
          "A text-type file is treated as binary by git, so its contents do "
          "not appear in the diff.",
          file.path,
      )
    elif file.suffix in _LOADABLE_BINARY_SUFFIXES or in_package:
      yield Finding(
          MEDIUM,
          "binary-file",
          "Adds a binary file that Python or pip could load.",
          file.path,
      )
  if (
      file.is_new
      and file.suffix in _CODE_SUFFIXES
      and file.added
      and file.added[0][1].startswith("\ufeff")
  ):
    yield Finding(
        MEDIUM,
        "byte-order-mark",
        "New code file starts with a byte order mark.",
        file.path,
        file.added[0][0],
    )
  crlf = sum(1 for _, text in file.added if text.endswith("\r"))
  if crlf >= 20 and crlf * 2 > len(file.added):
    yield Finding(
        LOW,
        "line-ending-churn",
        f"{crlf} added lines use CRLF endings, which can bury other changes.",
        file.path,
    )


def _check_code_patterns(file: FileDiff) -> Iterable[Finding]:
  for line_no, raw in file.added:
    text = normalize(raw)
    if _DYNAMIC_EXEC_RE.search(text) and _ENCODING_RE.search(text):
      yield Finding(
          HIGH,
          "decoded-execution",
          "Executes content that is decoded at runtime.",
          file.path,
          line_no,
          safe_excerpt(raw),
      )
      return
  text = normalize(file.added_text)
  if _DYNAMIC_EXEC_RE.search(text) and _ENCODING_RE.search(text):
    line_no, line = _first_line(file, _DYNAMIC_EXEC_RE)
    yield Finding(
        MEDIUM,
        "dynamic-execution-with-encoding",
        "Combines dynamic execution with decoding in the same file.",
        file.path,
        line_no,
        safe_excerpt(line),
    )
  if _IMPORT_TIME_FILES_RE.search(file.path):
    side_effects = [
        (line_no, raw)
        for line_no, raw in file.added
        if _is_network_line(file, normalize(raw))
        or _PROCESS_RE.search(normalize(raw))
    ]
    if side_effects:
      line_no, line = side_effects[0]
      yield Finding(
          MEDIUM,
          "import-time-side-effect",
          "Adds network or process calls to a file that runs on import or "
          "test collection.",
          file.path,
          line_no,
          safe_excerpt(line),
      )


def _check_configuration(file: FileDiff, maintainer: bool) -> Iterable[Finding]:
  """Checks automation and build configuration from non-maintainers."""
  if maintainer:
    return
  text = file.added_text
  if file.path.startswith(".github/"):
    if re.search(r"\b(?:pull_request_targe[t]|workflow_ru[n])\b", text):
      yield Finding(
          HIGH,
          "privileged-workflow",
          "Adds a workflow trigger that runs with repository privileges.",
          file.path,
      )
    else:
      yield Finding(
          MEDIUM,
          "infrastructure-change",
          "Changes repository automation; only maintainers may do this.",
          file.path,
      )
    return
  if file.path in (".gitattributes", ".gitmodules"):
    yield Finding(
        MEDIUM,
        "repository-config-change",
        "Changes how git presents or assembles the repository.",
        file.path,
    )
  if _BUILD_CONFIG_RE.search(file.path):
    match = _REMOTE_DEPENDENCY_RE.search(text)
    if match:
      line_no, line = _first_line(file, _REMOTE_DEPENDENCY_RE)
      yield Finding(
          HIGH,
          "remote-dependency",
          "Adds a dependency fetched from a URL or an alternate package "
          "index, which CI would install.",
          file.path,
          line_no,
          safe_excerpt(line),
      )
    else:
      yield Finding(
          MEDIUM,
          "build-config-change",
          "Changes build or test-runner configuration that CI reads before "
          "running tests.",
          file.path,
      )


def _check_across_files(files: Sequence[FileDiff]) -> Iterable[Finding]:
  """Correlates signals that a single change can split across files."""
  dump_file = next(
      (
          f
          for f in files
          if not f.path.startswith(".github/")
          and _ENV_DUMP_RE.search(normalize(f.added_text))
      ),
      None,
  )
  if dump_file is None:
    return
  for file in files:
    if file is dump_file or not file.runs_in_ci:
      continue
    if _network_lines(file, include_imports=False):
      yield Finding(
          HIGH,
          "credential-egress",
          f"Copies the whole environment in `{printable(dump_file.path)}` "
          "and uses the network in a file that runs at import or test time.",
          file.path,
      )
      return


def check_files(
    files: Sequence[FileDiff], maintainer: bool = False
) -> list[Finding]:
  """Runs all content rules over parsed diff files.

  Args:
    files: Parsed diff, one entry per file.
    maintainer: Whether the author may change repository automation.
  """
  findings: list[Finding] = []
  for file in files:
    findings.extend(_check_credential_egress(file))
    findings.extend(_check_hosts(file))
    findings.extend(_check_hidden_characters(file))
    findings.extend(_check_file_shape(file))
    findings.extend(_check_code_patterns(file))
    findings.extend(_check_configuration(file, maintainer))
  findings.extend(_check_across_files(files))
  return findings


def check_metadata(title: str, body: str, branch: str) -> list[Finding]:
  """Checks the PR title, description, and branch name.

  These fields reach shells and logs through automation, so they are
  checked for content that would run if expanded.
  """
  findings: list[Finding] = []
  fields = (("title", title), ("description", body), ("branch", branch))
  for field, value in fields:
    match = _COMMAND_SUBSTITUTION_RE.search(value)
    if match:
      start = max(match.start() - 20, 0)
      findings.append(
          Finding(
              HIGH,
              "command-substitution",
              f"PR {field} contains a shell command substitution.",
              excerpt=safe_excerpt(value[start : match.end() + 60]),
          )
      )
    if _CALLBACK_HOST_RE.search(value):
      findings.append(
          Finding(
              HIGH,
              "callback-service",
              f"PR {field} references a request-logging service.",
          )
      )
    if any(char in _BIDI_CHARS or char == "\x00" for char in value):
      findings.append(
          Finding(
              MEDIUM,
              "hidden-characters",
              f"PR {field} contains hidden control characters.",
          )
      )
    if _RAW_IP_URL_RE.search(value):
      findings.append(
          Finding(
              MEDIUM,
              "raw-ip-url",
              f"PR {field} links to a literal public IP address.",
          )
      )
    if _PIPE_TO_SHELL_RE.search(value):
      findings.append(
          Finding(LOW, "pipe-to-shell", f"PR {field} pipes output to a shell.")
      )
  for comment in _HTML_COMMENT_RE.findall(body):
    if _COMMENT_COMMAND_RE.search(comment):
      findings.append(
          Finding(
              MEDIUM,
              "hidden-comment-command",
              "PR description has an HTML comment containing a command.",
              excerpt=safe_excerpt(comment),
          )
      )
      break
  return findings


def git_diff(base: str, head: str, repo_dir: str = ".") -> str:
  """Returns the diff without running any repository-defined filters."""
  result = subprocess.run(
      [
          "git",
          "-C",
          repo_dir,
          "-c",
          "core.quotePath=false",
          "diff",
          "--no-color",
          "--no-ext-diff",
          "--no-textconv",
          "--find-renames",
          "--unified=0",
          f"{base}...{head}",
      ],
      check=True,
      capture_output=True,
  )
  return result.stdout.decode("utf-8", errors="replace")


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
  return sorted(
      findings, key=lambda f: (_SEVERITY_ORDER[f.severity], f.path, f.line)
  )


def render_summary(
    findings: Sequence[Finding], brief: bool = False, repro: str = ""
) -> str:
  """Renders findings as Markdown.

  Args:
    findings: Sorted findings.
    brief: Report counts only. CI uses this because job summaries are
      public; maintainers reproduce the details locally.
    repro: Command to show in brief mode for reproducing the details.
  """
  counts = {sev: 0 for sev in _SEVERITY_ORDER}
  for finding in findings:
    counts[finding.severity] += 1
  lines = ["### PR security check", ""]
  if not findings:
    lines.append("No findings.")
  elif counts[HIGH]:
    lines.append(
        f"Review required: {counts[HIGH]} high, {counts[MEDIUM]} medium, "
        f"{counts[LOW]} low."
    )
  else:
    lines.append(
        f"Passed with notes: {counts[MEDIUM]} medium, {counts[LOW]} low."
    )
  if brief:
    if findings and repro:
      lines += [
          "",
          "Maintainers can see details locally after fetching the commit:",
          (
              f"`{repro}`, adding `--title-file`, `--body-file`, and `--branch`"
              " to include the pull request text."
          ),
      ]
    return "\n".join(lines) + "\n"
  lines.append("")
  for finding in findings:
    path = printable(finding.path)
    location = f"{path}:{finding.line}" if finding.line else path
    location = f" `{location}`" if location else ""
    lines.append(
        f"- **{finding.severity}** `{finding.rule}`{location}: "
        f"{finding.message}"
    )
    if finding.excerpt:
      lines.append(f"  - excerpt: `{finding.excerpt}`")
  return "\n".join(lines) + "\n"


def _read_optional(path: str | None) -> str:
  if not path:
    return ""
  return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the check from the command line; see the module docstring."""
  parser = argparse.ArgumentParser(
      description=__doc__.split("\n", maxsplit=1)[0]
  )
  parser.add_argument("--base", required=True)
  parser.add_argument("--head", required=True)
  parser.add_argument("--repo-dir", default=".")
  parser.add_argument("--title-file")
  parser.add_argument("--body-file")
  parser.add_argument("--branch", default="")
  parser.add_argument(
      "--maintainer",
      action="store_true",
      help="The author is a maintainer; skips repository automation rules.",
  )
  parser.add_argument("--json-out")
  parser.add_argument("--summary-out")
  parser.add_argument(
      "--brief",
      action="store_true",
      help="Report counts only, for public job summaries.",
  )
  parser.add_argument(
      "--fail-on-high",
      action="store_true",
      help=f"Exit with status {EXIT_HIGH_FINDINGS} on a high finding.",
  )
  args = parser.parse_args(argv)

  files = parse_diff(git_diff(args.base, args.head, args.repo_dir))
  findings = check_files(files, maintainer=args.maintainer)
  findings += check_metadata(
      _read_optional(args.title_file),
      _read_optional(args.body_file),
      args.branch,
  )
  findings = sort_findings(findings)

  repro = (
      "python .github/scripts/pr_security_check.py "
      f"--base {args.base} --head {args.head}"
  )
  summary = render_summary(findings, brief=args.brief, repro=repro)
  if args.summary_out:
    with open(args.summary_out, "a", encoding="utf-8") as out:
      out.write(summary)
  else:
    sys.stdout.write(summary)
  if args.json_out:
    payload = {
        "findings": [dataclasses.asdict(f) for f in findings],
        "high": sum(f.severity == HIGH for f in findings),
        "medium": sum(f.severity == MEDIUM for f in findings),
        "low": sum(f.severity == LOW for f in findings),
    }
    pathlib.Path(args.json_out).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
  has_high = any(f.severity == HIGH for f in findings)
  return EXIT_HIGH_FINDINGS if (args.fail_on_high and has_high) else 0


if __name__ == "__main__":
  sys.exit(main())
