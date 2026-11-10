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

"""Centralized format handler for prompts and parsing."""

from __future__ import annotations

import json
import re
import string
from typing import Mapping, Sequence
import warnings

import yaml

from langextract.core import data
from langextract.core import exceptions

ExtractionValueType = str | int | float | dict | list | None

_JSON_FORMAT = "json"
_YAML_FORMAT = "yaml"
_YML_FORMAT = "yml"

_FENCE_OPEN_LINE_RE = re.compile(r"^[ \t]{0,3}(`{3,})([A-Za-z0-9_+-]+)?[ \t]*$")
_FENCE_CLOSE_LINE_RE = re.compile(r"^[ \t]{0,3}(`{3,})[ \t]*$")
_LANG_TAG_PREFIX_RE = re.compile(r"^[A-Za-z0-9_+-]+")
_FENCE_CLOSE_WITH_TEXT_RE = re.compile(r"^[ \t]{0,3}(`{3,})[ \t]+\S")
_LANG_TAG_CHARS = string.ascii_letters + string.digits + "_+-"

# Language tag, fence length, and body lines of a fence that was never closed.
_UnclosedBlock = tuple[str | None, int, list[str]]

_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)


def _parse_single_line_fence(line: str) -> tuple[str | None, str] | None:
  """Parses a single-line fenced block in O(n) time.

  The block may follow prose on the same line ('Result: ```json {...}```').
  """
  fence_start = line.find("```")
  if fence_start < 0:
    return None
  # A fence indented by 4+ spaces is nested content, not a block of its own.
  if fence_start > 3 and not line[:fence_start].strip():
    return None
  content = line[fence_start:].rstrip(" \t\r\n")
  open_ticks = len(content) - len(content.lstrip("`"))
  if open_ticks < 3:
    return None
  after_open = content[open_ticks:]
  close_ticks = len(after_open) - len(after_open.rstrip("`"))
  if close_ticks < open_ticks or len(after_open) == close_ticks:
    return None
  inner = after_open[:-close_ticks]
  lang_match = _LANG_TAG_PREFIX_RE.match(inner)
  if lang_match is not None:
    remainder = inner[lang_match.end() :]
    if not remainder or remainder[0] in " \t{[":
      return lang_match.group(0), remainder
  return None, inner


def _parse_inline_opener(line: str) -> tuple[int, str | None] | None:
  """Parses an opening fence that follows prose on the same line.

  Args:
    line: A line without its line ending, such as 'Result: ```json'.

  Returns:
    The fence length and language tag, or None if the line does not end with
    an opening fence preceded by other text.
  """
  content = line.rstrip(" \t")
  without_lang = content.rstrip(_LANG_TAG_CHARS)
  ticks = len(without_lang) - len(without_lang.rstrip("`"))
  if ticks < 3:
    return None
  prose = without_lang[:-ticks]
  if not prose.strip() or "```" in prose:
    return None
  return ticks, content[len(without_lang) :] or None


def _scan_fenced_blocks(
    lines: Sequence[str], *, allow_inline_opener: bool
) -> tuple[list[tuple[str | None, str]], _UnclosedBlock | None]:
  """Collects closed `(lang, body)` blocks and any block left unclosed."""
  blocks: list[tuple[str | None, str]] = []
  open_ticks: int | None = None
  open_lang: str | None = None
  body_lines: list[str] = []

  for line in lines:
    line_no_eol = line.rstrip("\r\n")
    if open_ticks is None:
      # Only treat multi-line fence headers (followed by another line) here;
      # single-line fences are handled by _parse_single_line_fence.
      if line_no_eol == line:
        continue
      open_match = _FENCE_OPEN_LINE_RE.match(line_no_eol)
      if open_match is not None:
        open_ticks = len(open_match.group(1))
        open_lang = open_match.group(2)
        body_lines = []
      elif allow_inline_opener:
        inline_opener = _parse_inline_opener(line_no_eol)
        if inline_opener is not None:
          open_ticks, open_lang = inline_opener
          body_lines = []
    else:
      close_match = _FENCE_CLOSE_LINE_RE.match(line_no_eol)
      if close_match is not None and len(close_match.group(1)) >= open_ticks:
        blocks.append((open_lang, "".join(body_lines)))
        open_ticks = None
      else:
        body_lines.append(line)

  if open_ticks is None:
    return blocks, None
  return blocks, (open_lang, open_ticks, body_lines)


def _recover_unclosed_block(
    unclosed: _UnclosedBlock,
) -> tuple[str | None, str] | None:
  """Closes a block whose closing fence shares a line with other text."""
  open_lang, open_ticks, body_lines = unclosed

  # Closing fence at the end of the final non-blank content line (requiring a
  # non-whitespace prefix so 4+-space indented fences inside an unclosed block
  # are never treated as closing the outer fence).
  last_idx = len(body_lines) - 1
  while last_idx >= 0 and not body_lines[last_idx].strip():
    last_idx -= 1
  if last_idx >= 0:
    last_trimmed = body_lines[last_idx].rstrip(" \t\r\n")
    trailing_ticks = len(last_trimmed) - len(last_trimmed.rstrip("`"))
    prefix = last_trimmed[:-trailing_ticks] if trailing_ticks else ""
    if trailing_ticks >= open_ticks and prefix.strip():
      return open_lang, "".join(body_lines[:last_idx] + [prefix])

  # Closing fence followed by trailing text, such as '``` done'. Whitespace
  # must separate the two so a nested '```python' is not taken as a close.
  for idx, line in enumerate(body_lines):
    close_match = _FENCE_CLOSE_WITH_TEXT_RE.match(line)
    if close_match is not None and len(close_match.group(1)) >= open_ticks:
      return open_lang, "".join(body_lines[:idx])
  return None


def _extract_fenced_blocks(text: str) -> list[tuple[str | None, str]]:
  """Extracts `(lang, body)` fenced code blocks in linear time.

  Fences on their own lines are preferred. Fences that share a line with other
  text are only considered when no such block exists, so backticks inside
  extraction values do not split a well-formed block.
  """
  lines = text.splitlines(keepends=True)
  for allow_inline_opener in (False, True):
    blocks, unclosed = _scan_fenced_blocks(
        lines, allow_inline_opener=allow_inline_opener
    )
    if blocks:
      return blocks
    if unclosed is not None:
      recovered = _recover_unclosed_block(unclosed)
      if recovered is not None:
        return [recovered]

  # Compact single-line fenced blocks (```json {...}```).
  single_line_blocks = []
  for line in lines:
    parsed = _parse_single_line_fence(line)
    if parsed is not None:
      single_line_blocks.append(parsed)
  return single_line_blocks


def _normalize_format_type(
    format_type: data.FormatType | str | None,
) -> data.FormatType:
  """Normalize a FormatType enum, string, or None into a FormatType enum."""
  if format_type is None:
    return data.FormatType.JSON
  if isinstance(format_type, data.FormatType):
    return format_type
  if isinstance(format_type, str):
    return data.FormatType(format_type.strip().lower())
  if hasattr(format_type, "value") and isinstance(format_type.value, str):
    return data.FormatType(format_type.value.strip().lower())
  raise TypeError(
      "format_type must be a FormatType enum or string ('json', 'yaml'), "
      f"got {type(format_type).__name__}"
  )


class FormatHandler:
  """Handles all format-specific logic for prompts and parsing.

  This class centralizes format handling for JSON and YAML outputs,
  including fence detection, wrapper management, and parsing.

  Attributes:
    format_type: The output format ('json' or 'yaml').
    use_wrapper: Whether to wrap extractions in a container dictionary.
    wrapper_key: The key name for the container dictionary (e.g., creates
      {"extractions": [...]} instead of just [...]).
    use_fences: Whether to use code fences in formatted output.
    attribute_suffix: Suffix for attribute fields in extractions.
    strict_fences: Whether to enforce strict fence validation.
    allow_top_level_list: Whether to allow top-level lists in parsing.
  """

  def __init__(
      self,
      format_type: data.FormatType | str | None = data.FormatType.JSON,
      use_wrapper: bool = True,
      wrapper_key: str | None = None,
      use_fences: bool = True,
      attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
      strict_fences: bool = False,
      allow_top_level_list: bool = True,
  ) -> None:
    """Initialize format handler.

    Args:
      format_type: Output format type enum or string ('json', 'yaml').
      use_wrapper: Whether to wrap extractions in a container dictionary.
        True: {"extractions": [...]}, False: [...]
      wrapper_key: Key name for the container dictionary. When use_wrapper=True:
        - If None: defaults to EXTRACTIONS_KEY ("extractions")
        - If provided: uses the specified key as container
        When use_wrapper=False, this parameter is ignored.
      use_fences: Whether to use ```json or ```yaml fences.
      attribute_suffix: Suffix for attribute fields.
      strict_fences: If True, require exact fence format. If False, be lenient
        with model output variations.
      allow_top_level_list: Allow top-level list when not strict and
        wrapper not required.
    """
    self.format_type = _normalize_format_type(format_type)
    self.use_wrapper = use_wrapper
    if use_wrapper:
      self.wrapper_key = (
          wrapper_key if wrapper_key is not None else data.EXTRACTIONS_KEY
      )
    else:
      self.wrapper_key = None
    self.use_fences = use_fences
    self.attribute_suffix = attribute_suffix
    self.strict_fences = strict_fences
    self.allow_top_level_list = allow_top_level_list

  def __repr__(self) -> str:
    return (
        "FormatHandler("
        f"format_type={self.format_type!r}, use_wrapper={self.use_wrapper}, "
        f"wrapper_key={self.wrapper_key!r}, use_fences={self.use_fences}, "
        f"attribute_suffix={self.attribute_suffix!r}, "
        f"strict_fences={self.strict_fences}, "
        f"allow_top_level_list={self.allow_top_level_list})"
    )

  def format_extraction_example(
      self, extractions: list[data.Extraction]
  ) -> str:
    """Format extractions for a prompt example.

    Args:
      extractions: List of extractions to format

    Returns:
      Formatted string for the prompt
    """
    items = [
        {
            ext.extraction_class: ext.extraction_text,
            f"{ext.extraction_class}{self.attribute_suffix}": (
                ext.attributes or {}
            ),
        }
        for ext in extractions
    ]

    if self.use_wrapper and self.wrapper_key:
      payload = {self.wrapper_key: items}
    else:
      payload = items

    if self.format_type == data.FormatType.YAML:
      formatted = yaml.safe_dump(
          payload, default_flow_style=False, sort_keys=False
      )
    else:
      formatted = json.dumps(payload, indent=2, ensure_ascii=False)

    return self._add_fences(formatted) if self.use_fences else formatted

  def parse_output(
      self, text: str, *, strict: bool | None = None
  ) -> Sequence[Mapping[str, ExtractionValueType]]:
    """Parse model output to extract data.

    Args:
      text: Raw model output.
      strict: If True, enforce strict schema validation. When strict is
        True, always require wrapper object if wrapper_key is configured,
        reject top-level lists even if allow_top_level_list is True, and
        enforce exact format compliance.

    Returns:
      List of extraction dictionaries.

    Raises:
      FormatError: Various subclasses for specific parsing failures.
    """
    if not text:
      raise exceptions.FormatParseError("Empty or invalid input string.")

    content = self._extract_content(text)

    try:
      parsed = self._parse_with_fallback(content, strict)
    except (yaml.YAMLError, json.JSONDecodeError) as e:
      msg = (
          f"Failed to parse {self.format_type.value.upper()} content:"
          f" {str(e)[:200]}"
      )
      raise exceptions.FormatParseError(msg) from e

    if parsed is None:
      if self.use_wrapper:
        raise exceptions.FormatParseError(
            f"Content must be a mapping with an '{self.wrapper_key}' key."
        )
      else:
        raise exceptions.FormatParseError(
            "Content must be a list of extractions or a dict."
        )

    require_wrapper = self.wrapper_key is not None and (
        self.use_wrapper or bool(strict)
    )

    if isinstance(parsed, dict):
      if require_wrapper:
        if self.wrapper_key not in parsed:
          raise exceptions.FormatParseError(
              f"Content must contain an '{self.wrapper_key}' key."
          )
        items = parsed[self.wrapper_key]
      else:
        if data.EXTRACTIONS_KEY in parsed:
          items = parsed[data.EXTRACTIONS_KEY]
        elif self.wrapper_key and self.wrapper_key in parsed:
          items = parsed[self.wrapper_key]
        else:
          items = [parsed]
    elif isinstance(parsed, list):
      if require_wrapper and (strict or not self.allow_top_level_list):
        raise exceptions.FormatParseError(
            f"Content must be a mapping with an '{self.wrapper_key}' key."
        )
      if strict and self.use_wrapper:
        raise exceptions.FormatParseError(
            "Strict mode requires a wrapper object."
        )
      if not self.allow_top_level_list:
        raise exceptions.FormatParseError("Top-level list is not allowed.")
      # Some models return [...] instead of {"extractions": [...]}.
      items = parsed
    else:
      raise exceptions.FormatParseError(
          f"Expected list or dict, got {type(parsed)}"
      )

    if not isinstance(items, list):
      raise exceptions.FormatParseError(
          "The extractions must be a sequence (list) of mappings."
      )

    for item in items:
      if not isinstance(item, dict):
        raise exceptions.FormatParseError(
            "Each item in the sequence must be a mapping."
        )
      for k in item.keys():
        if not isinstance(k, str):
          raise exceptions.FormatParseError(
              "All extraction keys must be strings (got a non-string key)."
          )

    return items

  def _add_fences(self, content: str) -> str:
    """Add code fences around content."""
    fence_type = self.format_type.value
    return f"```{fence_type}\n{content.strip()}\n```"

  def _is_valid_language_tag(
      self, lang: str | None, valid_tags: dict[data.FormatType, set[str]]
  ) -> bool:
    """Check if language tag is valid for the format type."""
    if lang is None:
      return True
    tag = lang.strip().lower()
    return tag in valid_tags.get(self.format_type, set())

  def _parse_with_fallback(self, content: str, strict: bool):
    """Parse content, retrying without <think> tags on failure."""
    try:
      if self.format_type == data.FormatType.YAML:
        return yaml.safe_load(content)
      return json.loads(content)
    except (yaml.YAMLError, json.JSONDecodeError):
      if strict:
        raise
      # Reasoning models (DeepSeek-R1, QwQ) emit <think> tags before JSON.
      if _THINK_TAG_RE.search(content):
        stripped = _THINK_TAG_RE.sub("", content).strip()
        if self.format_type == data.FormatType.YAML:
          return yaml.safe_load(stripped)
        return json.loads(stripped)
      raise

  def _extract_content(self, text: str) -> str:
    """Extract content from text, handling fences if configured.

    Args:
      text: Input text that may contain fenced blocks

    Returns:
      Extracted content

    Raises:
      FormatParseError: When fences required but not found or multiple
        blocks found.
    """
    if not self.use_fences:
      return text.strip()

    blocks = _extract_fenced_blocks(text)

    valid_tags = {
        data.FormatType.YAML: {_YAML_FORMAT, _YML_FORMAT},
        data.FormatType.JSON: {_JSON_FORMAT},
    }

    candidates = [
        body
        for lang, body in blocks
        if self._is_valid_language_tag(lang, valid_tags)
    ]

    if self.strict_fences:
      if len(candidates) != 1:
        if len(candidates) == 0:
          raise exceptions.FormatParseError(
              "Input string does not contain valid fence markers."
          )
        else:
          raise exceptions.FormatParseError(
              "Multiple fenced blocks found. Expected exactly one."
          )
      return candidates[0].strip()

    if len(candidates) == 1:
      return candidates[0].strip()
    elif len(candidates) > 1:
      raise exceptions.FormatParseError(
          "Multiple fenced blocks found. Expected exactly one."
      )

    if blocks:
      if not self.strict_fences and len(blocks) == 1:
        return blocks[0][1].strip()
      raise exceptions.FormatParseError(
          f"No {self.format_type.value} code block found."
      )

    return text.strip()

  # ---- Backward compatibility methods (to be removed in v2.0.0) ----

  _LEGACY_FORMAT_KEYS = frozenset({
      "fence_output",
      "format_type",
      "strict_fences",
      "require_extractions_key",
      "extraction_attributes_suffix",
      "attribute_suffix",
      "format_handler",
  })

  @classmethod
  def from_resolver_params(
      cls,
      *,
      resolver_params: dict | None,
      base_format_type: data.FormatType,
      base_use_fences: bool,
      base_attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
      base_use_wrapper: bool = True,
      base_wrapper_key: str | None = data.EXTRACTIONS_KEY,
      warn_on_legacy: bool = True,
  ) -> tuple[FormatHandler, dict]:
    """Create FormatHandler from resolver_params with legacy support.

    This method handles backward compatibility for legacy resolver parameters
    and will be removed in v2.0.0.

    Args:
      resolver_params: May contain legacy keys or a 'format_handler'.
      base_format_type: Default format when not overridden.
      base_use_fences: Default fence usage from the model.
      base_attribute_suffix: Default attribute suffix.
      base_use_wrapper: Default wrapper behavior.
      base_wrapper_key: Default wrapper key.
      warn_on_legacy: If True, emit DeprecationWarnings.

    Returns:
      (format_handler, remaining_resolver_params)
    """
    rp = dict(resolver_params or {})

    if rp.get("format_handler") is not None:
      handler = rp.pop("format_handler")
      for k in list(rp.keys()):
        if k in cls._LEGACY_FORMAT_KEYS:
          rp.pop(k, None)
      return handler, rp

    kwargs = {
        "format_type": base_format_type,
        "use_fences": base_use_fences,
        "attribute_suffix": base_attribute_suffix,
        "use_wrapper": base_use_wrapper,
        "wrapper_key": base_wrapper_key if base_use_wrapper else None,
    }

    mapping = {
        "fence_output": "use_fences",
        "format_type": "format_type",
        "strict_fences": "strict_fences",
        "require_extractions_key": "use_wrapper",
        "extraction_attributes_suffix": "attribute_suffix",
        "attribute_suffix": "attribute_suffix",
    }

    used_legacy = []
    for legacy_key, fh_key in mapping.items():
      if legacy_key in rp and rp[legacy_key] is not None:
        val = rp.pop(legacy_key)
        if fh_key == "format_type":
          val = _normalize_format_type(val)
        kwargs[fh_key] = val
        used_legacy.append(legacy_key)

    if warn_on_legacy and used_legacy:
      warnings.warn(
          "Resolver legacy params are deprecated and will be removed in"
          f" v2.0.0: {used_legacy}. Pass a FormatHandler explicitly via"
          " `resolver_params={'format_handler': FormatHandler(...)}` or rely"
          " on defaults configured by the model.",
          DeprecationWarning,
          stacklevel=3,
      )

    handler = cls(**kwargs)
    return handler, rp

  @classmethod
  def from_kwargs(cls, **kwargs) -> FormatHandler:
    """Create FormatHandler from legacy resolver keyword arguments.

    This method will be removed in v2.0.0.

    Args:
      **kwargs: Legacy parameters like fence_output, format_type, etc.

    Returns:
      FormatHandler configured with legacy parameters.
    """
    legacy_params = {
        "fence_output",
        "format_type",
        "strict_fences",
        "require_extractions_key",
    }
    used_legacy = legacy_params.intersection(kwargs.keys())

    if used_legacy:
      warnings.warn(
          f"Using legacy Resolver parameters {used_legacy} is deprecated. "
          "Please use FormatHandler directly. "
          "This compatibility layer will be removed in v2.0.0.",
          DeprecationWarning,
          stacklevel=3,
      )

    fence_output = kwargs.pop("fence_output", True)
    format_type = kwargs.pop("format_type", None)
    strict_fences = kwargs.pop("strict_fences", False)
    require_extractions_key = kwargs.pop("require_extractions_key", True)
    attribute_suffix = kwargs.pop("attribute_suffix", data.ATTRIBUTE_SUFFIX)

    if format_type is None:
      format_type = data.FormatType.JSON
    elif hasattr(format_type, "value"):
      pass
    else:
      format_type = (
          data.FormatType.JSON
          if str(format_type).lower() == "json"
          else data.FormatType.YAML
      )

    return cls(
        format_type=format_type,
        use_wrapper=require_extractions_key,
        wrapper_key=data.EXTRACTIONS_KEY if require_extractions_key else None,
        use_fences=fence_output,
        strict_fences=strict_fences,
        attribute_suffix=attribute_suffix,
    )
