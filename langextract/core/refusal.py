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

"""Heuristics for detecting refusal / non-structured model outputs."""

from __future__ import annotations

import re

_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
_FENCE_LINE_RE = re.compile(r"^\s*```.*$", re.MULTILINE)

_REFUSAL_PHRASES = (
    "no entities",
    "no entity",
    "nothing to extract",
    "nothing found",
    "none found",
    "no relevant",
    "no relevant entities",
    "no relevant information",
    "not found",
    "not mentioned",
    "cannot find",
    "can't find",
    "unable to find",
    "i cannot",
    "i can't",
    "i am sorry",
    "i'm sorry",
    "sorry",
    "unable to comply",
    "cannot comply",
    "can't comply",
    "refuse",
    "refused",
)

_REFUSAL_EXACT = frozenset({"none", "no", "n/a", "na", "null"})


def _strip_common_wrappers(text: str) -> str:
  text = _THINK_TAG_RE.sub("", text)
  text = _FENCE_LINE_RE.sub("", text)
  return text.strip()


def looks_like_refusal(text: str) -> bool:
  """Best-effort check for refusal / non-JSON/YAML outputs.

  This is intentionally conservative: it only returns True for short,
  plain-language responses that resemble common refusal / no-entity replies.
  """
  cleaned = _strip_common_wrappers(text)
  if not cleaned:
    return False

  lowered = cleaned.lower().strip()
  if lowered in _REFUSAL_EXACT:
    return True

  # If the model is attempting structured output, don't treat it as refusal.
  if any(ch in cleaned for ch in ("{", "}", "[", "]")):
    return False

  for phrase in _REFUSAL_PHRASES:
    if phrase in lowered:
      return True

  return False
