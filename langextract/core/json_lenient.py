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

"""Lenient JSON parsing for LLM outputs.

Primary use case: handle unescaped control characters (e.g. literal newlines or
tabs inside strings) sometimes emitted by local models.
"""

from __future__ import annotations

import json
from typing import Any

from langextract.core import text_sanitizer


def loads(text: str) -> Any:
  """Parse JSON, allowing common LLM output quirks.

  Falls back to a decoder with strict=False, which accepts unescaped control
  characters inside strings (e.g. raw newlines).
  """
  sanitized = text_sanitizer.sanitize_for_parsing(text)

  try:
    return json.loads(sanitized)
  except json.JSONDecodeError:
    decoder = json.JSONDecoder(strict=False)
    try:
      return decoder.decode(sanitized)
    except json.JSONDecodeError:
      # If there is leading non-JSON text, try to locate the first object/array.
      starts = [i for i, c in enumerate(sanitized) if c in "{["]
      if not starts:
        raise
      start = min(starts)
      obj, _ = decoder.raw_decode(sanitized, start)
      return obj

