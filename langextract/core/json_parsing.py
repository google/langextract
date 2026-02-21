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

"""Helpers for lenient JSON extraction from mixed model outputs."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any


def parse_last_json(
    text: str, *, accept: Callable[[Any], bool] | None = None
) -> Any:
  """Parse the last JSON object/array found in `text`.

  This is useful when a model echoes few-shot examples before emitting the
  actual answer; selecting the last JSON payload avoids parsing the example.

  Args:
    text: Model output that may contain leading/trailing non-JSON text or
      multiple JSON objects/arrays.

  Returns:
    Parsed Python object (dict/list/etc.).

  Raises:
    json.JSONDecodeError: If no parseable JSON object/array is found.
  """
  decoder = json.JSONDecoder()
  starts = [i for i, c in enumerate(text) if c in "{["]
  last_err: json.JSONDecodeError | None = None
  for start in reversed(starts):
    try:
      obj, _ = decoder.raw_decode(text, start)
      if accept is not None and not accept(obj):
        continue
      return obj
    except json.JSONDecodeError as e:
      last_err = e
      continue
  if last_err is not None:
    raise last_err
  raise json.JSONDecodeError("No JSON object found", text, 0)

