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

"""JSON serialization helpers.

These helpers provide a *stable*, best-effort conversion of common Python
objects (dataclasses, enums, pydantic models, sets, etc.) into JSON-serializable
structures.

Primary use case: building deterministic cache keys (hashing request payloads)
without crashing on non-JSON-native types.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime
import enum
import json
import pathlib
import re
from typing import Any

_MEMORY_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]+")


def dumps_canonical(obj: Any) -> str:
  """Serialize `obj` to a canonical JSON string.

  The output is deterministic across runs for supported types:
  - Dict keys are sorted
  - Sets are converted to sorted lists (by canonical JSON of elements)
  - Dataclasses and enums are converted into JSON-serializable forms

  Args:
    obj: Any Python object.

  Returns:
    Canonical JSON string (UTF-8 safe, no ASCII escaping).
  """
  return json.dumps(
      to_jsonable(obj),
      sort_keys=True,
      ensure_ascii=False,
      separators=(",", ":"),
      allow_nan=False,
  )


def to_jsonable(obj: Any) -> Any:
  """Convert `obj` into a JSON-serializable structure."""
  if obj is None or isinstance(obj, (bool, int, float, str)):
    return obj

  # Dataclasses
  if dataclasses.is_dataclass(obj):
    return to_jsonable(dataclasses.asdict(obj))

  # Enums
  if isinstance(obj, enum.Enum):
    return {
        "__enum__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
        "name": obj.name,
    }

  # Paths
  if isinstance(obj, pathlib.Path):
    return str(obj)

  # Bytes-like
  if isinstance(obj, (bytes, bytearray, memoryview)):
    return {"__bytes__": base64.b64encode(bytes(obj)).decode("ascii")}

  # Datetime-like
  if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
    return obj.isoformat()

  # Sequences
  if isinstance(obj, (list, tuple)):
    return [to_jsonable(v) for v in obj]

  # Sets (order-independent)
  if isinstance(obj, (set, frozenset)):
    items = [to_jsonable(v) for v in obj]
    items.sort(key=_canonical_sort_key)
    return items

  # Mappings
  if isinstance(obj, dict):
    # JSON requires string keys. For non-strings, use canonical JSON of the key.
    out: dict[str, Any] = {}
    for k, v in obj.items():
      if isinstance(k, str):
        key = k
      else:
        key = dumps_canonical(k)
      out[key] = to_jsonable(v)
    return out

  # Pydantic v2 models (and similarly shaped objects).
  model_dump = getattr(obj, "model_dump", None)
  if callable(model_dump):
    try:
      return to_jsonable(model_dump(mode="json"))
    except TypeError:
      return to_jsonable(model_dump())

  # Pydantic v1 models / other libs.
  to_dict = getattr(obj, "to_dict", None)
  if callable(to_dict):
    return to_jsonable(to_dict())

  dict_method = getattr(obj, "dict", None)
  if callable(dict_method):
    return to_jsonable(dict_method())

  # Fall back to a stable-ish repr (strip memory addresses).
  return _stable_repr(obj)


def _canonical_sort_key(obj: Any) -> str:
  """Key function for deterministic sorting of set elements."""
  return json.dumps(
      obj,
      sort_keys=True,
      ensure_ascii=False,
      separators=(",", ":"),
      default=_stable_repr,
      allow_nan=False,
  )


def _stable_repr(obj: Any) -> str:
  """Return a representation that avoids non-deterministic memory addresses."""
  return _MEMORY_ADDRESS_RE.sub("0x...", repr(obj))

