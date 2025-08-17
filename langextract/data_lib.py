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

"""Library for data conversion between AnnotatedDocument and JSON."""
from __future__ import annotations

import dataclasses
import enum
import numbers
from typing import Any, Iterable, Mapping

from langextract.core import data
from langextract.core import tokenizer


def enum_asdict_factory(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
  """Custom dict_factory for dataclasses.asdict.

  Recursively converts dataclass instances, converts enum values to their
  underlying values, converts integral numeric types to int, and skips any
  field whose name starts with an underscore.

  Args:
    items: An iterable of (key, value) pairs from fields of a dataclass.

  Returns:
    A mapping of field names to their values, with special handling for
    dataclasses, enums, and numeric types.
  """
  result: dict[str, Any] = {}
  for key, value in items:
    # Skip internal fields.
    if key.startswith("_"):
      continue
    if dataclasses.is_dataclass(value):
      result[key] = dataclasses.asdict(value, dict_factory=enum_asdict_factory)
    elif isinstance(value, enum.Enum):
      result[key] = value.value
    elif isinstance(value, numbers.Integral) and not isinstance(value, bool):
      result[key] = int(value)
    else:
      result[key] = value
  return result


def annotated_document_to_dict(
    adoc: data.AnnotatedDocument | None,
) -> dict[str, Any]:
  """Converts an AnnotatedDocument into a Python dict.

  This function converts an AnnotatedDocument object into a Python dict, making
  it easier to serialize or deserialize the document. Enum values and NumPy
  integers are converted to their underlying values, while other data types are
  left unchanged. Private fields with an underscore prefix are not included in
  the output.

  Args:
    adoc: The AnnotatedDocument object to convert.

  Returns:
    A Python dict representing the AnnotatedDocument.
  """

  if not adoc:
    return {}

  result = dataclasses.asdict(adoc, dict_factory=enum_asdict_factory)

  result["document_id"] = adoc.document_id

  return result


def dict_to_annotated_document(
    adoc_dic: Mapping[str, Any],
) -> data.AnnotatedDocument:
  """Converts a Python dict back to an AnnotatedDocument.

  Args:
    adoc_dic: A Python dict representing an AnnotatedDocument.

  Returns:
    An AnnotatedDocument object.
  """
  if not adoc_dic:
    return data.AnnotatedDocument()

  for extractions in adoc_dic.get("extractions", []):
    token_int = extractions.get("token_interval")
    if token_int:
      extractions["token_interval"] = tokenizer.TokenInterval(**token_int)
    else:
      extractions["token_interval"] = None

    char_int = extractions.get("char_interval")
    if char_int:
      extractions["char_interval"] = data.CharInterval(**char_int)
    else:
      extractions["char_interval"] = None

    status_str = extractions.get("alignment_status")
    if status_str:
      extractions["alignment_status"] = data.AlignmentStatus(status_str)
    else:
      extractions["alignment_status"] = None

  return data.AnnotatedDocument(
      document_id=adoc_dic.get("document_id"),
      text=adoc_dic.get("text"),
      extractions=[
          data.Extraction(**ent) for ent in adoc_dic.get("extractions", [])
      ],
  )
