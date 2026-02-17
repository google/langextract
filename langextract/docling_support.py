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

"""Optional Docling integration.

This module is intentionally dependency-free at import time. Docling is imported
only when file conversion is requested.

Primary responsibilities:
- Convert a supported document (e.g., PDF/DOCX/PPTX) to a text representation.
- Preserve provenance by mapping serialized text spans back to source items.
- Attach provenance metadata to grounded extractions via char intervals.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

from langextract.core import data
from langextract.core import exceptions


@dataclasses.dataclass(frozen=True, slots=True)
class ProvenanceSpan:
  """Span of serialized text mapped to provenance items."""

  start: int
  end: int
  provenance: list[dict[str, Any]]


@dataclasses.dataclass(frozen=True, slots=True)
class ConvertedDocument:
  """Converted document text plus provenance mapping."""

  text: str
  spans: list[ProvenanceSpan]
  source_path: str


def convert_file(path: pathlib.Path) -> ConvertedDocument:
  """Convert a file to text and provenance spans via Docling.

  Args:
    path: Local file path.

  Returns:
    ConvertedDocument containing serialized text and provenance spans.

  Raises:
    InferenceConfigError: If Docling is not installed.
    InferenceRuntimeError: If conversion fails.
  """
  try:
    # pylint: disable=import-outside-toplevel
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
  except ImportError as e:
    raise exceptions.InferenceConfigError(
        "Docling is required for file inputs. Install Docling and its"
        " dependencies (e.g., `pip install docling docling-core`)."
    ) from e

  try:
    converter = DocumentConverter()
    doc = converter.convert(source=str(path)).document
    serializer = MarkdownDocSerializer(doc=doc)
    ser_result = serializer.serialize()
    text = getattr(ser_result, "text", None)
    if not isinstance(text, str):
      raise TypeError("Docling serializer did not return text")
  except Exception as e:
    raise exceptions.InferenceRuntimeError(
        f"Failed to convert document via Docling: {e}", original=e
    ) from e

  spans: list[ProvenanceSpan] = []
  raw_spans = getattr(ser_result, "spans", None)
  if raw_spans:
    for sp in raw_spans:
      start = getattr(sp, "start", None)
      end = getattr(sp, "end", None)
      if not isinstance(start, int) or not isinstance(end, int):
        continue
      item = getattr(sp, "item", None) or getattr(sp, "span_source", None)
      prov = _extract_provenance(item)
      spans.append(ProvenanceSpan(start=start, end=end, provenance=prov))

  return ConvertedDocument(text=text, spans=spans, source_path=str(path))


def attach_provenance(
    doc: data.AnnotatedDocument, converted: ConvertedDocument
) -> data.AnnotatedDocument:
  """Attach provenance metadata to grounded extractions in-place."""
  if not doc.extractions:
    return doc
  if not converted.spans:
    return doc

  for ext in doc.extractions:
    if ext.char_interval is None:
      continue
    start = ext.char_interval.start_pos
    end = ext.char_interval.end_pos
    if start is None or end is None:
      continue
    prov_items = _collect_overlapping_provenance(converted.spans, start, end)
    if prov_items:
      ext.provenance = prov_items

  return doc


def _collect_overlapping_provenance(
    spans: list[ProvenanceSpan], start: int, end: int
) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  seen: set[str] = set()

  for sp in spans:
    if start < sp.end and end > sp.start:
      for item in sp.provenance:
        key = repr(item)
        if key in seen:
          continue
        seen.add(key)
        out.append(item)

  return out


def _extract_provenance(item: Any) -> list[dict[str, Any]]:
  """Best-effort provenance extraction from a Docling span source."""
  if item is None:
    return []

  prov = getattr(item, "prov", None)
  if not prov:
    return []

  out: list[dict[str, Any]] = []
  for p in prov:
    out.append(_as_jsonable(p))
  return out


def _as_jsonable(value: Any) -> dict[str, Any]:
  """Convert an arbitrary provenance object into a JSON-serializable dict."""
  # Pydantic v2
  if hasattr(value, "model_dump"):
    try:
      dumped = value.model_dump()
      if isinstance(dumped, dict):
        return dumped
    except Exception:
      pass
  # Dataclasses
  if dataclasses.is_dataclass(value):
    try:
      dumped = dataclasses.asdict(value)
      if isinstance(dumped, dict):
        return dumped
    except Exception:
      pass
  # Mapping
  if isinstance(value, dict):
    return dict(value)

  # Attribute-based fallback
  out: dict[str, Any] = {}
  for k in ("page_no", "page", "bbox", "x0", "y0", "x1", "y1"):
    if hasattr(value, k):
      out[k] = getattr(value, k)
  if not out:
    out["value"] = str(value)
  return out

