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

"""Tests for optional Docling support."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import types as py_types
from unittest import mock
import builtins

from absl.testing import absltest

import langextract as lx
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import types
from langextract import docling_support


class _TestModel(base_model.BaseLanguageModel):  # pylint: disable=too-few-public-methods
  """Deterministic model for unit tests."""

  def __init__(self, output: str):
    super().__init__()
    self._output = output
    self.format_type = data.FormatType.JSON

  def infer(self, batch_prompts, **kwargs):  # pytype: disable=signature-mismatch
    del kwargs
    for _ in batch_prompts:
      yield [types.ScoredOutput(score=1.0, output=self._output)]


class DoclingSupportTest(absltest.TestCase):

  def test_attach_provenance_collects_overlapping_spans(self):
    adoc = data.AnnotatedDocument(
        text="Hello world",
        extractions=[
            data.Extraction(
                extraction_class="entity",
                extraction_text="Hello world",
                char_interval=data.CharInterval(start_pos=0, end_pos=11),
            )
        ],
    )
    converted = docling_support.ConvertedDocument(
        text="Hello world",
        spans=[
            docling_support.ProvenanceSpan(
                start=0,
                end=5,
                provenance=[{"page_no": 1, "bbox": [0, 0, 1, 1]}],
            ),
            docling_support.ProvenanceSpan(
                start=6,
                end=11,
                provenance=[{"page_no": 1, "bbox": [2, 0, 3, 1]}],
            ),
        ],
        source_path="/tmp/doc.pdf",
    )

    out = docling_support.attach_provenance(adoc, converted)
    self.assertIs(out, adoc)
    self.assertLen(out.extractions or [], 1)
    self.assertLen(out.extractions[0].provenance or [], 2)

  def test_extract_with_file_input_attaches_provenance(self):
    # Fake docling modules.
    docling = py_types.ModuleType("docling")
    docling_document_converter = py_types.ModuleType("docling.document_converter")

    class DocumentConverter:  # pylint: disable=too-few-public-methods
      def convert(self, source: str):
        del source
        return mock.Mock(document=object())

    docling_document_converter.DocumentConverter = DocumentConverter

    docling_core = py_types.ModuleType("docling_core")
    docling_core_transforms = py_types.ModuleType("docling_core.transforms")
    docling_core_transforms_serializer = py_types.ModuleType(
        "docling_core.transforms.serializer"
    )
    docling_core_transforms_serializer_markdown = py_types.ModuleType(
        "docling_core.transforms.serializer.markdown"
    )

    class _Span:  # pylint: disable=too-few-public-methods
      def __init__(self, start: int, end: int, item):
        self.start = start
        self.end = end
        self.item = item

    class _Item:  # pylint: disable=too-few-public-methods
      def __init__(self, prov):
        self.prov = prov

    class MarkdownDocSerializer:  # pylint: disable=too-few-public-methods
      def __init__(self, doc):
        del doc

      def serialize(self):
        item1 = _Item([{"page_no": 1, "bbox": [0, 0, 1, 1]}])
        item2 = _Item([{"page_no": 1, "bbox": [2, 0, 3, 1]}])
        return mock.Mock(
            text="Hello world",
            spans=[
                _Span(0, 5, item1),
                _Span(6, 11, item2),
            ],
        )

    docling_core_transforms_serializer_markdown.MarkdownDocSerializer = (
        MarkdownDocSerializer
    )

    fake_modules = {
        "docling": docling,
        "docling.document_converter": docling_document_converter,
        "docling_core": docling_core,
        "docling_core.transforms": docling_core_transforms,
        "docling_core.transforms.serializer": docling_core_transforms_serializer,
        "docling_core.transforms.serializer.markdown": (
            docling_core_transforms_serializer_markdown
        ),
    }

    with tempfile.TemporaryDirectory() as td:
      pdf_path = pathlib.Path(td) / "input.pdf"
      pdf_path.write_bytes(b"%PDF-1.4\n%...")

      model = _TestModel(
          output='{"extractions":[{"entity":"Hello world","entity_attributes":{}}]}'
      )

      examples = [
          data.ExampleData(
              text="Hello world",
              extractions=[
                  data.Extraction(
                      extraction_class="entity", extraction_text="Hello world"
                  )
              ],
          )
      ]

      with mock.patch.dict(sys.modules, fake_modules, clear=False):
        result = lx.extract(
            text_or_documents=str(pdf_path),
            prompt_description="Extract entity.",
            examples=examples,
            model=model,
            fence_output=False,
            use_schema_constraints=False,
            show_progress=False,
        )

    self.assertIsInstance(result, data.AnnotatedDocument)
    self.assertLen(result.extractions or [], 1)
    self.assertLen(result.extractions[0].provenance or [], 2)

  def test_extract_with_file_input_requires_docling(self):
    with tempfile.TemporaryDirectory() as td:
      pdf_path = pathlib.Path(td) / "input.pdf"
      pdf_path.write_bytes(b"%PDF-1.4\n%...")

      model = _TestModel(
          output='{"extractions":[{"entity":"Hello","entity_attributes":{}}]}'
      )
      examples = [
          data.ExampleData(
              text="Hello",
              extractions=[
                  data.Extraction(extraction_class="entity", extraction_text="Hello")
              ],
          )
      ]

      orig_import = builtins.__import__

      def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("docling") or name.startswith("docling_core"):
          raise ImportError("docling blocked for test")
        return orig_import(name, globals, locals, fromlist, level)

      with mock.patch("builtins.__import__", side_effect=_blocked_import):
        with self.assertRaises(exceptions.InferenceConfigError):
          _ = lx.extract(
              text_or_documents=str(pdf_path),
              prompt_description="Extract entity.",
              examples=examples,
              model=model,
              fence_output=False,
              use_schema_constraints=False,
              show_progress=False,
          )


if __name__ == "__main__":
  absltest.main()

