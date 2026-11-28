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

import json
import pathlib
import re
import tempfile
from unittest import mock

from absl.testing import absltest

from langextract import io
from langextract import visualization
from langextract.core import data

_PALETTE = visualization._PALETTE
_HTML_CHARSET_META = visualization._HTML_CHARSET_META
_VISUALIZATION_CSS = visualization._VISUALIZATION_CSS

# Extraction values can contain markup, as in text taken from a web page.
_MARKUP_TEXT = "</script><b id=probe>x</b><!-- a & b"
_MARKUP_CLASS = "</script><!--CLASS&"


def _markup_document() -> data.AnnotatedDocument:
  """Returns a document whose extraction text and class contain markup."""
  return data.AnnotatedDocument(
      text=_MARKUP_TEXT,
      extractions=[
          data.Extraction(
              extraction_class=_MARKUP_CLASS,
              extraction_text=_MARKUP_TEXT,
              char_interval=data.CharInterval(
                  start_pos=0, end_pos=len(_MARKUP_TEXT)
              ),
          )
      ],
  )


def _single_extraction_document(
    document_id: str, text: str
) -> data.AnnotatedDocument:
  """Returns a document whose whole text is one extraction."""
  return data.AnnotatedDocument(
      document_id=document_id,
      text=text,
      extractions=[
          data.Extraction(
              extraction_class="C",
              extraction_text=text,
              char_interval=data.CharInterval(start_pos=0, end_pos=len(text)),
          )
      ],
  )


class VisualizationTest(absltest.TestCase):

  def test_assign_colors_basic_assignment(self):

    extractions = [
        data.Extraction(
            extraction_class="CLASS_A",
            extraction_text="text_a",
            char_interval=data.CharInterval(start_pos=0, end_pos=1),
        ),
        data.Extraction(
            extraction_class="CLASS_B",
            extraction_text="text_b",
            char_interval=data.CharInterval(start_pos=1, end_pos=2),
        ),
    ]
    # Classes are sorted alphabetically before color assignment.
    expected_color_map = {
        "CLASS_A": _PALETTE[0],
        "CLASS_B": _PALETTE[1],
    }

    actual_color_map = visualization._assign_colors(extractions)

    self.assertDictEqual(actual_color_map, expected_color_map)

  def test_build_highlighted_text_single_span_correct_html(self):

    text = "Hello world"
    extraction = data.Extraction(
        extraction_class="GREETING",
        extraction_text="Hello",
        char_interval=data.CharInterval(start_pos=0, end_pos=5),
    )
    extractions = [extraction]
    color_map = {"GREETING": "#ff0000"}
    expected_html = (
        '<span class="lx-highlight lx-current-highlight" data-idx="0" '
        'style="background-color:#ff0000;">Hello</span> world'
    )

    actual_html = visualization._build_highlighted_text(
        text, extractions, color_map
    )

    self.assertEqual(actual_html, expected_html)

  def test_build_highlighted_text_escapes_html_in_text_and_tooltip(self):

    text = "Text with <unsafe> content & ampersand."
    extraction = data.Extraction(
        extraction_class="UNSAFE_CLASS",
        extraction_text="<unsafe> content & ampersand.",
        char_interval=data.CharInterval(start_pos=10, end_pos=39),
        attributes={"detail": "Attribute with <tag> & 'quote'"},
    )
    # Highlighting "<unsafe> content & ampersand"
    extractions = [extraction]
    color_map = {"UNSAFE_CLASS": "#00ff00"}
    expected_highlighted_segment = "&lt;unsafe&gt; content &amp; ampersand."
    expected_html = (
        'Text with <span class="lx-highlight lx-current-highlight"'
        ' data-idx="0" '
        f'style="background-color:#00ff00;">{expected_highlighted_segment}</span>'
    )

    actual_html = visualization._build_highlighted_text(
        text, extractions, color_map
    )

    self.assertEqual(actual_html, expected_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_basic_document_renders_correctly(self):

    doc = data.AnnotatedDocument(
        text="Patient needs Aspirin.",
        extractions=[
            data.Extraction(
                extraction_class="MEDICATION",
                extraction_text="Aspirin",
                char_interval=data.CharInterval(
                    start_pos=14, end_pos=21
                ),  # "Aspirin"
            )
        ],
    )
    # Predictable color based on sorted class name "MEDICATION"
    med_color = _PALETTE[0]
    body_html = (
        'Patient needs <span class="lx-highlight lx-current-highlight"'
        f' data-idx="0" style="background-color:{med_color};">Aspirin</span>.'
    )
    legend_html = (
        '<div class="lx-legend">Highlights Legend: <span class="lx-label" '
        f'style="background-color:{med_color};">MEDICATION</span></div>'
    )
    css_html = _VISUALIZATION_CSS
    expected_components = [
        css_html,
        "lx-animated-wrapper",
        body_html,
        legend_html,
    ]

    actual_html = visualization.visualize(doc)

    self.assertStartsWith(actual_html, '<meta charset="UTF-8">')
    # Verify expected components appear in output
    for component in expected_components:
      self.assertIn(component, actual_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_no_extractions_renders_text_and_empty_legend(self):

    doc = data.AnnotatedDocument(text="No entities here.", extractions=[])
    body_html = (
        '<div class="lx-animated-wrapper"><p>No valid extractions to'
        " animate.</p></div>"
    )
    css_html = _VISUALIZATION_CSS
    expected_html = f"{_HTML_CHARSET_META}{css_html}{body_html}"

    actual_html = visualization.visualize(doc)

    self.assertEqual(actual_html, expected_html)

  def test_to_script_safe_json_escapes_markup_and_line_separators(self):

    value = {"text": "</script><!-- a & b -->\u2028\u2029"}
    expected_json = (
        '{"text": "\\u003c/script\\u003e\\u003c!-- a \\u0026 b --\\u003e'
        '\\u2028\\u2029"}'
    )

    actual_json = visualization._to_script_safe_json(value)

    self.assertEqual(actual_json, expected_json)
    self.assertEqual(json.loads(actual_json), value)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_markup_in_extraction_does_not_end_script_early(self):

    actual_html = visualization.visualize(_markup_document())

    script_start = actual_html.index("<script>")
    script_end = actual_html.index("</script>", script_start)
    script_body = actual_html[script_start:script_end]
    self.assertIn("function updateDisplay", script_body)
    self.assertNotIn("<!--", script_body)
    self.assertEqual(actual_html.count("</script>"), 1)
    self.assertNotIn(_MARKUP_TEXT, actual_html)
    self.assertNotIn(_MARKUP_CLASS, actual_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_script_safe_json_round_trips_extraction_values(self):

    actual_html = visualization.visualize(_markup_document())

    match = re.search(r"const extractions = (.*);\n", actual_html)
    self.assertIsNotNone(match)
    embedded_data = json.loads(match.group(1))
    self.assertEqual(embedded_data[0]["text"], _MARKUP_TEXT)
    self.assertEqual(embedded_data[0]["class"], _MARKUP_CLASS)

  def _write_two_documents(self) -> pathlib.Path:
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    tmp_dir = pathlib.Path(tmp.name)
    docs = [
        _single_extraction_document(doc_id, f"{doc_id} text")
        for doc_id in ("first", "second")
    ]
    io.save_annotated_documents(
        docs, output_dir=tmp_dir, output_name="docs.jsonl", show_progress=False
    )
    return tmp_dir / "docs.jsonl"

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_jsonl_defaults_to_first_document(self):
    path = self._write_two_documents()

    actual_html = visualization.visualize(path)

    self.assertIn("first text", actual_html)
    self.assertNotIn("second text", actual_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_jsonl_selects_document_by_id(self):
    path = self._write_two_documents()

    actual_html = visualization.visualize(path, document_id="second")

    self.assertIn("second text", actual_html)
    self.assertNotIn("first text", actual_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_jsonl_selects_last_document_and_accepts_str_path(self):
    path = self._write_two_documents()

    actual_html = visualization.visualize(str(path), document_id="second")

    self.assertIn("second text", actual_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_jsonl_duplicate_document_id_uses_first_match(self):
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    docs = [
        _single_extraction_document("dup", "one text"),
        _single_extraction_document("dup", "two text"),
    ]
    io.save_annotated_documents(
        docs, output_dir=tmp.name, output_name="d.jsonl", show_progress=False
    )

    actual_html = visualization.visualize(
        pathlib.Path(tmp.name) / "d.jsonl", document_id="dup"
    )

    self.assertIn("one text", actual_html)
    self.assertNotIn("two text", actual_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_in_memory_document_ignores_document_id(self):
    doc = _single_extraction_document("mine", "memory text")

    actual_html = visualization.visualize(doc, document_id="other")

    self.assertIn("memory text", actual_html)

  def test_visualize_empty_document_id_is_not_treated_as_default(self):
    path = self._write_two_documents()

    with self.assertRaisesRegex(ValueError, "''"):
      visualization.visualize(path, document_id="")

  def test_visualize_missing_file_still_raises_with_document_id(self):
    with self.assertRaises(FileNotFoundError):
      visualization.visualize("/nonexistent/x.jsonl", document_id="a")

  def test_visualize_jsonl_unknown_document_id_lists_available_ids(self):
    path = self._write_two_documents()

    with self.assertRaisesRegex(ValueError, "'missing'.*first.*second"):
      visualization.visualize(path, document_id="missing")


if __name__ == "__main__":
  absltest.main()
