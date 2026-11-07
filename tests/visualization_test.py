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
import subprocess
import sys
import textwrap
from unittest import mock

from absl.testing import absltest

from langextract import visualization
from langextract.core import data

_PALETTE = visualization._PALETTE
_HTML_CHARSET_META = visualization._HTML_CHARSET_META
_VISUALIZATION_CSS = visualization._VISUALIZATION_CSS
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

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


def _word_document(text: str) -> data.AnnotatedDocument:
  """Returns a document with one extraction per whitespace-separated word."""
  return data.AnnotatedDocument(
      text=text,
      extractions=[
          data.Extraction(
              extraction_class="WORD",
              extraction_text=word,
              char_interval=data.CharInterval(
                  start_pos=text.index(word),
                  end_pos=text.index(word) + len(word),
              ),
          )
          for word in text.split()
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

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_two_viewers_on_one_page_keep_instance_scoped_state(self):
    # Regression test for https://github.com/google/langextract/issues/578:
    # two visualize() results on one page must not share element ids or
    # handlers, otherwise the second viewer drives the first viewer's panel.
    combined_html = visualization.visualize(
        _word_document("Alice Bob")
    ) + visualization.visualize(_word_document("Carol Dave"))

    ids = re.findall(r'\bid="([^"]+)"', combined_html)
    self.assertNotEmpty(ids)
    duplicated_ids = sorted({i for i in ids if ids.count(i) > 1})
    self.assertEqual(duplicated_ids, [])

    viewers = re.findall(
        r'<div class="lx-animated-wrapper.*?</script>',
        combined_html,
        re.DOTALL,
    )
    self.assertLen(viewers, 2)
    viewer_ids = []
    for viewer in viewers:
      wrapper_id_match = re.search(
          r'<div class="lx-animated-wrapper[^>]*\bid="([^"]+)"', viewer
      )
      self.assertIsNotNone(wrapper_id_match)
      wrapper_id = wrapper_id_match.group(1)
      self.assertRegex(wrapper_id, r"^lx-viewer-[0-9a-f]+$")
      suffix = wrapper_id.rsplit("-", 1)[-1]

      script_match = re.search(r"<script>(.*?)</script>", viewer, re.DOTALL)
      self.assertIsNotNone(script_match)
      referenced_ids = re.findall(
          r"lx-[a-z-]+-[0-9a-f]+", script_match.group(1)
      )
      self.assertNotEmpty(referenced_ids)
      # Every id the script touches must belong to this viewer's own container,
      # so a script bound to another viewer's wrapper fails here.
      for referenced_id in referenced_ids:
        self.assertEqual(
            referenced_id.rsplit("-", 1)[-1],
            suffix,
            f"{referenced_id} is referenced by viewer {wrapper_id}",
        )

      viewer_ids.append(set(re.findall(r'\bid="([^"]+)"', viewer)))

    for index, viewer in enumerate(viewers):
      self.assertNotEmpty(viewer_ids[index])
      for other_viewer_id in viewer_ids[1 - index]:
        self.assertNotIn(other_viewer_id, viewer)

    # Each viewer binds its own handlers instead of overwriting window globals.
    for global_name in (
        "window.playPause",
        "window.nextExtraction",
        "window.prevExtraction",
        "window.jumpToExtraction",
    ):
      self.assertNotIn(global_name, combined_html)

  def test_viewer_ids_are_unique_across_independent_processes(self):
    # Follow-up to https://github.com/google/langextract/issues/578: the id
    # suffix must not come from a process-local counter, because every process
    # starts its own counter at one. Viewers rendered by separate processes
    # would collide again, so the second viewer's panel and buttons would
    # drive the first viewer's document.
    script = textwrap.dedent("""
        import re
        from langextract import data, visualization

        document = data.AnnotatedDocument(
            text="Alice",
            extractions=[
                data.Extraction(
                    extraction_class="WORD",
                    extraction_text="Alice",
                    char_interval=data.CharInterval(start_pos=0, end_pos=5),
                )
            ],
        )
        html = visualization.visualize(document)
        print(re.search(r'id="(lx-viewer-[^"]+)"', html).group(1))
        """)

    def viewer_id_from_fresh_process() -> str:
      completed = subprocess.run(
          [sys.executable, "-c", script],
          capture_output=True,
          text=True,
          check=True,
          cwd=_REPO_ROOT,
      )
      return completed.stdout.strip()

    self.assertNotEqual(
        viewer_id_from_fresh_process(),
        viewer_id_from_fresh_process(),
        "viewers rendered by separate processes must not share an element id",
    )


if __name__ == "__main__":
  absltest.main()
