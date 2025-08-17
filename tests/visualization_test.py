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

"""Tests for langextract.visualization."""

from unittest import mock

from absl.testing import absltest

from langextract import visualization
from langextract.core import data as lx_data

_PALETTE = visualization._PALETTE
_VISUALIZATION_CSS = visualization._VISUALIZATION_CSS


class VisualizationTest(absltest.TestCase):

  def test_assign_colors_basic_assignment(self):

    extractions = [
        lx_data.Extraction(
            extraction_class="CLASS_A",
            extraction_text="text_a",
            char_interval=lx_data.CharInterval(start_pos=0, end_pos=1),
        ),
        lx_data.Extraction(
            extraction_class="CLASS_B",
            extraction_text="text_b",
            char_interval=lx_data.CharInterval(start_pos=1, end_pos=2),
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
    extraction = lx_data.Extraction(
        extraction_class="GREETING",
        extraction_text="Hello",
        char_interval=lx_data.CharInterval(start_pos=0, end_pos=5),
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
    extraction = lx_data.Extraction(
        extraction_class="UNSAFE_CLASS",
        extraction_text="<unsafe> content & ampersand.",
        char_interval=lx_data.CharInterval(start_pos=10, end_pos=39),
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

    doc = lx_data.AnnotatedDocument(
        text="Patient needs Aspirin.",
        extractions=[
            lx_data.Extraction(
                extraction_class="MEDICATION",
                extraction_text="Aspirin",
                char_interval=lx_data.CharInterval(
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

    # Verify expected components appear in output
    for component in expected_components:
      self.assertIn(component, actual_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_no_extractions_renders_text_and_empty_legend(self):

    doc = lx_data.AnnotatedDocument(text="No entities here.", extractions=[])
    body_html = (
        '<div class="lx-animated-wrapper"><p>No valid extractions to'
        " animate.</p></div>"
    )
    css_html = _VISUALIZATION_CSS
    expected_html = css_html + body_html

    actual_html = visualization.visualize(doc)

    self.assertEqual(actual_html, expected_html)


if __name__ == "__main__":
  absltest.main()
