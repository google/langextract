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

"""Tests for langextract.io."""

from unittest import mock

from absl.testing import absltest

from langextract import io


class HtmlNormalizationTest(absltest.TestCase):

  def test_normalize_html_text_strips_boilerplate(self):
    html = """
<!doctype html>
<html>
  <body>
    <header><a href=\"#\">Site Header</a></header>
    <div class=\"menu\">Products | About | Careers</div>
    <aside>Sidebar legal disclaimer link list</aside>
    <main>
      <article>
        <h1>Terms and Conditions</h1>
        <p>Section 1. You agree to arbitration.</p>
        <section>
          <h2>Liability</h2>
          <p>We are not liable except as required by law.</p>
          <ul>
            <li>Clause A applies.</li>
            <li>Clause B applies.</li>
          </ul>
        </section>
      </article>
    </main>
    <footer>Footer links and social icons</footer>
  </body>
</html>
"""

    normalized = io._normalize_html_text(html)

    self.assertIn("Terms and Conditions", normalized)
    self.assertIn("Section 1. You agree to arbitration.", normalized)
    self.assertIn("Liability", normalized)
    self.assertIn("Clause A applies.", normalized)
    self.assertNotIn("Products | About | Careers", normalized)
    self.assertNotIn("Sidebar legal disclaimer", normalized)
    self.assertNotIn("Footer links", normalized)

  @mock.patch("requests.get", autospec=True)
  def test_download_text_from_url_normalizes_html(self, mock_get):
    html = """
<html>
  <body>
    <nav>Main menu item</nav>
    <div id=\"sidebar\">Sidebar links</div>
    <main>
      <h1>Master Service Agreement</h1>
      <p>This agreement governs use of the platform.</p>
      <div>
        <p>Either party may terminate with 30 days notice.</p>
      </div>
    </main>
  </body>
</html>
""".strip()

    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.headers = {
        "Content-Type": "text/html; charset=utf-8",
        "Content-Length": str(len(html.encode("utf-8"))),
    }
    response.iter_content.return_value = [html.encode("utf-8")]
    mock_get.return_value = response

    text = io.download_text_from_url(
        "https://example.com/legal", show_progress=False
    )

    self.assertIn("Master Service Agreement", text)
    self.assertIn("This agreement governs use of the platform.", text)
    self.assertIn("Either party may terminate with 30 days notice.", text)
    self.assertNotIn("Main menu item", text)
    self.assertNotIn("Sidebar links", text)

  @mock.patch("requests.get", autospec=True)
  def test_download_text_from_url_non_html_unchanged(self, mock_get):
    plain_text = "Plain legal content with no html tags."

    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Length": str(len(plain_text.encode("utf-8"))),
    }
    response.iter_content.return_value = [plain_text.encode("utf-8")]
    mock_get.return_value = response

    text = io.download_text_from_url(
        "https://example.com/legal.txt", show_progress=False
    )

    self.assertEqual(text, plain_text)


if __name__ == "__main__":
  absltest.main()
