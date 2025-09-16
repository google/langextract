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

"""Tests for langextract.progress module."""

import unittest
from unittest import mock

import tqdm

from langextract import progress


class ProgressTest(unittest.TestCase):
    def test_download_progress_bar(self):
        """Test download progress bar creation."""
        pbar = progress.create_download_progress_bar(
            1024, "https://example.com/file.txt"
        )

        self.assertIsInstance(pbar, tqdm.tqdm)
        self.assertEqual(pbar.total, 1024)
        self.assertIn("Downloading", pbar.desc)

    def test_extraction_progress_bar(self):
        """Test extraction progress bar creation."""
        pbar = progress.create_extraction_progress_bar(range(10), "gemini-2.0-flash")

        self.assertIsInstance(pbar, tqdm.tqdm)
        self.assertIn("LangExtract", pbar.desc)
        self.assertIn("gemini-2.0-flash", pbar.desc)

    def test_save_load_progress_bars(self):
        """Test save and load progress bar creation."""
        save_pbar = progress.create_save_progress_bar("/path/file.json")
        load_pbar = progress.create_load_progress_bar("/path/file.json")

        self.assertIsInstance(save_pbar, tqdm.tqdm)
        self.assertIsInstance(load_pbar, tqdm.tqdm)
        self.assertIn("Saving", save_pbar.desc)
        self.assertIn("Loading", load_pbar.desc)

    def test_model_info_extraction(self):
        """Test extracting model info from objects."""
        mock_model = mock.MagicMock()
        mock_model.model_id = "gemini-1.5-pro"
        self.assertEqual(progress.get_model_info(mock_model), "gemini-1.5-pro")

        mock_model = mock.MagicMock()
        del mock_model.model_id
        del mock_model.model_url
        self.assertIsNone(progress.get_model_info(mock_model))

    def test_formatting_functions(self):
        """Test message formatting functions."""
        stats = progress.format_extraction_stats(1500, 5000)
        self.assertIn("1,500", stats)
        self.assertIn("5,000", stats)

        desc = progress.format_extraction_progress("gemini-2.0-flash")
        self.assertIn("LangExtract", desc)
        self.assertIn("gemini-2.0-flash", desc)

        desc_no_model = progress.format_extraction_progress(None)
        self.assertIn("Processing", desc_no_model)


if __name__ == "__main__":
    unittest.main()
