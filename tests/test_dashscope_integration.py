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

"""Integration tests for the DashScope provider."""

import os
import unittest
from unittest import mock

from langextract.core import exceptions
from langextract.extraction import extract_schema
from langextract.providers import dashscope


class DashScopeIntegrationTest(unittest.TestCase):
    """Integration tests for the DashScope provider."""

    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if API key is not available
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        self.skip_reason = "DASHSCOPE_API_KEY not set in environment"

    @unittest.skipIf(
        os.environ.get("SKIP_LIVE_TESTS", "0") == "1", "Skipping live API tests"
    )
    def test_dashscope_model_initialization(self):
        """Test initialization of DashScope model."""
        if not self.api_key:
            self.skipTest(self.skip_reason)
            return

        # Test basic initialization
        model = dashscope.DashScopeLanguageModel(
            model_id="qwen3-72b-instruct", api_key=self.api_key
        )
        self.assertEqual(model.model_id, "qwen3-72b-instruct")
        self.assertEqual(
            model.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @unittest.skipIf(
        os.environ.get("SKIP_LIVE_TESTS", "0") == "1", "Skipping live API tests"
    )
    def test_dashscope_infer(self):
        """Test inference with DashScope model."""
        if not self.api_key:
            self.skipTest(self.skip_reason)
            return

        model = dashscope.DashScopeLanguageModel(
            model_id="qwen3-72b-instruct", api_key=self.api_key
        )

        prompts = ["Say hello in one sentence"]
        results = list(model.infer(prompts))

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][0].output)
        self.assertEqual(results[0][0].score, 1.0)

    @unittest.skipIf(
        os.environ.get("SKIP_LIVE_TESTS", "0") == "1", "Skipping live API tests"
    )
    def test_extract_schema_with_dashscope(self):
        """Test schema extraction using DashScope model."""
        if not self.api_key:
            self.skipTest(self.skip_reason)
            return

        # Define a simple schema
        class Person:
            """A person."""

            name: str
            age: int

        text = "John is 30 years old."

        # Use DashScope model for extraction
        with mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": self.api_key}):
            # Force using DashScope model
            results = extract_schema(
                text, schema_type=Person, model="qwen3-72b-instruct"
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "John")
        self.assertEqual(results[0].age, 30)

    def test_dashscope_without_api_key(self):
        """Test that initialization fails without API key."""
        with mock.patch.dict("os.environ", clear=True):
            with self.assertRaises(exceptions.InferenceConfigError):
                dashscope.DashScopeLanguageModel(
                    model_id="qwen3-72b-instruct", api_key=None
                )


if __name__ == "__main__":
    unittest.main()
