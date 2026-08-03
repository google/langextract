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

"""Tests for the built-in provider routing patterns in patterns.py."""

from absl.testing import absltest

from langextract import providers as providers_module
from langextract.providers import openai as openai_provider
from langextract.providers import router


class OpenAIPatternsTest(absltest.TestCase):
  """Regression test for issue #492: o-series and gpt-3.5 model IDs
  previously matched no provider pattern and raised InferenceConfigError."""

  def setUp(self):
    super().setUp()
    router.clear()
    providers_module._builtins_loaded = False  # pylint: disable=protected-access
    providers_module.load_builtins_once()

  def tearDown(self):
    super().tearDown()
    router.clear()
    providers_module._builtins_loaded = False  # pylint: disable=protected-access

  def test_reasoning_and_legacy_openai_models_resolve_to_openai_provider(self):
    model_ids = [
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
        "gpt-3.5-turbo",
    ]
    for model_id in model_ids:
      with self.subTest(model_id=model_id):
        resolved = router.resolve(model_id)
        self.assertEqual(resolved, openai_provider.OpenAILanguageModel)

  def test_ollama_gpt_oss_pattern_is_unaffected(self):
    # Guards against the new `^o[1-9]` OpenAI pattern accidentally
    # colliding with Ollama's `^gpt-oss` pattern (it shouldn't - different
    # literal prefixes - but this pins the expected behavior explicitly).
    resolved = router.resolve("gpt-oss:20b")
    self.assertNotEqual(resolved, openai_provider.OpenAILanguageModel)


if __name__ == "__main__":
  absltest.main()
