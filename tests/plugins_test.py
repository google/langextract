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

from importlib import metadata
from unittest import mock

from absl.testing import absltest

from langextract import plugins

_GROUP = "langextract.providers"
_BUILTIN_GEMINI_SPEC = "langextract.providers.gemini:GeminiLanguageModel"
_BUILTIN_OLLAMA_SPEC = "langextract.providers.ollama:OllamaLanguageModel"
_BUILTIN_OPENAI_SPEC = "langextract.providers.openai:OpenAILanguageModel"
_THIRD_PARTY_GEMINI_SPEC = "third_party.gemini:CustomGemini"


class AvailableProvidersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    plugins._discovered.cache_clear()
    self.addCleanup(plugins._discovered.cache_clear)
    # LangExtract's own entry points come first (worst-case scan order), so
    # an unskipped built-in would shadow the colliding third-party entry and
    # leak the optional built-in.
    self.enter_context(
        mock.patch.object(
            plugins,
            "_safe_entry_points",
            autospec=True,
            return_value=[
                metadata.EntryPoint("gemini", _BUILTIN_GEMINI_SPEC, _GROUP),
                metadata.EntryPoint("ollama", _BUILTIN_OLLAMA_SPEC, _GROUP),
                metadata.EntryPoint("openai", _BUILTIN_OPENAI_SPEC, _GROUP),
                metadata.EntryPoint("gemini", _THIRD_PARTY_GEMINI_SPEC, _GROUP),
                metadata.EntryPoint("my_gemini", _BUILTIN_GEMINI_SPEC, _GROUP),
            ],
        )
    )

  def test_builtins_take_precedence_by_default(self):
    providers = plugins.available_providers()

    self.assertEqual(providers["gemini"], _BUILTIN_GEMINI_SPEC)
    self.assertEqual(providers["openai"], _BUILTIN_OPENAI_SPEC)

  def test_allow_override_lets_third_party_replace_builtin(self):
    providers = plugins.available_providers(allow_override=True)

    self.assertEqual(providers["gemini"], _THIRD_PARTY_GEMINI_SPEC)
    self.assertEqual(providers["ollama"], _BUILTIN_OLLAMA_SPEC)

  def test_excluding_optional_omits_optional_builtin(self):
    providers = plugins.available_providers(include_optional=False)

    self.assertNotIn("openai", providers)

  def test_third_party_alias_for_builtin_class_is_kept(self):
    providers = plugins.available_providers()

    self.assertEqual(providers["my_gemini"], _BUILTIN_GEMINI_SPEC)


if __name__ == "__main__":
  absltest.main()
