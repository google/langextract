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

"""Tests for provider discovery precedence in langextract.plugins."""

from importlib import metadata
from unittest import mock

from absl.testing import absltest

from langextract import plugins

_BUILTIN_GEMINI_SPEC = "langextract.providers.gemini:GeminiLanguageModel"
_BUILTIN_OPENAI_SPEC = "langextract.providers.openai:OpenAILanguageModel"
_THIRD_PARTY_GEMINI_SPEC = "third_party.gemini:CustomGemini"
_THIRD_PARTY_OPENAI_SPEC = "third_party.openai:CustomOpenAI"
_THIRD_PARTY_ONLY_SPEC = "third_party.custom:CustomModel"


class AvailableProvidersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            plugins,
            "_discovered",
            autospec=True,
            return_value={
                "gemini": _THIRD_PARTY_GEMINI_SPEC,
                "openai": _THIRD_PARTY_OPENAI_SPEC,
                "custom": _THIRD_PARTY_ONLY_SPEC,
            },
        )
    )

  def test_builtins_take_precedence_by_default(self):
    providers = plugins.available_providers()

    self.assertEqual(providers["gemini"], _BUILTIN_GEMINI_SPEC)
    self.assertEqual(providers["openai"], _BUILTIN_OPENAI_SPEC)
    self.assertEqual(providers["custom"], _THIRD_PARTY_ONLY_SPEC)

  def test_allow_override_lets_third_party_replace_builtins(self):
    providers = plugins.available_providers(allow_override=True)

    self.assertEqual(providers["gemini"], _THIRD_PARTY_GEMINI_SPEC)
    self.assertEqual(providers["openai"], _THIRD_PARTY_OPENAI_SPEC)
    self.assertEqual(providers["custom"], _THIRD_PARTY_ONLY_SPEC)

  def test_allow_override_keeps_builtins_without_third_party_entry(self):
    providers = plugins.available_providers(allow_override=True)

    self.assertEqual(
        providers["ollama"], "langextract.providers.ollama:OllamaLanguageModel"
    )

  def test_excluding_optional_leaves_third_party_spec_in_place(self):
    providers = plugins.available_providers(include_optional=False)

    self.assertEqual(providers["openai"], _THIRD_PARTY_OPENAI_SPEC)
    self.assertEqual(providers["gemini"], _BUILTIN_GEMINI_SPEC)


class DiscoveredTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    plugins._discovered.cache_clear()
    self.addCleanup(plugins._discovered.cache_clear)
    # LangExtract's own entry point is listed first, as it is when its
    # distribution is scanned before the third-party plugin's.
    self.enter_context(
        mock.patch.object(
            plugins,
            "_safe_entry_points",
            autospec=True,
            return_value=[
                metadata.EntryPoint(
                    name="gemini",
                    value=_BUILTIN_GEMINI_SPEC,
                    group="langextract.providers",
                ),
                metadata.EntryPoint(
                    name="gemini",
                    value=_THIRD_PARTY_GEMINI_SPEC,
                    group="langextract.providers",
                ),
                metadata.EntryPoint(
                    name="openai",
                    value=_BUILTIN_OPENAI_SPEC,
                    group="langextract.providers",
                ),
            ],
        )
    )

  def test_own_entry_points_are_not_reported_as_third_party(self):
    self.assertEqual(
        plugins._discovered(), {"gemini": _THIRD_PARTY_GEMINI_SPEC}
    )

  def test_override_reaches_third_party_listed_after_builtin(self):
    providers = plugins.available_providers(allow_override=True)

    self.assertEqual(providers["gemini"], _THIRD_PARTY_GEMINI_SPEC)

  def test_excluding_optional_omits_own_optional_entry_point(self):
    providers = plugins.available_providers(include_optional=False)

    self.assertNotIn("openai", providers)


if __name__ == "__main__":
  absltest.main()
