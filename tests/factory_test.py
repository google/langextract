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

"""Tests for the factory module.

Note: This file tests the deprecated registry module which is now an alias
for router. The no-name-in-module warning for providers.registry is expected.
"""
# pylint: disable=no-name-in-module

import os
from unittest import mock

from absl.testing import absltest

from langextract import exceptions
from langextract import factory
from langextract import inference
from langextract.providers import registry


class FakeGeminiProvider(inference.BaseLanguageModel):
  """Fake Gemini provider for testing."""

  def __init__(self, model_id, api_key=None, **kwargs):
    self.model_id = model_id
    self.api_key = api_key
    self.kwargs = kwargs
    super().__init__()

  def infer(self, batch_prompts, **kwargs):
    return [[inference.ScoredOutput(score=1.0, output="gemini")]]

  def infer_batch(self, prompts, batch_size=32):
    return self.infer(prompts)


class FakeOpenAIProvider(inference.BaseLanguageModel):
  """Fake OpenAI provider for testing."""

  def __init__(self, model_id, api_key=None, **kwargs):
    if not api_key:
      raise ValueError("API key required")
    self.model_id = model_id
    self.api_key = api_key
    self.kwargs = kwargs
    super().__init__()

  def infer(self, batch_prompts, **kwargs):
    return [[inference.ScoredOutput(score=1.0, output="openai")]]

  def infer_batch(self, prompts, batch_size=32):
    return self.infer(prompts)


class FactoryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    registry.clear()
    import langextract.providers as providers_module  # pylint: disable=import-outside-toplevel

    providers_module._PLUGINS_LOADED = True
    registry.register_lazy(
        r"^gemini", target="factory_test:FakeGeminiProvider", priority=100
    )
    registry.register_lazy(
        r"^gpt", r"^o1", target="factory_test:FakeOpenAIProvider", priority=100
    )

  def tearDown(self):
    super().tearDown()
    registry.clear()
    import langextract.providers as providers_module  # pylint: disable=import-outside-toplevel

    providers_module._PLUGINS_LOADED = False

  def test_create_model_basic(self):
    """Test basic model creation."""
    config = factory.ModelConfig(
        model_id="gemini-pro", provider_kwargs={"api_key": "test-key"}
    )

    model = factory.create_model(config)
    self.assertIsInstance(model, FakeGeminiProvider)
    self.assertEqual(model.model_id, "gemini-pro")
    self.assertEqual(model.api_key, "test-key")

  def test_create_model_from_id(self):
    """Test convenience function for creating model from ID."""
    model = factory.create_model_from_id("gemini-flash", api_key="test-key")

    self.assertIsInstance(model, FakeGeminiProvider)
    self.assertEqual(model.model_id, "gemini-flash")
    self.assertEqual(model.api_key, "test-key")

  @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "env-gemini-key"})
  def test_uses_gemini_api_key_from_environment(self):
    """Factory should use GEMINI_API_KEY from environment for Gemini models."""
    config = factory.ModelConfig(model_id="gemini-pro")

    model = factory.create_model(config)
    self.assertEqual(model.api_key, "env-gemini-key")

  @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"})
  def test_uses_openai_api_key_from_environment(self):
    """Factory should use OPENAI_API_KEY from environment for OpenAI models."""
    config = factory.ModelConfig(model_id="gpt-4")

    model = factory.create_model(config)
    self.assertEqual(model.api_key, "env-openai-key")

  @mock.patch.dict(
      os.environ, {"LANGEXTRACT_API_KEY": "env-langextract-key"}, clear=True
  )
  def test_falls_back_to_langextract_api_key_when_provider_key_missing(self):
    """Factory uses LANGEXTRACT_API_KEY when provider-specific key is missing."""
    config = factory.ModelConfig(model_id="gemini-pro")

    model = factory.create_model(config)
    self.assertEqual(model.api_key, "env-langextract-key")

  @mock.patch.dict(
      os.environ,
      {
          "GEMINI_API_KEY": "gemini-key",
          "LANGEXTRACT_API_KEY": "langextract-key",
      },
  )
  def test_provider_specific_key_takes_priority_over_langextract_key(self):
    """Factory prefers provider-specific API key over LANGEXTRACT_API_KEY."""
    config = factory.ModelConfig(model_id="gemini-pro")

    model = factory.create_model(config)
    self.assertEqual(model.api_key, "gemini-key")

  def test_explicit_kwargs_override_env(self):
    """Test that explicit kwargs override environment variables."""
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
      config = factory.ModelConfig(
          model_id="gemini-pro", provider_kwargs={"api_key": "explicit-key"}
      )

      model = factory.create_model(config)
      self.assertEqual(model.api_key, "explicit-key")

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_wraps_provider_initialization_error_in_inference_config_error(self):
    """Factory should wrap provider errors in InferenceConfigError."""
    config = factory.ModelConfig(model_id="gpt-4")

    with self.assertRaises(exceptions.InferenceConfigError) as cm:
      factory.create_model(config)

    self.assertIn("Failed to create provider", str(cm.exception))
    self.assertIn("API key required", str(cm.exception))

  def test_raises_error_when_no_provider_matches_model_id(self):
    """Factory should raise InferenceConfigError for unregistered model IDs."""
    config = factory.ModelConfig(model_id="unknown-model")

    with self.assertRaises(exceptions.InferenceConfigError) as cm:
      factory.create_model(config)

    self.assertIn("No provider registered", str(cm.exception))

  def test_additional_kwargs_passed_through(self):
    """Test that additional kwargs are passed to provider."""
    config = factory.ModelConfig(
        model_id="gemini-pro",
        provider_kwargs={
            "api_key": "test-key",
            "temperature": 0.5,
            "max_tokens": 100,
            "custom_param": "value",
        },
    )

    model = factory.create_model(config)
    self.assertEqual(model.kwargs["temperature"], 0.5)
    self.assertEqual(model.kwargs["max_tokens"], 100)
    self.assertEqual(model.kwargs["custom_param"], "value")

  @mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:11434"})
  def test_ollama_uses_base_url_from_environment(self):
    """Factory should use OLLAMA_BASE_URL from environment for Ollama models."""

    @registry.register(r"^ollama")
    class FakeOllamaProvider(inference.BaseLanguageModel):  # pylint: disable=unused-variable

      def __init__(self, model_id, base_url=None, **kwargs):
        self.model_id = model_id
        self.base_url = base_url
        super().__init__()

      def infer(self, batch_prompts, **kwargs):
        return [[inference.ScoredOutput(score=1.0, output="ollama")]]

      def infer_batch(self, prompts, batch_size=32):
        return self.infer(prompts)

    config = factory.ModelConfig(model_id="ollama/llama2")
    model = factory.create_model(config)

    self.assertEqual(model.base_url, "http://custom:11434")

  def test_ollama_models_select_without_api_keys(self):
    """Test that Ollama models resolve without API keys or explicit type."""

    @registry.register(
        r"^llama", r"^gemma", r"^mistral", r"^qwen", priority=100
    )
    class FakeOllamaProvider(inference.BaseLanguageModel):

      def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        super().__init__()

      def infer(self, batch_prompts, **kwargs):
        return [[inference.ScoredOutput(score=1.0, output="test")]]

      def infer_batch(self, prompts, batch_size=32):
        return self.infer(prompts)

    test_models = ["llama3", "gemma2:2b", "mistral:7b", "qwen3:0.6b"]

    for model_id in test_models:
      with self.subTest(model_id=model_id):
        with mock.patch.dict(os.environ, {}, clear=True):
          config = factory.ModelConfig(model_id=model_id)
          model = factory.create_model(config)
          self.assertIsInstance(model, FakeOllamaProvider)
          self.assertEqual(model.model_id, model_id)

  def test_model_config_fields_are_immutable(self):
    """ModelConfig fields should not be modifiable after creation."""
    config = factory.ModelConfig(
        model_id="gemini-pro", provider_kwargs={"api_key": "test"}
    )

    with self.assertRaises(AttributeError):
      config.model_id = "different"

  def test_model_config_allows_dict_contents_modification(self):
    """ModelConfig allows modification of dict contents (not deeply frozen)."""
    config = factory.ModelConfig(
        model_id="gemini-pro", provider_kwargs={"api_key": "test"}
    )

    config.provider_kwargs["new_key"] = "value"

    self.assertEqual(config.provider_kwargs["new_key"], "value")

  def test_uses_highest_priority_provider_when_multiple_match(self):
    """Factory uses highest priority provider when multiple patterns match."""

    @registry.register(r"^gemini", priority=90)
    class AnotherGeminiProvider(inference.BaseLanguageModel):  # pylint: disable=unused-variable

      def __init__(self, model_id=None, **kwargs):
        self.model_id = model_id or "default-model"
        self.kwargs = kwargs
        super().__init__()

      def infer(self, batch_prompts, **kwargs):
        return [[inference.ScoredOutput(score=1.0, output="another")]]

      def infer_batch(self, prompts, batch_size=32):
        return self.infer(prompts)

    config = factory.ModelConfig(model_id="gemini-pro")
    model = factory.create_model(config)

    self.assertIsInstance(model, FakeGeminiProvider)  # Priority 100 wins

  def test_explicit_provider_overrides_pattern_matching(self):
    """Factory should use explicit provider even when pattern doesn't match."""

    @registry.register(r"^another", priority=90)
    class AnotherProvider(inference.BaseLanguageModel):

      def __init__(self, model_id=None, **kwargs):
        self.model_id = model_id or "default-model"
        self.kwargs = kwargs
        super().__init__()

      def infer(self, batch_prompts, **kwargs):
        return [[inference.ScoredOutput(score=1.0, output="another")]]

      def infer_batch(self, prompts, batch_size=32):
        return self.infer(prompts)

    config = factory.ModelConfig(
        model_id="gemini-pro", provider="AnotherProvider"
    )
    model = factory.create_model(config)

    self.assertIsInstance(model, AnotherProvider)
    self.assertEqual(model.model_id, "gemini-pro")

  def test_provider_without_model_id_uses_provider_default(self):
    """Factory should use provider's default model_id when none specified."""

    @registry.register(r"^default-provider$", priority=50)
    class DefaultProvider(inference.BaseLanguageModel):

      def __init__(self, model_id="default-model", **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        super().__init__()

      def infer(self, batch_prompts, **kwargs):
        return [[inference.ScoredOutput(score=1.0, output="default")]]

      def infer_batch(self, prompts, batch_size=32):
        return self.infer(prompts)

    config = factory.ModelConfig(provider="DefaultProvider")
    model = factory.create_model(config)

    self.assertIsInstance(model, DefaultProvider)
    self.assertEqual(model.model_id, "default-model")

  def test_raises_error_when_neither_model_id_nor_provider_specified(self):
    """Factory raises ValueError when config has neither model_id nor provider."""
    config = factory.ModelConfig()

    with self.assertRaises(ValueError) as cm:
      factory.create_model(config)

    self.assertIn(
        "Either model_id or provider must be specified", str(cm.exception)
    )


if __name__ == "__main__":
  absltest.main()
