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

"""Tests for the new provider abstraction layer.

This module tests:
- ProviderRegistry registration and lookup
- MockProvider returning preset values
- Backward compatibility with old API
- Multiple providers coexisting
- Entry points with priority suffix
- Context manager resource cleanup
- Async context manager
"""

from __future__ import annotations

import asyncio
from importlib import metadata
from types import SimpleNamespace
from typing import Any, Sequence
from unittest import mock

import pytest

from langextract.core import base_model
from langextract.core import types as core_types
from langextract.providers import (
    GenerateResult,
    LLMProvider,
    MockProvider,
    ProviderRegistry,
    Usage,
)
from langextract.providers import _parse_entry_point_value, _reset_for_testing, router


class TestGenerateResult:
    """Tests for GenerateResult dataclass."""

    def test_GenerateResult_creation(self):
        """Test creating a GenerateResult instance."""
        result = GenerateResult(
            text="Generated text",
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            raw_response={"key": "value"},
        )

        assert result.text == "Generated text"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.raw_response == {"key": "value"}

    def test_GenerateResult_defaults(self):
        """Test GenerateResult with default values."""
        result = GenerateResult(text="Simple text")

        assert result.text == "Simple text"
        assert result.usage is None
        assert result.raw_response is None


class TestUsage:
    """Tests for Usage dataclass."""

    def test_Usage_creation(self):
        """Test creating a Usage instance."""
        usage = Usage(input_tokens=5, output_tokens=10, total_tokens=15)

        assert usage.input_tokens == 5
        assert usage.output_tokens == 10
        assert usage.total_tokens == 15

    def test_Usage_defaults(self):
        """Test Usage with default values."""
        usage = Usage()

        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.total_tokens is None


class TestLLMProviderInterface:
    """Tests for the LLMProvider abstract interface."""

    def test_BaseLanguageModel_implements_LLMProvider(self):
        """Test that BaseLanguageModel implements LLMProvider interface."""

        class ConcreteModel(base_model.BaseLanguageModel):
            def infer(
                self, batch_prompts: Sequence[str], **kwargs: Any
            ):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output=f"Response to: {prompt}")]

        model = ConcreteModel()

        # Verify it's an instance of LLMProvider
        assert isinstance(model, LLMProvider)

        # Verify default implementations work
        assert isinstance(model.name, str)
        assert isinstance(model.supported_models, Sequence)

        # Test generate method
        result = model.generate("Test prompt")
        assert isinstance(result, GenerateResult)
        assert "Test prompt" in result.text

        # Test close method (no-op by default)
        model.close()

    def test_LLMProvider_name_property(self):
        """Test that name property returns correct value based on class name."""

        class MyCustomLanguageModel(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output=prompt)]

        model = MyCustomLanguageModel()
        # Should strip "LanguageModel" suffix and lowercase
        assert model.name == "mycustom"


class TestMockProvider:
    """Tests for MockProvider."""

    def test_MockProvider_fixed_response(self):
        """Test MockProvider with fixed response."""
        expected_response = '{"result": "test", "value": 42}'
        mock = MockProvider(fixed_response=expected_response)

        result = mock.generate("Any prompt")

        assert result.text == expected_response
        assert result.raw_response is not None
        assert result.raw_response["prompt"] == "Any prompt"

    def test_MockProvider_response_fn(self):
        """Test MockProvider with response function."""

        def custom_response(prompt: str, **kwargs) -> str:
            return f"Custom response to: {prompt}"

        mock = MockProvider(response_fn=custom_response)

        result1 = mock.generate("First prompt")
        result2 = mock.generate("Second prompt")

        assert result1.text == "Custom response to: First prompt"
        assert result2.text == "Custom response to: Second prompt"

    def test_MockProvider_default_response(self):
        """Test MockProvider default response."""
        mock = MockProvider()

        result = mock.generate("Test")

        assert result.text is not None
        assert len(result.text) > 0

    def test_MockProvider_usage(self):
        """Test MockProvider with usage information."""
        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        mock = MockProvider(fixed_response="Test", usage=usage)

        result = mock.generate("Prompt")

        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150

    def test_MockProvider_infer_backward_compatibility(self):
        """Test that MockProvider supports infer() for backward compatibility."""
        mock = MockProvider(fixed_response="Fixed")

        results = list(mock.infer(["Prompt 1", "Prompt 2"]))

        assert len(results) == 2
        assert results[0][0].output == "Fixed"
        assert results[1][0].output == "Fixed"

    def test_MockProvider_async_generate(self):
        """Test MockProvider async generate method."""
        import asyncio

        mock = MockProvider(fixed_response="Async test")

        result = asyncio.run(mock.agenerate("Async prompt"))

        assert result.text == "Async test"

    def test_MockProvider_name_and_supported_models(self):
        """Test MockProvider name and supported_models properties."""
        mock = MockProvider()

        assert mock.name == "mock"
        assert len(mock.supported_models) > 0
        for pattern in mock.supported_models:
            assert pattern.startswith("^mock")

    def test_MockProvider_close(self):
        """Test that MockProvider.close() is a no-op."""
        mock = MockProvider()
        mock.close()  # Should not raise


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def setup_method(self):
        """Reset router and global registry before each test."""
        router.clear()
        ProviderRegistry._reset_global()

    def teardown_method(self):
        """Clean up after each test."""
        router.clear()
        ProviderRegistry._reset_global()

    def test_ProviderRegistry_get_global(self):
        """Test that get_global() returns a singleton instance."""
        registry1 = ProviderRegistry.get_global()
        registry2 = ProviderRegistry.get_global()

        assert registry1 is registry2

    def test_ProviderRegistry_register_custom_provider(self):
        """Test registering a custom provider."""

        @router.register(r"^custom")
        class CustomProvider(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output=f"Custom: {prompt}")]

        registry = ProviderRegistry()
        # The provider should be registered via decorator

        # Verify we can resolve it
        provider_cls = router.resolve("custom-model")
        assert provider_cls is CustomProvider

    def test_ProviderRegistry_get_for_model(self):
        """Test getting provider by model ID."""

        @router.register(r"^test-gemini")
        class TestGeminiProvider(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output=prompt)]

        registry = ProviderRegistry()
        registry.register(TestGeminiProvider, patterns=[r"^test-gemini"])

        provider_cls = registry.get_for_model("test-gemini-pro")
        assert provider_cls is TestGeminiProvider

    def test_ProviderRegistry_clear(self):
        """Test clearing the registry."""

        @router.register(r"^temp")
        class TempProvider(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output=prompt)]

        registry = ProviderRegistry()

        # Should be able to resolve before clear
        provider_cls = router.resolve("temp-model")
        assert provider_cls is TempProvider

        # Clear registry
        registry.clear()

        # Should fail after clear
        from langextract import exceptions

        with pytest.raises(exceptions.InferenceConfigError):
            router.resolve("temp-model")

    def test_multiple_providers_coexist(self):
        """Test that multiple providers can coexist."""

        @router.register(r"^provider-a", priority=10)
        class ProviderA(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output="A")]

        @router.register(r"^provider-b", priority=10)
        class ProviderB(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output="B")]

        registry = ProviderRegistry()

        # Resolve each provider
        cls_a = registry.get_for_model("provider-a-model")
        cls_b = registry.get_for_model("provider-b-model")

        assert cls_a is ProviderA
        assert cls_b is ProviderB

    def test_provider_priority(self):
        """Test that higher priority provider wins on conflict."""

        @router.register(r"^conflict", priority=0)
        class LowPriorityProvider(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output="low")]

        @router.register(r"^conflict", priority=10)
        class HighPriorityProvider(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output="high")]

        registry = ProviderRegistry()

        # High priority should win
        provider_cls = registry.get_for_model("conflict-model")
        assert provider_cls is HighPriorityProvider


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def setup_method(self):
        """Reset router before each test."""
        router.clear()

    def teardown_method(self):
        """Clean up after each test."""
        router.clear()

    def test_old_API_infer_still_works(self):
        """Test that the old infer() API still works."""

        class LegacyProvider(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output=f"Legacy: {prompt}")]

        provider = LegacyProvider()

        # Old API
        results = list(provider.infer(["Prompt 1", "Prompt 2"]))

        assert len(results) == 2
        assert results[0][0].output == "Legacy: Prompt 1"
        assert results[1][0].output == "Legacy: Prompt 2"

    def test_new_API_generate_works(self):
        """Test that the new generate() API works."""

        class NewProvider(base_model.BaseLanguageModel):
            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output=f"New: {prompt}")]

        provider = NewProvider()

        # New API
        result = provider.generate("Test prompt")

        assert isinstance(result, GenerateResult)
        assert result.text == "New: Test prompt"

    def test_LLMProvider_context_manager(self):
        """Test that LLMProvider can be used as context manager."""

        class ContextTestProvider(base_model.BaseLanguageModel):
            def __init__(self):
                super().__init__()
                self.closed = False

            def infer(self, batch_prompts, **kwargs):
                for prompt in batch_prompts:
                    yield [core_types.ScoredOutput(score=1.0, output=prompt)]

            def close(self):
                self.closed = True

        with ContextTestProvider() as provider:
            result = provider.generate("Test")
            assert result.text == "Test"
            assert not provider.closed

        # Should be closed after context exit
        assert provider.closed


class TestMockProviderFixture:
    """Tests for using MockProvider as a pytest fixture."""

    @pytest.fixture
    def mock_provider(self):
        """Fixture that creates a MockProvider."""
        return MockProvider(fixed_response='{"fixture": "response"}')

    def test_use_mock_provider_fixture(self, mock_provider):
        """Test using the MockProvider fixture."""
        result = mock_provider.generate("Fixture test")

        assert result.text == '{"fixture": "response"}'

    @pytest.fixture
    def dynamic_mock_provider(self):
        """Fixture that creates a MockProvider with dynamic response."""

        def response_fn(prompt, **kwargs):
            if "name" in prompt:
                return '{"name": "Test"}'
            elif "value" in prompt:
                return '{"value": 42}'
            return '{"default": true}'

        return MockProvider(response_fn=response_fn)

    def test_dynamic_mock_provider(self, dynamic_mock_provider):
        """Test dynamic responses based on prompt content."""
        result1 = dynamic_mock_provider.generate("Get the name")
        result2 = dynamic_mock_provider.generate("Get the value")
        result3 = dynamic_mock_provider.generate("Something else")

        assert result1.text == '{"name": "Test"}'
        assert result2.text == '{"value": 42}'
        assert result3.text == '{"default": true}'


class TestEntryPointsPriority:
    """Tests for entry_points priority suffix syntax."""

    def test_parse_entry_point_value_basic(self):
        """Test parsing basic entry point value without priority."""
        value = "my_pkg.provider:MyProvider"
        target, priority = _parse_entry_point_value(value)

        assert target == "my_pkg.provider:MyProvider"
        assert priority is None

    def test_parse_entry_point_value_with_priority(self):
        """Test parsing entry point value with priority suffix."""
        value = "my_pkg.provider:MyProvider:priority=100"
        target, priority = _parse_entry_point_value(value)

        assert target == "my_pkg.provider:MyProvider"
        assert priority == 100

    def test_parse_entry_point_value_with_zero_priority(self):
        """Test parsing entry point value with priority=0."""
        value = "my_pkg.provider:MyProvider:priority=0"
        target, priority = _parse_entry_point_value(value)

        assert target == "my_pkg.provider:MyProvider"
        assert priority == 0

    def test_parse_entry_point_value_with_high_priority(self):
        """Test parsing entry point value with high priority."""
        value = "my_pkg.provider:MyProvider:priority=999"
        target, priority = _parse_entry_point_value(value)

        assert target == "my_pkg.provider:MyProvider"
        assert priority == 999


class TestMockProviderRegistry:
    """Tests for MockProvider being accessible via registry.get('mock')."""

    def setup_method(self):
        """Reset router and global registry before each test."""
        router.clear()
        ProviderRegistry._reset_global()

    def teardown_method(self):
        """Clean up after each test."""
        router.clear()
        ProviderRegistry._reset_global()

    def test_registry_get_mock(self):
        """Test that registry.get('mock') returns MockProvider class."""
        registry = ProviderRegistry()

        mock_cls = registry.get("mock")

        assert mock_cls is MockProvider

    def test_registry_get_for_model_mock(self):
        """Test that registry.get_for_model('mock-model') returns MockProvider."""
        registry = ProviderRegistry()

        mock_cls = registry.get_for_model("mock-model")

        assert mock_cls is MockProvider

    def test_registry_get_for_model_mock_custom(self):
        """Test that registry.get_for_model('mock-custom') returns MockProvider."""
        registry = ProviderRegistry()

        mock_cls = registry.get_for_model("mock-custom-v1")

        assert mock_cls is MockProvider

    def test_mock_provider_via_registry_usage(self):
        """Test using MockProvider obtained via registry."""
        registry = ProviderRegistry()
        mock_cls = registry.get("mock")

        mock = mock_cls(fixed_response='{"from_registry": true}')
        result = mock.generate("Test prompt")

        assert result.text == '{"from_registry": true}'
        assert result.raw_response is not None
        assert result.raw_response["prompt"] == "Test prompt"


class TestContextManagerClose:
    """Tests for context manager calling close() and tracking close_called."""

    def test_mock_provider_close_called_counter(self):
        """Test that close() increments close_called counter."""
        mock = MockProvider()

        assert mock.close_called == 0

        mock.close()
        assert mock.close_called == 1

        mock.close()
        assert mock.close_called == 2

    def test_context_manager_calls_close(self):
        """Test that context manager calls close() on exit."""
        mock = MockProvider()

        assert mock.close_called == 0

        with mock:
            result = mock.generate("Test")
            assert result.text is not None
            assert mock.close_called == 0

        assert mock.close_called == 1

    def test_context_manager_with_exception(self):
        """Test that close() is called even when exception occurs."""
        mock = MockProvider()

        assert mock.close_called == 0

        try:
            with mock:
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert mock.close_called == 1

    def test_multiple_context_managers(self):
        """Test multiple context manager usages increment counter."""
        mock = MockProvider()

        with mock:
            pass
        assert mock.close_called == 1

        with mock:
            pass
        assert mock.close_called == 2


class TestAsyncContextManager:
    """Tests for async context manager support."""

    def test_async_context_manager_calls_close(self):
        """Test that async context manager calls close() on exit."""

        async def test_async():
            mock = MockProvider()

            assert mock.close_called == 0

            async with mock:
                result = await mock.agenerate("Test")
                assert result.text is not None
                assert mock.close_called == 0

            assert mock.close_called == 1
            return mock.close_called

        result = asyncio.run(test_async())
        assert result == 1

    def test_async_context_manager_with_exception(self):
        """Test that close() is called even when exception occurs in async context."""

        async def test_async():
            mock = MockProvider()

            assert mock.close_called == 0

            try:
                async with mock:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            assert mock.close_called == 1
            return mock.close_called

        result = asyncio.run(test_async())
        assert result == 1

    def test_provider_implements_async_context_manager(self):
        """Test that LLMProvider implements async context manager protocol."""
        mock = MockProvider()

        assert hasattr(mock, "__aenter__")
        assert hasattr(mock, "__aexit__")
        assert callable(mock.__aenter__)
        assert callable(mock.__aexit__)

    def test_provider_implements_sync_context_manager(self):
        """Test that LLMProvider implements sync context manager protocol."""
        mock = MockProvider()

        assert hasattr(mock, "__enter__")
        assert hasattr(mock, "__exit__")
        assert callable(mock.__enter__)
        assert callable(mock.__exit__)
