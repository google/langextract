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

"""Provider registry for LangExtract.

This module provides a centralized registry for LLM providers, allowing
registration, lookup, and enumeration of available providers.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, Type

from langextract.core import base_model
from langextract.core import exceptions
from langextract.core import types as core_types
from langextract.providers import router
from langextract.providers import builtin_registry


@dataclasses.dataclass
class ProviderInfo:
    """Information about a registered provider.

    Attributes:
        name: The provider name identifier.
        cls: The provider class.
        patterns: The model ID patterns this provider handles.
        priority: The resolution priority.
    """

    name: str
    cls: Type[base_model.BaseLanguageModel]
    patterns: Sequence[str]
    priority: int


class ProviderRegistry:
    """Registry for LLM providers.

    This class provides a simplified interface for registering and looking up
    LLM providers. It wraps the existing router module for backward compatibility.

    Example usage:
        # Get the global registry
        registry = ProviderRegistry.get_global()

        # Register a custom provider
        registry.register(MyCustomProvider)

        # Look up a provider by name
        provider = registry.get("gemini")

        # Get MockProvider for testing
        mock_provider = registry.get("mock")
        result = mock_provider(fixed_response="test").generate("hello")

        # List all registered providers
        providers = registry.list_all()
    """

    _global_instance: ProviderRegistry | None = None

    def __init__(self) -> None:
        """Initialize a new ProviderRegistry.

        The registry automatically registers MockProvider with name 'mock'
        for testing purposes.
        """
        self._builtins_loaded = False
        self._mock_registered = False
        self._register_mock_provider()

    def _register_mock_provider(self) -> None:
        """Register MockProvider with the registry.

        This is called during initialization so that MockProvider is always
        available via registry.get('mock').
        """
        if not self._mock_registered:
            router.register(r"^mock$", r"^mock-", priority=100)(MockProvider)
            self._mock_registered = True

    @classmethod
    def get_global(cls) -> ProviderRegistry:
        """Get the global ProviderRegistry instance.

        Returns:
            The singleton global registry instance.
        """
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    def _reset_global(cls) -> None:
        """Reset the global instance for testing.

        This method should only be used in tests.
        """
        cls._global_instance = None

    def _ensure_builtins_loaded(self) -> None:
        """Ensure built-in providers are loaded."""
        if not self._builtins_loaded:
            for config in builtin_registry.BUILTIN_PROVIDERS:
                router.register_lazy(
                    *config["patterns"],
                    target=config["target"],
                    priority=config["priority"],
                )
            self._builtins_loaded = True

    def register(
        self,
        provider_cls: Type[base_model.BaseLanguageModel],
        patterns: Sequence[str] | None = None,
        priority: int = 0,
    ) -> Type[base_model.BaseLanguageModel]:
        """Register a provider class.

        Args:
            provider_cls: The provider class to register.
            patterns: Optional model ID patterns this provider handles.
                If not provided, attempts to get patterns from the class's
                `supported_models` property or uses the class name.
            priority: Resolution priority (higher wins on conflicts).

        Returns:
            The registered provider class (for decorator usage).
        """
        if patterns is None:
            try:
                patterns = []
            except Exception:
                class_name = provider_cls.__name__
                patterns = [f"^{class_name.lower()}"]

        return router.register(*patterns, priority=priority)(provider_cls)

    def get(self, name: str) -> Type[base_model.BaseLanguageModel]:
        """Look up a provider by name.

        Args:
            name: The provider name (e.g., "gemini", "openai", "mock") or class name.

        Returns:
            The provider class.

        Raises:
            InferenceConfigError: If no provider matches the name.
        """
        self._ensure_builtins_loaded()
        return router.resolve_provider(name)

    def get_for_model(self, model_id: str) -> Type[base_model.BaseLanguageModel]:
        """Look up a provider by model ID.

        Args:
            model_id: The model identifier (e.g., "gemini-2.5-flash", "mock-model").

        Returns:
            The provider class that handles this model.

        Raises:
            InferenceConfigError: If no provider is registered for the model ID.
        """
        self._ensure_builtins_loaded()
        return router.resolve(model_id)

    def list_all(self) -> list[ProviderInfo]:
        """List all registered providers.

        Returns:
            A list of ProviderInfo objects for all registered providers.
        """
        self._ensure_builtins_loaded()

        providers: list[ProviderInfo] = []
        entries = router.list_entries()

        for patterns, priority in entries:
            try:
                if patterns:
                    pass
            except Exception:
                pass

        return providers

    def clear(self) -> None:
        """Clear all registered providers.

        This method is mainly for testing. Note that MockProvider will be
        re-registered on the next operation.
        """
        router.clear()
        self._builtins_loaded = False
        self._mock_registered = False


class MockProvider(base_model.BaseLanguageModel):
    """A mock provider for testing purposes.

    This provider returns predefined responses without making any API calls.
    It's useful for unit tests and integration tests.

    Example usage:
        # Create a mock provider with fixed responses
        mock = MockProvider(fixed_response='{"result": "test"}')

        # Or use it with a response function
        def my_response(prompt, **kwargs):
            return f"Response to: {prompt}"

        mock = MockProvider(response_fn=my_response)

        # Use in tests
        result = mock.generate("Test prompt")
        print(result.text)

        # Check if close() was called
        print(mock.close_called)  # Counter increments on each close()

        # Use as context manager
        with MockProvider(fixed_response="test") as mock:
            result = mock.generate("hello")
        print(mock.close_called)  # 1
    """

    def __init__(
        self,
        fixed_response: str | None = None,
        response_fn: Callable[..., str] | None = None,
        usage: base_model.Usage | None = None,
        model_id: str = "mock-model",
        **kwargs: Any,
    ) -> None:
        """Initialize the MockProvider.

        Args:
            fixed_response: A fixed response string to return for all prompts.
                Either fixed_response or response_fn must be provided.
            response_fn: A function that takes (prompt, **kwargs) and returns
                a response string. Used if fixed_response is None.
            usage: Optional usage information to return in GenerateResult.
            model_id: The model ID to report.
            **kwargs: Additional keyword arguments (ignored).
        """
        super().__init__()
        self.model_id = model_id
        self.fixed_response = fixed_response
        self.response_fn = response_fn
        self.usage = usage
        self._extra_kwargs = kwargs
        self.close_called: int = 0

        if fixed_response is None and response_fn is None:
            self.fixed_response = '{"mock": "response"}'

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "mock"

    @property
    def supported_models(self) -> Sequence[str]:
        """Return the supported models."""
        return ["^mock$", "^mock-"]

    def _get_response(self, prompt: str, **kwargs: Any) -> str:
        """Get the response for a prompt."""
        if self.response_fn is not None:
            return self.response_fn(prompt, **kwargs)
        return self.fixed_response or ""

    def generate(
        self, prompt: str, model: str | None = None, **kwargs: Any
    ) -> base_model.GenerateResult:
        """Generate a mock response.

        Args:
            prompt: The input prompt.
            model: Optional model ID (ignored).
            **kwargs: Additional keyword arguments.

        Returns:
            A GenerateResult with the mock response.
        """
        response_text = self._get_response(prompt, **kwargs)
        return base_model.GenerateResult(
            text=response_text,
            usage=self.usage,
            raw_response={"prompt": prompt, "model": model or self.model_id},
        )

    async def agenerate(
        self, prompt: str, model: str | None = None, **kwargs: Any
    ) -> base_model.GenerateResult:
        """Generate a mock response asynchronously.

        Args:
            prompt: The input prompt.
            model: Optional model ID (ignored).
            **kwargs: Additional keyword arguments.

        Returns:
            A GenerateResult with the mock response.
        """
        return self.generate(prompt, model, **kwargs)

    def close(self) -> None:
        """Clean up resources and increment close_called counter.

        This is useful for testing that context managers properly release resources.
        """
        self.close_called += 1

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ):
        """Infer method for backward compatibility.

        Args:
            batch_prompts: A list of prompts.
            **kwargs: Additional keyword arguments.

        Yields:
            Lists of ScoredOutput objects.
        """
        for prompt in batch_prompts:
            response_text = self._get_response(prompt, **kwargs)
            yield [core_types.ScoredOutput(score=1.0, output=response_text)]
