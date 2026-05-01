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

"""Base interfaces for language models."""
from __future__ import annotations

import abc
import asyncio
from collections.abc import Iterator, Sequence
import concurrent.futures
import dataclasses
import json
from typing import Any, Mapping, Optional, Sequence as TypingSequence

import yaml

from langextract.core import schema
from langextract.core import types

__all__ = ['BaseLanguageModel', 'LLMProvider', 'GenerateResult', 'Usage']


@dataclasses.dataclass
class Usage:
    """Token usage information.

    Attributes:
        input_tokens: Number of tokens in the input prompt.
        output_tokens: Number of tokens in the generated output.
        total_tokens: Total number of tokens used.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclasses.dataclass
class GenerateResult:
    """Result of a generate() call.

    Attributes:
        text: The generated text output.
        usage: Optional token usage information.
        raw_response: The raw response from the provider API.
    """

    text: str
    usage: Usage | None = None
    raw_response: Any = None


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers.

    This interface defines the contract that all LLM providers must implement.
    It provides both synchronous and asynchronous generation methods, as well
    as context manager support for automatic resource cleanup.

    Example usage:
        # Synchronous usage
        provider = MyProvider(api_key="...")
        result = provider.generate("Hello", model="gpt-4")
        print(result.text)
        provider.close()

        # As context manager (auto-close)
        with MyProvider(api_key="...") as provider:
            result = provider.generate("Hello")

        # Async usage
        async with MyProvider(api_key="...") as provider:
            result = await provider.agenerate("Hello")
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the provider name identifier.

        Returns:
            A short string identifying the provider (e.g., "gemini", "openai", "mock").
        """
        ...

    @property
    @abc.abstractmethod
    def supported_models(self) -> TypingSequence[str]:
        """Return the model ID patterns supported by this provider.

        Returns:
            A list of regex patterns matching model IDs that this provider handles.
        """
        ...

    @abc.abstractmethod
    def generate(
        self, prompt: str, model: str | None = None, **kwargs: Any
    ) -> GenerateResult:
        """Generate text synchronously.

        Args:
            prompt: The input prompt to send to the model.
            model: Optional model ID to use. If None, uses the provider's default.
            **kwargs: Additional provider-specific arguments.

        Returns:
            A GenerateResult containing the generated text and metadata.
        """
        ...

    @abc.abstractmethod
    async def agenerate(
        self, prompt: str, model: str | None = None, **kwargs: Any
    ) -> GenerateResult:
        """Generate text asynchronously.

        Args:
            prompt: The input prompt to send to the model.
            model: Optional model ID to use. If None, uses the provider's default.
            **kwargs: Additional provider-specific arguments.

        Returns:
            A GenerateResult containing the generated text and metadata.
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up any resources held by this provider.

        This method should be called when the provider is no longer needed.
        It closes HTTP clients, connection pools, etc.
        """
        ...

    def __enter__(self) -> LLMProvider:
        """Enter context manager.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit context manager, calling close().

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Traceback if an exception occurred.
        """
        self.close()

    async def __aenter__(self) -> LLMProvider:
        """Enter async context manager.

        Returns:
            Self for use in async with statement.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit async context manager, calling close().

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Traceback if an exception occurred.
        """
        self.close()


class BaseLanguageModel(LLMProvider):
    """An abstract inference class for managing LLM inference.

    This class implements the LLMProvider interface with default implementations
    for backward compatibility. Providers should inherit from this class and
    implement infer() at minimum.

    Attributes:
        _constraint: A `Constraint` object specifying constraints for model output.
    """

    def __init__(self, constraint: types.Constraint | None = None, **kwargs: Any):
        """Initializes the BaseLanguageModel with an optional constraint.

        Args:
            constraint: Applies constraints when decoding the output. Defaults to no
                constraint.
            **kwargs: Additional keyword arguments passed to the model.
        """
        self._constraint = constraint or types.Constraint()
        self._schema: schema.BaseSchema | None = None
        self._fence_output_override: bool | None = None
        self._extra_kwargs: dict[str, Any] = kwargs.copy()

    @property
    def name(self) -> str:
        """Return the provider name identifier.

        Default implementation derives the name from the class name by:
        1. Stripping "LanguageModel" suffix if present
        2. Converting to lowercase

        Returns:
            The provider name.
        """
        class_name = self.__class__.__name__
        if class_name.endswith("LanguageModel"):
            return class_name[:-13].lower()
        return class_name.lower()

    @property
    def supported_models(self) -> TypingSequence[str]:
        """Return the model ID patterns supported by this provider.

        Default implementation returns an empty list. Providers should override
        this to return the actual patterns they support.

        Returns:
            List of regex patterns.
        """
        return []

    @classmethod
    def get_schema_class(cls) -> type[Any] | None:
        """Return the schema class this provider supports."""
        return None

    def apply_schema(self, schema_instance: schema.BaseSchema | None) -> None:
        """Apply a schema instance to this provider.

        Optional method that providers can override to store the schema instance
        for runtime use. The default implementation stores it as _schema.

        Args:
            schema_instance: The schema instance to apply, or None to clear.
        """
        self._schema = schema_instance

    @property
    def schema(self) -> schema.BaseSchema | None:
        """The current schema instance if one is configured.

        Returns:
            The schema instance or None if no schema is applied.
        """
        return self._schema

    def set_fence_output(self, fence_output: bool | None) -> None:
        """Set explicit fence output preference.

        Args:
            fence_output: True to force fences, False to disable, None for auto.
        """
        if not hasattr(self, '_fence_output_override'):
            self._fence_output_override = None
        self._fence_output_override = fence_output

    @property
    def requires_fence_output(self) -> bool:
        """Whether this model requires fence output for parsing.

        Uses explicit override if set, otherwise computes from schema.
        Returns True if no schema or schema doesn't require raw output.
        """
        if (
            hasattr(self, '_fence_output_override')
            and self._fence_output_override is not None
        ):
            return self._fence_output_override

        schema_obj = self.schema
        if schema_obj is None:
            return True
        return not schema_obj.requires_raw_output

    def merge_kwargs(
        self, runtime_kwargs: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Merge stored extra kwargs with runtime kwargs.

        Runtime kwargs take precedence over stored kwargs.

        Args:
            runtime_kwargs: Kwargs provided at inference time, or None.

        Returns:
            Merged kwargs dictionary.
        """
        base = getattr(self, '_extra_kwargs', {}) or {}
        incoming = dict(runtime_kwargs or {})
        return {**base, **incoming}

    @abc.abstractmethod
    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[types.ScoredOutput]]:
        """Implements language model inference.

        Args:
            batch_prompts: Batch of inputs for inference. Single element list can be
                used for a single input.
            **kwargs: Additional arguments for inference, like temperature and
                max_decode_steps.

        Returns: Batch of Sequence of probable output text outputs, sorted by
            descending score.
        """

    def infer_batch(
        self, prompts: Sequence[str], batch_size: int = 32
    ) -> list[list[types.ScoredOutput]]:
        """Batch inference with configurable batch size.

        This is a convenience method that collects all results from infer().

        Args:
            prompts: List of prompts to process.
            batch_size: Batch size (currently unused, for future optimization).

        Returns:
            List of lists of ScoredOutput objects.
        """
        results = []
        for output in self.infer(prompts):
            results.append(list(output))
        return results

    def generate(
        self, prompt: str, model: str | None = None, **kwargs: Any
    ) -> GenerateResult:
        """Generate text synchronously (default implementation based on infer()).

        This is a fallback implementation for providers that haven't implemented
        the new generate() API yet. It wraps infer() to provide the new interface.

        Args:
            prompt: The input prompt.
            model: Optional model ID (passed to infer() if applicable).
            **kwargs: Additional arguments.

        Returns:
            A GenerateResult with the output.
        """
        results = list(self.infer([prompt], **kwargs))
        if results and results[0]:
            output_text = results[0][0].output
        else:
            output_text = ""

        return GenerateResult(
            text=output_text,
            usage=None,
            raw_response={"infer_results": results},
        )

    async def agenerate(
        self, prompt: str, model: str | None = None, **kwargs: Any
    ) -> GenerateResult:
        """Generate text asynchronously (default implementation).

        Default implementation runs the synchronous generate() in a thread pool.
        Providers should override this for native async support.

        Args:
            prompt: The input prompt.
            model: Optional model ID.
            **kwargs: Additional arguments.

        Returns:
            A GenerateResult with the output.
        """
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self.generate(prompt, model, **kwargs),
            )

    def close(self) -> None:
        """Clean up resources (default no-op implementation).

        Providers should override this if they hold resources like HTTP clients
        that need to be explicitly closed.
        """
        pass

    def parse_output(self, output: str) -> Any:
        """Parses model output as JSON or YAML.

        Note: This expects raw JSON/YAML without code fences.
        Code fence extraction is handled by resolver.py.

        Args:
            output: Raw output string from the model.

        Returns:
            Parsed Python object (dict or list).

        Raises:
            ValueError: If output cannot be parsed as JSON or YAML.
        """
        format_type = getattr(self, 'format_type', types.FormatType.JSON)

        try:
            if format_type == types.FormatType.JSON:
                return json.loads(output)
            else:
                return yaml.safe_load(output)
        except Exception as e:
            raise ValueError(
                f'Failed to parse output as {format_type.name}: {str(e)}'
            ) from e
