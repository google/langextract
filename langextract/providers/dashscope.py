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

"""DashScope provider for LangExtract."""
# pylint: disable=duplicate-code

from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import Any, Iterator, Sequence

from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import patterns
from langextract.providers import router


@router.register(
    *patterns.DASHSCOPE_PATTERNS,
    priority=patterns.DASHSCOPE_PRIORITY,
)
@dataclasses.dataclass(init=False)
class DashScopeLanguageModel(base_model.BaseLanguageModel):
    """Language model inference using Alibaba DashScope API with structured output."""

    model_id: str = "qwen3-72b-instruct"
    api_key: str | None = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    format_type: data.FormatType = data.FormatType.JSON
    temperature: float | None = None
    max_workers: int = 10
    _client: Any = dataclasses.field(default=None, repr=False, compare=False)
    _extra_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def requires_fence_output(self) -> bool:
        """DashScope JSON mode returns raw JSON without fences."""
        if self.format_type == data.FormatType.JSON:
            return False
        return super().requires_fence_output

    def __init__(
        self,
        model_id: str = "qwen3-72b-instruct",
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        format_type: data.FormatType = data.FormatType.JSON,
        temperature: float | None = None,
        max_workers: int = 10,
        **kwargs,
    ) -> None:
        """Initialize the DashScope language model.

        Args:
          model_id: The DashScope model ID to use (e.g., 'qwen3-72b-instruct').
          api_key: API key for DashScope service.
          base_url: Base URL for DashScope service.
          format_type: Output format (JSON or YAML).
          temperature: Sampling temperature.
          max_workers: Maximum number of parallel API calls.
          **kwargs: Ignored extra parameters so callers can pass a superset of
            arguments shared across back-ends without raising ``TypeError``.
        """
        # Lazy import: OpenAI package required (since DashScope uses OpenAI compatible API)
        try:
            # pylint: disable=import-outside-toplevel
            import openai
        except ImportError as e:
            raise exceptions.InferenceConfigError(
                "DashScope provider requires openai package. "
                "Install with: pip install langextract[openai]"
            ) from e

        self.model_id = model_id
        # Allow env fallbacks
        try:
            import os as _os  # pylint: disable=import-outside-toplevel
        except Exception:  # pragma: no cover
            _os = None  # type: ignore

        self.api_key = api_key or (
            _os.environ.get("DASHSCOPE_API_KEY") if _os else None
        )
        self.base_url = base_url
        self.format_type = format_type
        self.temperature = temperature
        self.max_workers = max_workers

        # Debug switches (observability)
        self._lex_debug = False
        self._lex_debug_log: str | None = None
        try:
            self._lex_debug = bool(_os and _os.environ.get("LEX_DEBUG"))
            self._lex_debug_log = _os.environ.get("LEX_DEBUG_LOG") if _os else None
        except Exception:
            self._lex_debug = False
            self._lex_debug_log = None

        # Emit debug info about resolved configuration (without leaking secrets)
        if self._lex_debug:
            try:
                dbg = {
                    "resolved_model_id": self.model_id,
                    "resolved_base_url": self.base_url,
                    "has_api_key": bool(self.api_key),
                }
                msg = f"[langextract.dashscope] init: {dbg}"
                print(msg)
                if self._lex_debug_log:
                    with open(self._lex_debug_log, "a", encoding="utf-8") as fp:
                        fp.write(msg + "\n")
            except Exception:
                pass

        if not self.api_key:
            raise exceptions.InferenceConfigError("API key not provided.")

        # Initialize the OpenAI client (compatible with DashScope)
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        self._client = openai.OpenAI(**client_kwargs)

        if self._lex_debug:
            try:
                dbg2 = (
                    "[langextract.dashscope] client_ready" f" base_url={self.base_url}"
                )
                print(dbg2)
                if self._lex_debug_log:
                    with open(self._lex_debug_log, "a", encoding="utf-8") as fp:
                        fp.write(dbg2 + "\n")
            except Exception:
                pass

        super().__init__(
            constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
        )
        self._extra_kwargs = kwargs or {}

    def _process_single_prompt(
        self, prompt: str, config: dict
    ) -> core_types.ScoredOutput:
        """Process a single prompt and return a ScoredOutput."""
        try:
            system_message = ""
            if self.format_type == data.FormatType.JSON:
                system_message = (
                    "You are a helpful assistant that responds in JSON format."
                )
            elif self.format_type == data.FormatType.YAML:
                system_message = (
                    "You are a helpful assistant that responds in YAML format."
                )

            messages = [{"role": "user", "content": prompt}]
            if system_message:
                messages.insert(0, {"role": "system", "content": system_message})

            api_params = {
                "model": self.model_id,
                "messages": messages,
                "n": 1,
                # DashScope requires explicit disabling of thinking
                "extra_body": {"enable_thinking": False},
            }

            # Only set temperature if explicitly provided
            temp = config.get("temperature", self.temperature)
            if temp is not None:
                api_params["temperature"] = temp

            if self.format_type == data.FormatType.JSON:
                # Enables structured JSON output for compatible models
                api_params["response_format"] = {"type": "json_object"}

            if (v := config.get("max_output_tokens")) is not None:
                api_params["max_tokens"] = v
            if (v := config.get("top_p")) is not None:
                api_params["top_p"] = v
            for key in [
                "frequency_penalty",
                "presence_penalty",
                "seed",
                "stop",
            ]:
                if (v := config.get(key)) is not None:
                    api_params[key] = v

            response = self._client.chat.completions.create(**api_params)

            # Extract the response text using the v1.x response format
            output_text = response.choices[0].message.content

            return core_types.ScoredOutput(score=1.0, output=output_text)

        except Exception as e:
            if self._lex_debug:
                try:
                    em = f"[langextract.dashscope] exception: {type(e).__name__}: {e}"
                    print(em)
                    if self._lex_debug_log:
                        with open(self._lex_debug_log, "a", encoding="utf-8") as fp:
                            fp.write(em + "\n")
                except Exception:
                    pass
            raise exceptions.InferenceRuntimeError(
                f"DashScope API error: {str(e)}", original=e
            ) from e

    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[core_types.ScoredOutput]]:
        """Runs inference on a list of prompts via DashScope's API.

        Args:
          batch_prompts: A list of string prompts.
          **kwargs: Additional generation params (temperature, top_p, etc.)

        Yields:
          Lists of ScoredOutputs.
        """
        merged_kwargs = self.merge_kwargs(kwargs)

        config = {}

        # Only add temperature if it's not None
        temp = merged_kwargs.get("temperature", self.temperature)
        if temp is not None:
            config["temperature"] = temp
        if "max_output_tokens" in merged_kwargs:
            config["max_output_tokens"] = merged_kwargs["max_output_tokens"]
        if "top_p" in merged_kwargs:
            config["top_p"] = merged_kwargs["top_p"]

        # Forward DashScope-specific parameters
        for key in [
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
        ]:
            if key in merged_kwargs:
                config[key] = merged_kwargs[key]

        # Use parallel processing for batches larger than 1
        if len(batch_prompts) > 1 and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(batch_prompts))
            ) as executor:
                future_to_index = {
                    executor.submit(
                        self._process_single_prompt, prompt, config.copy()
                    ): i
                    for i, prompt in enumerate(batch_prompts)
                }

                results: list[core_types.ScoredOutput | None] = [None] * len(
                    batch_prompts
                )
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        raise exceptions.InferenceRuntimeError(
                            f"Parallel inference error: {str(e)}", original=e
                        ) from e

                for result in results:
                    if result is None:
                        raise exceptions.InferenceRuntimeError(
                            "Failed to process one or more prompts"
                        )
                    yield [result]
        else:
            # Sequential processing for single prompt or worker
            for prompt in batch_prompts:
                result = self._process_single_prompt(prompt, config.copy())
                yield [result]
