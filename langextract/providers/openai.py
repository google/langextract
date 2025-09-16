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

"""OpenAI provider for LangExtract."""
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
    *patterns.OPENAI_PATTERNS,
    priority=patterns.OPENAI_PRIORITY,
)
@dataclasses.dataclass(init=False)
class OpenAILanguageModel(base_model.BaseLanguageModel):
    """Language model inference using OpenAI's API with structured output."""

    model_id: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    api_version: str | None = None
    format_type: data.FormatType = data.FormatType.JSON
    temperature: float | None = None
    max_workers: int = 10
    _client: Any = dataclasses.field(default=None, repr=False, compare=False)
    _extra_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )
    _is_azure: bool = dataclasses.field(default=False, repr=False, compare=False)
    _azure_endpoint: str | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    _api_version: str | None = dataclasses.field(
        default=None, repr=False, compare=False
    )

    @property
    def requires_fence_output(self) -> bool:
        """OpenAI JSON mode returns raw JSON without fences."""
        if self.format_type == data.FormatType.JSON:
            return False
        return super().requires_fence_output

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        api_version: str | None = None,
        format_type: data.FormatType = data.FormatType.JSON,
        temperature: float | None = None,
        max_workers: int = 10,
        **kwargs,
    ) -> None:
        """Initialize the OpenAI language model.

        Args:
          model_id: The OpenAI model ID to use (e.g., 'gpt-4o-mini', 'gpt-4o').
          api_key: API key for OpenAI service.
          base_url: Base URL for OpenAI service.
          organization: Optional OpenAI organization ID.
          format_type: Output format (JSON or YAML).
          temperature: Sampling temperature.
          max_workers: Maximum number of parallel API calls.
          **kwargs: Ignored extra parameters so callers can pass a superset of
            arguments shared across back-ends without raising ``TypeError``.
        """
        # Lazy import: OpenAI package required
        try:
            # pylint: disable=import-outside-toplevel
            import openai
        except ImportError as e:
            raise exceptions.InferenceConfigError(
                "OpenAI provider requires openai package. "
                "Install with: pip install langextract[openai]"
            ) from e

        self.model_id = model_id
        # Allow env fallbacks for Azure/OpenAI
        try:
            import os as _os  # pylint: disable=import-outside-toplevel
        except Exception:  # pragma: no cover
            _os = None  # type: ignore

        self.api_key = api_key or (
            (_os.environ.get("AZURE_OPENAI_API_KEY") if _os else None)
            or (_os.environ.get("OPENAI_API_KEY") if _os else None)
        )
        self.base_url = base_url or (
            (_os.environ.get("AZURE_OPENAI_ENDPOINT") if _os else None)
            or (_os.environ.get("OPENAI_BASE_URL") if _os else None)
        )
        self.organization = organization
        self.format_type = format_type
        self.temperature = temperature
        self.max_workers = max_workers
        self.api_version = api_version or (kwargs.get("api_version"))

        # Debug switches (observability)
        self._lex_debug = False
        self._lex_debug_log: str | None = None
        try:
            self._lex_debug = bool(_os and _os.environ.get("LEX_DEBUG"))
            self._lex_debug_log = _os.environ.get("LEX_DEBUG_LOG") if _os else None
        except Exception:
            self._lex_debug = False
            self._lex_debug_log = None

        # Built-in fallbacks by model_id so callers don't need to pass credentials every time
        try:
            mid = (model_id or "").lower()
            if (
                (not self.base_url)
                or (not self.api_key)
                or (not self.api_version and mid.startswith("gpt-4o"))
            ):
                # DashScope (Bailian) - OpenAI compatible
                if mid.startswith("qwen3-"):
                    self.base_url = (
                        self.base_url
                        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )
                    self.api_key = self.api_key or "your-key"
                # Volcengine (Ark) - OpenAI compatible
                elif mid.startswith("doubao"):
                    self.base_url = (
                        self.base_url or "https://ark.cn-beijing.volces.com/api/v3"
                    )
                    self.api_key = self.api_key or "your-key"
                # Azure OpenAI (deployment name typically used as model_id)
                elif mid.startswith("gpt-4o"):
                    self.base_url = self.base_url or "https://coe0118.openai.azure.com/"
                    self.api_key = self.api_key or "your-key"
                    # Default api-version for Azure
                    if not self.api_version:
                        self.api_version = "your-version"
        except Exception:
            pass

        # Emit debug info about resolved configuration (without leaking secrets)
        if self._lex_debug:
            try:
                dbg = {
                    "resolved_model_id": self.model_id,
                    "resolved_base_url": self.base_url,
                    "has_api_key": bool(self.api_key),
                    "api_version": self.api_version,
                }
                msg = f"[langextract.openai] init: {dbg}"
                print(msg)
                if self._lex_debug_log:
                    with open(self._lex_debug_log, "a", encoding="utf-8") as fp:
                        fp.write(msg + "\n")
            except Exception:
                pass

        if not self.api_key:
            raise exceptions.InferenceConfigError("API key not provided.")

        # Initialize the OpenAI client
        # 仅当未显式提供非 Azure 的 base_url 时，才回退到 AZURE_OPENAI_ENDPOINT 判定
        is_azure = False
        if self.base_url and "azure.com" in self.base_url:
            is_azure = True
        elif (not self.base_url) and _os and _os.environ.get("AZURE_OPENAI_ENDPOINT"):
            is_azure = True

        self._is_azure = is_azure
        if is_azure:
            # Use AzureOpenAI client with proper parameters
            azure_endpoint = self.base_url or (
                _os.environ.get("AZURE_OPENAI_ENDPOINT") if _os else None
            )
            api_ver = self.api_version or (
                (_os.environ.get("AZURE_OPENAI_API_VERSION") if _os else None)
            )
            if not azure_endpoint:
                raise exceptions.InferenceConfigError("Azure endpoint not provided.")
            if not api_ver:
                raise exceptions.InferenceConfigError("Azure API version not provided.")
            # 记录以供 REST 路径使用
            self._azure_endpoint = azure_endpoint.rstrip("/")
            self._api_version = api_ver
            # 为确保使用带 api-version 的部署级 REST 路径，这里不初始化 SDK 客户端，统一走 REST
            self._client = None
        else:
            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "organization": self.organization,
            }
            self._client = openai.OpenAI(**client_kwargs)

        if self._lex_debug:
            try:
                dbg2 = (
                    "[langextract.openai] client_ready"
                    f" azure={self._is_azure} base_url={self.base_url} api_version={self.api_version}"
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
                "logprobs",
                "top_logprobs",
            ]:
                if (v := config.get(key)) is not None:
                    api_params[key] = v

            # DashScope(百炼) 兼容模式：非 streaming 必须显式关闭思维链
            try:
                is_dashscope = bool(
                    self.base_url and "dashscope.aliyuncs.com" in self.base_url
                )
            except Exception:
                is_dashscope = False
            if (not self._is_azure) and (
                is_dashscope or self.model_id.lower().startswith("qwen")
            ):
                # OpenAI SDK v1 需要通过 extra_body 传递自定义字段
                extra_body = api_params.get("extra_body", {})
                extra_body.update({"enable_thinking": False})
                api_params["extra_body"] = extra_body

            if self._is_azure:
                # 统一走 Azure 部署级 REST，保证携带 api-version
                response = self._azure_chat_completion_http(api_params)
            else:
                response = self._client.chat.completions.create(**api_params)

            # Extract the response text using the v1.x response format
            output_text = response.choices[0].message.content

            return core_types.ScoredOutput(score=1.0, output=output_text)

        except Exception as e:
            if self._lex_debug:
                try:
                    em = f"[langextract.openai] exception: {type(e).__name__}: {e}"
                    print(em)
                    if self._lex_debug_log:
                        with open(self._lex_debug_log, "a", encoding="utf-8") as fp:
                            fp.write(em + "\n")
                except Exception:
                    pass
            raise exceptions.InferenceRuntimeError(
                f"OpenAI API error: {str(e)}", original=e
            ) from e

    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[core_types.ScoredOutput]]:
        """Runs inference on a list of prompts via OpenAI's API.

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

        # Forward OpenAI-specific parameters
        for key in [
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
            "logprobs",
            "top_logprobs",
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
                yield [result]  # pylint: disable=duplicate-code

    def _azure_chat_completion_http(self, api_params: dict) -> Any:
        """调用 Azure OpenAI Chat Completions（部署级 REST 接口）。

        期望 api_params 含有：model(=deployment), messages, temperature/max_tokens 等。
        """
        try:
            import requests  # pylint: disable=import-outside-toplevel
        except Exception as e:  # pragma: no cover
            raise exceptions.InferenceRuntimeError("requests not installed") from e

        if not self._azure_endpoint or not self._api_version:
            raise exceptions.InferenceConfigError("Azure endpoint/api-version not set")

        deployment = api_params.get("model")
        if not deployment:
            raise exceptions.InferenceConfigError("Azure deployment (model) not set")

        url = f"{self._azure_endpoint}/openai/deployments/{deployment}/chat/completions"
        params = {"api-version": self._api_version}
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        payload = {k: v for k, v in api_params.items() if k != "model"}

        if self._lex_debug:
            try:
                dbg = f"[langextract.openai] azure_http url={url} params={params}"
                print(dbg)
                if self._lex_debug_log:
                    with open(self._lex_debug_log, "a", encoding="utf-8") as fp:
                        fp.write(dbg + "\n")
            except Exception:
                pass
        resp = requests.post(
            url, params=params, headers=headers, json=payload, timeout=60
        )
        if resp.status_code >= 400:
            raise exceptions.InferenceRuntimeError(
                f"Azure REST error {resp.status_code}: {resp.text}"
            )
        data = resp.json()

        # 适配到 openai v1 响应对象的最小字段
        class _Msg:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, content):
                self.message = _Msg(content)

        class _Resp:
            def __init__(self, content):
                self.choices = [_Choice(content)]

        first = ""
        try:
            first = data["choices"][0]["message"]["content"]
        except Exception:
            first = ""
        return _Resp(first)
