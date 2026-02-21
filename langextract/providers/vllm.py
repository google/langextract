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

"""vLLM provider for LangExtract.

This provider targets vLLM's OpenAI-compatible API server:
https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urljoin
from urllib.parse import urlparse
import warnings

import requests

from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import patterns
from langextract.providers import router

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_TIMEOUT_S = 120


def _normalize_base_url(base_url: str) -> str:
  base = (base_url or "").strip() or _DEFAULT_BASE_URL
  base = base[:-1] if base.endswith("/") else base
  if not base.endswith("/v1"):
    base = base + "/v1"
  return base


@router.register(
    *patterns.VLLM_PATTERNS,
    priority=patterns.VLLM_PRIORITY,
)
@dataclasses.dataclass(init=False)
class VLLMLanguageModel(base_model.BaseLanguageModel):
  """Language model inference via vLLM OpenAI-compatible server."""

  model_id: str
  base_url: str = _DEFAULT_BASE_URL
  api_key: str | None = None
  organization: str | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float | None = None
  max_workers: int = 10
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  # Authentication
  _auth_scheme: str = "Bearer"
  _auth_header: str = "Authorization"

  def __init__(
      self,
      model_id: str,
      base_url: str = _DEFAULT_BASE_URL,
      api_key: str | None = None,
      organization: str | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float | None = None,
      max_workers: int = 10,
      auth_scheme: str = "Bearer",
      auth_header: str = "Authorization",
      timeout: int | None = None,
      **kwargs,
  ) -> None:
    """Initialize the vLLM provider.

    Args:
      model_id: vLLM model name as exposed by the server.
      base_url: Base URL to the OpenAI-compatible API (usually ends with /v1).
      api_key: Optional API key for proxied/secured deployments.
      organization: Optional organization header for OpenAI-compatible gateways.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Parallelism for prompt batches.
      auth_scheme: Authorization scheme, defaults to "Bearer".
      auth_header: Authorization header name, defaults to "Authorization".
      timeout: Request timeout in seconds (defaults to 120).
      **kwargs: Additional OpenAI-compatible params forwarded in request body.
    """
    self._requests = requests
    self.model_id = model_id
    self.base_url = _normalize_base_url(base_url)
    self.api_key = api_key
    self.organization = organization
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self._auth_scheme = auth_scheme
    self._auth_header = auth_header
    self._timeout_s = timeout if timeout is not None else _DEFAULT_TIMEOUT_S

    if self.api_key:
      host = urlparse(self.base_url).hostname
      if host in ("localhost", "127.0.0.1", "::1"):
        warnings.warn(
            "API key provided for localhost vLLM server. "
            "This is typically only needed for proxied instances.",
            UserWarning,
        )

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )
    self._extra_kwargs = kwargs or {}

  def _chat_completions_url(self) -> str:
    base = self.base_url
    if not base.endswith("/"):
      base = base + "/"
    return urljoin(base, "chat/completions")

  def _headers(self) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if self.organization:
      headers["OpenAI-Organization"] = self.organization
    if self.api_key:
      if self._auth_scheme:
        headers[self._auth_header] = f"{self._auth_scheme} {self.api_key}"
      else:
        headers[self._auth_header] = self.api_key
    return headers

  def _build_messages(self, prompt: str) -> list[dict[str, Any]]:
    system_message = ""
    if self.format_type == data.FormatType.JSON:
      system_message = "You are a helpful assistant that responds in JSON."
    elif self.format_type == data.FormatType.YAML:
      system_message = "You are a helpful assistant that responds in YAML."

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    if system_message:
      messages.insert(0, {"role": "system", "content": system_message})
    return messages

  def _process_single_prompt(
      self, prompt: str, config: Mapping[str, Any]
  ) -> core_types.ScoredOutput:
    url = self._chat_completions_url()

    api_params: dict[str, Any] = {
        "model": self.model_id,
        "messages": self._build_messages(prompt),
        "n": 1,
    }

    temp = config.get("temperature", self.temperature)
    if temp is not None:
      api_params["temperature"] = temp

    # Map common params used by LangExtract to OpenAI-compatible fields.
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
        "response_format",
    ]:
      if (v := config.get(key)) is not None:
        api_params[key] = v

    # Forward any extra kwargs captured at init time, unless overridden.
    for k, v in (self._extra_kwargs or {}).items():
      if k not in api_params and v is not None:
        api_params[k] = v

    try:
      resp = self._requests.post(
          url,
          headers=self._headers(),
          json=api_params,
          timeout=self._timeout_s,
      )
    except self._requests.exceptions.RequestException as e:
      raise exceptions.InferenceRuntimeError(
          f"vLLM request failed: {e}", original=e, provider="vLLM"
      ) from e

    resp.encoding = "utf-8"
    if resp.status_code != 200:
      try:
        payload = resp.json()
      except ValueError:
        payload = {"raw": resp.text[:2000]}
      raise exceptions.InferenceRuntimeError(
          f"vLLM API error: status={resp.status_code}, body={payload}",
          provider="vLLM",
      )

    try:
      payload = resp.json()
    except ValueError as e:
      raise exceptions.InferenceRuntimeError(
          f"vLLM returned non-JSON response: {resp.text[:2000]}",
          original=e,
          provider="vLLM",
      ) from e

    try:
      content = payload["choices"][0]["message"]["content"]
    except Exception as e:  # pylint: disable=broad-exception-caught
      raise exceptions.InferenceRuntimeError(
          f"Unexpected vLLM response shape: {json.dumps(payload)[:2000]}",
          original=e,
          provider="vLLM",
      ) from e

    return core_types.ScoredOutput(score=1.0, output=content)

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[core_types.ScoredOutput]]:
    merged_kwargs = self.merge_kwargs(kwargs)
    config: dict[str, Any] = {}

    temp = merged_kwargs.get("temperature", self.temperature)
    if temp is not None:
      config["temperature"] = temp

    for key in [
        "max_output_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
        "logprobs",
        "top_logprobs",
        "response_format",
    ]:
      if key in merged_kwargs:
        config[key] = merged_kwargs[key]

    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[core_types.ScoredOutput | None] = [None] * len(
            batch_prompts
        )
        for future in concurrent.futures.as_completed(future_to_index):
          idx = future_to_index[future]
          try:
            results[idx] = future.result()
          except Exception as e:  # pylint: disable=broad-exception-caught
            raise exceptions.InferenceRuntimeError(
                f"Parallel inference error: {e}", original=e, provider="vLLM"
            ) from e

        for result in results:
          if result is None:
            raise exceptions.InferenceRuntimeError(
                "Failed to process one or more prompts", provider="vLLM"
            )
          yield [result]
    else:
      for prompt in batch_prompts:
        yield [self._process_single_prompt(prompt, config)]

