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

"""llama.cpp provider for LangExtract.

This provider targets llama.cpp's lightweight HTTP server (llama-server),
which exposes an OpenAI-compatible API at /v1/chat/completions. See:
https://github.com/ggml-org/llama.cpp

Usage examples:
  - Start server: `llama-server -m model.gguf --port 8080`
  - Use via LangExtract:
      lx.extract(..., model_id="llama-cpp:Qwen2.5-1.5B-Instruct",
                 base_url="http://localhost:8080", api_key=None)

Notes:
  - api_key is optional. llama.cpp accepts requests without authentication by
    default unless configured otherwise.
  - Structured JSON output is requested by setting response_format when
    format_type is JSON.
"""
# pylint: disable=duplicate-code

from __future__ import annotations

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
    *patterns.LLAMA_CPP_PATTERNS,
    priority=patterns.LLAMA_CPP_PRIORITY,
)
@dataclasses.dataclass(init=False)
class LlamaCppLanguageModel(base_model.BaseLanguageModel):
  """Language model inference via llama.cpp's OpenAI-compatible endpoint."""

  model_id: str = 'unknown'
  api_key: str | None = None
  base_url: str | None = 'http://localhost:8080'
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float | None = None
  max_workers: int = 1
  _client: Any = dataclasses.field(default=None, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @classmethod
  def get_schema_class(cls) -> type[schema.BaseSchema] | None:
    # JSON mode can be requested; treat similarly to FormatModeSchema
    return schema.FormatModeSchema

  @property
  def requires_fence_output(self) -> bool:
    # When JSON mode requested, the server should return raw JSON
    if self.format_type == data.FormatType.JSON:
      return False
    return super().requires_fence_output

  def __init__(
      self,
      model_id: str,
      api_key: str | None = None,
      base_url: str | None = 'http://localhost:8080',
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float | None = None,
      max_workers: int = 1,
      **kwargs,
  ) -> None:
    """Initialize the llama.cpp language model.

    Args:
      model_id: Model name to pass through to llama.cpp server.
      api_key: Optional API key (llama.cpp typically doesn't require one).
      base_url: Base URL of llama.cpp server (default http://localhost:8080).
      format_type: Output format (JSON or YAML). YAML won't be constrained.
      temperature: Sampling temperature.
      max_workers: Not used currently; kept for parity with OpenAI provider.
      **kwargs: Ignored extra parameters to be lenient with callers.
    """
    # Lazy import to avoid hard dep
    try:
      # pylint: disable=import-outside-toplevel
      import openai
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'llama.cpp provider requires openai package for client. '
          'Install with: pip install langextract[openai]'
      ) from e

    # Allow scheme prefixes in model_id and strip them
    # e.g., "llama-cpp:my-model" or "llama.cpp:my-model"
    if model_id.startswith('llama-cpp:'):
      effective_model_id = model_id.split(':', 1)[1]
    elif model_id.startswith('llama.cpp:'):
      effective_model_id = model_id.split(':', 1)[1]
    else:
      effective_model_id = model_id

    self.model_id = effective_model_id
    self.api_key = api_key
    self.base_url = base_url or 'http://localhost:8080'
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers

    # Unlike OpenAI, api_key is optional; some deployments may enforce auth
    client_args: dict[str, Any] = {}
    if self.api_key:
      client_args['api_key'] = self.api_key
    if self.base_url:
      client_args['base_url'] = f"{self.base_url.rstrip('/')}/v1"

    self._client = openai.OpenAI(**client_args)

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )
    self._extra_kwargs = kwargs or {}

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> core_types.ScoredOutput:
    """Process a single prompt via llama.cpp's OpenAI-compatible endpoint."""
    try:
      system_message = ''
      if self.format_type == data.FormatType.JSON:
        system_message = 'You are a helpful assistant that responds in JSON.'

      messages = [{'role': 'user', 'content': prompt}]
      if system_message:
        messages.insert(0, {'role': 'system', 'content': system_message})

      api_params: dict[str, Any] = {
          'model': self.model_id,
          'messages': messages,
          'n': 1,
      }

      temp = config.get('temperature', self.temperature)
      if temp is not None:
        api_params['temperature'] = temp

      if self.format_type == data.FormatType.JSON:
        api_params['response_format'] = {'type': 'json_object'}

      if (v := config.get('max_output_tokens')) is not None:
        api_params['max_tokens'] = v
      if (v := config.get('top_p')) is not None:
        api_params['top_p'] = v

      response = self._client.chat.completions.create(**api_params)
      output_text = response.choices[0].message.content
      return core_types.ScoredOutput(score=1.0, output=output_text)
    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'llama.cpp API error: {str(e)}', original=e
      ) from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[core_types.ScoredOutput]]:
    """Inference via llama.cpp's OpenAI-compatible endpoint."""
    merged_kwargs = self.merge_kwargs(kwargs)

    config: dict[str, Any] = {}
    temp = merged_kwargs.get('temperature', self.temperature)
    if temp is not None:
      config['temperature'] = temp
    if 'max_output_tokens' in merged_kwargs:
      config['max_output_tokens'] = merged_kwargs['max_output_tokens']
    if 'top_p' in merged_kwargs:
      config['top_p'] = merged_kwargs['top_p']

    for prompt in batch_prompts:
      result = self._process_single_prompt(prompt, config.copy())
      yield [result]


