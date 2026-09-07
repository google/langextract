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

"""Anthropic (Claude) provider for LangExtract."""
# pylint: disable=duplicate-code

from __future__ import annotations

import concurrent.futures
import dataclasses
import os
from typing import Any, Iterator, Sequence

from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import latest_models
from langextract.providers import patterns
from langextract.providers import router


def _response_text(response: Any) -> str:
  """Flatten Anthropic message content into a single string."""
  parts = getattr(response, 'content', None)
  if isinstance(parts, str):
    return parts
  if not parts:
    return ''
  texts: list[str] = []
  for block in parts:
    text = getattr(block, 'text', None)
    if text:
      texts.append(text)
      continue
    if isinstance(block, dict) and block.get('text'):
      texts.append(str(block['text']))
  return ''.join(texts)


@router.register(
    *patterns.ANTHROPIC_PATTERNS,
    priority=patterns.ANTHROPIC_PRIORITY,
)
@dataclasses.dataclass(init=False)
class AnthropicLanguageModel(base_model.BaseLanguageModel):
  """Language model inference using Anthropic's Claude Messages API."""

  model_id: str = latest_models.CLAUDE_DEFAULT
  api_key: str | None = None
  base_url: str | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float | None = 0.0
  max_output_tokens: int = 8192
  max_workers: int = 10
  _client: Any = dataclasses.field(default=None, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = latest_models.CLAUDE_DEFAULT,
      api_key: str | None = None,
      base_url: str | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float | None = 0.0,
      max_output_tokens: int = 8192,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the Anthropic language model.

    Args:
      model_id: Claude model ID (claude-sonnet-5, claude-opus-5, ...).
      api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY.
      base_url: Optional Anthropic-compatible base URL.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_output_tokens: Maximum tokens in the completion (required by API).
      max_workers: Maximum number of parallel API calls.
      **kwargs: Extra Anthropic Messages parameters.
    """
    try:
      # pylint: disable=import-outside-toplevel
      import anthropic
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'Anthropic provider requires anthropic package. '
          'Install with: pip install anthropic'
      ) from e

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

    self.model_id = model_id or latest_models.CLAUDE_DEFAULT
    self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    self.base_url = base_url or os.environ.get('ANTHROPIC_BASE_URL')
    self.format_type = format_type
    self.temperature = temperature
    self.max_output_tokens = max_output_tokens
    self.max_workers = max_workers
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      raise exceptions.InferenceConfigError(
          'API key not provided for Anthropic/Claude. Set ANTHROPIC_API_KEY.'
      )

    client_kwargs: dict[str, Any] = {'api_key': self.api_key}
    if self.base_url:
      client_kwargs['base_url'] = self.base_url
    self._client = anthropic.Anthropic(**client_kwargs)

  def _system_message(self) -> str:
    if self.format_type == data.FormatType.JSON:
      return 'You are a helpful assistant that responds in JSON format.'
    if self.format_type == data.FormatType.YAML:
      return 'You are a helpful assistant that responds in YAML format.'
    return ''

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> core_types.ScoredOutput:
    """Sends one prompt while preserving provider-specific error types."""
    try:
      max_tokens = int(
          config.get('max_output_tokens', self.max_output_tokens) or 8192
      )
      api_params: dict[str, Any] = {
          'model': self.model_id,
          'max_tokens': max_tokens,
          'messages': [{'role': 'user', 'content': prompt}],
      }
      system = self._system_message()
      if system:
        api_params['system'] = system
      temp = config.get('temperature', self.temperature)
      if temp is not None:
        api_params['temperature'] = temp
      if (v := config.get('top_p')) is not None:
        api_params['top_p'] = v
      if (v := config.get('stop')) is not None:
        api_params['stop_sequences'] = v if isinstance(v, list) else [v]

      response = self._client.messages.create(**api_params)
      return core_types.ScoredOutput(score=1.0, output=_response_text(response))
    except exceptions.InferenceConfigError:
      raise
    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'Anthropic API error: {str(e)}', original=e
      ) from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[core_types.ScoredOutput]]:
    """Runs inference on a list of prompts via Anthropic's Messages API."""
    merged_kwargs = self.merge_kwargs(kwargs)
    config: dict[str, Any] = {}
    temp = merged_kwargs.get('temperature', self.temperature)
    if temp is not None:
      config['temperature'] = temp
    if 'max_output_tokens' in merged_kwargs:
      config['max_output_tokens'] = merged_kwargs['max_output_tokens']
    if 'top_p' in merged_kwargs:
      config['top_p'] = merged_kwargs['top_p']
    if 'stop' in merged_kwargs:
      config['stop'] = merged_kwargs['stop']

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
          except exceptions.InferenceConfigError:
            raise
          except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f'Parallel inference error: {str(e)}', original=e
            ) from e
        for result in results:
          if result is None:
            raise exceptions.InferenceRuntimeError(
                'Failed to process one or more prompts'
            )
          yield [result]
    else:
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]
