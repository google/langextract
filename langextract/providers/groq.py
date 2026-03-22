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

"""Groq provider for LangExtract."""

# pylint: disable=duplicate-code

from __future__ import annotations

import dataclasses
import os
from typing import Any, Iterator, Sequence

import requests

from langextract.core import base_model
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import patterns
from langextract.providers import router


@router.register(
    *patterns.GROQ_PATTERNS,
    priority=patterns.GROQ_PRIORITY,
)
@dataclasses.dataclass(init=False)
class GroqLanguageModel(base_model.BaseLanguageModel):
  """Language model inference using Groq's OpenAI-compatible API."""

  model_id: str = 'groq/llama-3.1-8b-instant'
  api_key: str | None = None
  base_url: str = 'https://api.groq.com/openai/v1'
  temperature: float | None = None
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = 'groq/llama-3.1-8b-instant',
      api_key: str | None = None,
      base_url: str = 'https://api.groq.com/openai/v1',
      temperature: float | None = None,
      **kwargs,
  ) -> None:
    """Initialize the Groq language model.

    Args:
      model_id: The Groq model ID to use (e.g., 'groq/llama-3.1-8b-instant').
      api_key: API key for Groq service (or set GROQ_API_KEY).
      base_url: Base URL for Groq OpenAI-compatible API.
      temperature: Sampling temperature.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    self.model_id = model_id
    self.api_key = api_key
    self.base_url = base_url
    self.temperature = temperature
    self._extra_kwargs = kwargs or {}

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

  def _resolve_api_key(self) -> str:
    key = self.api_key or os.getenv('GROQ_API_KEY')
    if not key:
      raise exceptions.InferenceConfigError(
          'Groq API key not found. Set GROQ_API_KEY or pass api_key=...'
      )
    return key

  def _resolve_groq_model_name(self) -> str:
    if self.model_id.startswith('groq/'):
      return self.model_id.split('/', 1)[1]
    return self.model_id

  def _call_groq_chat_completions(self, prompt: str, config: dict) -> str:
    api_key = self._resolve_api_key()
    model_name = self._resolve_groq_model_name()
    url = f'{self.base_url}/chat/completions'

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': prompt}],
        'n': 1,
    }

    if (v := config.get('temperature', self.temperature)) is not None:
      payload['temperature'] = v
      
    if (v := config.get('max_output_tokens')) is not None:
      payload['max_tokens'] = v
    if (v := config.get('top_p')) is not None:
      payload['top_p'] = v
    if (v := config.get('stop')) is not None:
      payload['stop'] = v

    # Allow extra kwargs from init() to pass through (non-None only)
    for k, v in (self._extra_kwargs or {}).items():
      if v is not None and k not in payload:
        payload[k] = v

    try:
      resp = requests.post(url, headers=headers, json=payload, timeout=60)
      resp.raise_for_status()
      data = resp.json()
      return data['choices'][0]['message']['content']
    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'Groq API error: {str(e)}', original=e
      ) from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[core_types.ScoredOutput]]:
    """Runs inference on a list of prompts via Groq's API."""
    merged_kwargs = self.merge_kwargs(kwargs)

    config: dict[str, Any] = {}
    temp = merged_kwargs.get('temperature', self.temperature)
    if temp is not None:
      config['temperature'] = temp

    for key in ['max_output_tokens', 'top_p', 'stop']:
      if key in merged_kwargs:
        config[key] = merged_kwargs[key]

    for prompt in batch_prompts:
      text = self._call_groq_chat_completions(prompt, config)
      yield [core_types.ScoredOutput(score=1.0, output=text)]
