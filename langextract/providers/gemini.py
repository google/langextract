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

"""Gemini provider for LangExtract."""
# pylint: disable=duplicate-code

from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import Any, Final, Iterator, Sequence

from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import patterns
from langextract.providers import router
from langextract.providers import schemas

_API_CONFIG_KEYS: Final[set[str]] = {
    'response_mime_type',
    'response_schema',
    'safety_settings',
    'system_instruction',
    'tools',
    'stop_sequences',
    'candidate_count',
}


@router.register(
    *patterns.GEMINI_PATTERNS,
    priority=patterns.GEMINI_PRIORITY,
)
@dataclasses.dataclass(init=False)
class GeminiLanguageModel(base_model.BaseLanguageModel):
  """Language model inference using Google's Gemini API with structured output."""

  model_id: str = 'gemini-2.5-flash'
  api_key: str | None = None
  gemini_schema: schemas.gemini.GeminiSchema | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  max_workers: int = 10
  fence_output: bool = False
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @classmethod
  def get_schema_class(cls) -> type[schema.BaseSchema] | None:
    """Return the GeminiSchema class for structured output support.

    Returns:
      The GeminiSchema class that supports strict schema constraints.
    """
    return schemas.gemini.GeminiSchema

  def apply_schema(self, schema_instance: schema.BaseSchema | None) -> None:
    """Apply a schema instance to this provider.

    Args:
      schema_instance: The schema instance to apply, or None to clear.
    """
    super().apply_schema(schema_instance)
    # Keep provider behavior consistent with legacy path
    if isinstance(schema_instance, schemas.gemini.GeminiSchema):
      self.gemini_schema = schema_instance

  def __init__(
      self,
      model_id: str = 'gemini-2.5-flash',
      api_key: str | None = None,
      gemini_schema: schemas.gemini.GeminiSchema | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      fence_output: bool = False,
      **kwargs,
  ) -> None:
    """Initialize the Gemini language model.

    Args:
      model_id: The Gemini model ID to use.
      api_key: API key for Gemini service.
      gemini_schema: Optional schema for structured output.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      fence_output: Whether to wrap output in markdown fences (ignored,
        Gemini handles this based on schema).
      **kwargs: Additional Gemini API parameters. Only allowlisted keys are
        forwarded to the API (response_schema, response_mime_type, tools,
        safety_settings, stop_sequences, candidate_count, system_instruction).
        See https://ai.google.dev/api/generate-content for details.
    """
    try:
      # pylint: disable=import-outside-toplevel
      from google import genai
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'Failed to import google-genai. Reinstall: pip install langextract'
      ) from e

    self.model_id = model_id
    self.api_key = api_key
    self.gemini_schema = gemini_schema
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self.fence_output = fence_output

    if not self.api_key:
      raise exceptions.InferenceConfigError('API key not provided.')

    self._client = genai.Client(api_key=self.api_key)

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )
    self._extra_kwargs = {
        k: v for k, v in (kwargs or {}).items() if k in _API_CONFIG_KEYS
    }

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> core_types.ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      # Apply stored kwargs that weren't already set in config
      for key, value in self._extra_kwargs.items():
        if key not in config and value is not None:
          config[key] = value

      if self.gemini_schema:
        # Structured output requires JSON format
        if self.format_type != data.FormatType.JSON:
          raise exceptions.InferenceConfigError(
              'Gemini structured output only supports JSON format. '
              'Set format_type=JSON or use_schema_constraints=False.'
          )
        config.setdefault('response_mime_type', 'application/json')
        config.setdefault('response_schema', self.gemini_schema.schema_dict)

      response = self._client.models.generate_content(
          model=self.model_id, contents=prompt, config=config
      )

      return core_types.ScoredOutput(score=1.0, output=response.text)

    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'Gemini API error: {str(e)}', original=e
      ) from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[core_types.ScoredOutput]]:
    """Runs inference on a list of prompts via Gemini's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, top_k, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    merged_kwargs = self.merge_kwargs(kwargs)

    config = {
        'temperature': merged_kwargs.get('temperature', self.temperature),
    }
    if 'max_output_tokens' in merged_kwargs:
      config['max_output_tokens'] = merged_kwargs['max_output_tokens']
    if 'top_p' in merged_kwargs:
      config['top_p'] = merged_kwargs['top_p']
    if 'top_k' in merged_kwargs:
      config['top_k'] = merged_kwargs['top_k']

    handled_keys = {'temperature', 'max_output_tokens', 'top_p', 'top_k'}
    for key, value in merged_kwargs.items():
      if (
          key not in handled_keys
          and key in _API_CONFIG_KEYS
          and value is not None
      ):
        config[key] = value

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
                f'Parallel inference error: {str(e)}', original=e
            ) from e

        for result in results:
          if result is None:
            raise exceptions.InferenceRuntimeError(
                'Failed to process one or more prompts'
            )
          yield [result]
    else:
      # Sequential processing for single prompt or worker
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]  # pylint: disable=duplicate-code
