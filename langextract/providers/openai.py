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
import warnings

from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import patterns
from langextract.providers import router
from langextract.providers import schemas


@router.register(
    *patterns.OPENAI_PATTERNS,
    priority=patterns.OPENAI_PRIORITY,
)
@dataclasses.dataclass(init=False)
class OpenAILanguageModel(base_model.BaseLanguageModel):
  """Language model inference using OpenAI's API with structured output."""

  model_id: str = 'gpt-4o-mini'
  api_key: str | None = None
  base_url: str | None = None
  organization: str | None = None
  openai_schema: schemas.openai.OpenAISchema | None = dataclasses.field(
      default=None, repr=False, compare=False
  )
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float | None = None
  max_workers: int = 10
  _client: Any = dataclasses.field(default=None, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @classmethod
  def get_schema_class(cls) -> type[schema.BaseSchema] | None:
    """Return the OpenAISchema class for structured output support."""
    return schemas.openai.OpenAISchema

  def apply_schema(self, schema_instance: schema.BaseSchema | None) -> None:
    """Applies an OpenAI schema instance to this provider.

    Args:
      schema_instance: An OpenAISchema to enforce, or None to clear.

    Raises:
      InferenceConfigError: if schema_instance is a non-OpenAI BaseSchema
        subclass, or if applying an OpenAI schema would conflict with a
        non-JSON format_type.
    """
    if schema_instance is None:
      self.openai_schema = None
    elif isinstance(schema_instance, schemas.openai.OpenAISchema):
      if self.format_type != data.FormatType.JSON:
        raise exceptions.InferenceConfigError(
            schemas.openai.JSON_SCHEMA_FORMAT_ERROR
        )
      self.openai_schema = schema_instance
    else:
      raise exceptions.InferenceConfigError(
          'OpenAILanguageModel only accepts OpenAISchema instances; got '
          f'{type(schema_instance).__name__}. Use the matching provider '
          'for this schema or construct an OpenAISchema via '
          'OpenAISchema.from_examples.'
      )
    super().apply_schema(schema_instance)

  @property
  def requires_fence_output(self) -> bool:
    """OpenAI JSON mode returns raw JSON unless callers override fences."""
    if (
        self._fence_output_override is None
        and self.format_type == data.FormatType.JSON
    ):
      return False
    return super().requires_fence_output

  def __init__(
      self,
      model_id: str = 'gpt-4o-mini',
      api_key: str | None = None,
      base_url: str | None = None,
      organization: str | None = None,
      openai_schema: schemas.openai.OpenAISchema | None = None,
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
      openai_schema: Optional schema for structured output.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising TypeError.
    """
    try:
      # pylint: disable=import-outside-toplevel
      import openai
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'OpenAI provider requires openai package. '
          'Install with: pip install langextract[openai]'
      ) from e

    # Constructor-provided schemas use BaseLanguageModel state when applied.
    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

    self.model_id = model_id
    self.api_key = api_key
    self.base_url = base_url
    self.organization = organization
    self.openai_schema = None
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      raise exceptions.InferenceConfigError('API key not provided.')

    if openai_schema is not None:
      self.apply_schema(openai_schema)

    # Keep SDK initialization after schema validation so LangExtract reports
    # configuration errors before any client-side transport checks.
    self._client = openai.OpenAI(
        api_key=self.api_key,
        base_url=self.base_url,
        organization=self.organization,
    )

  def _validate_schema_config(self) -> None:
    """Rejects schema settings the OpenAI API cannot honor."""
    if self.openai_schema and self.format_type != data.FormatType.JSON:
      raise exceptions.InferenceConfigError(
          schemas.openai.JSON_SCHEMA_FORMAT_ERROR
      )

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> core_types.ScoredOutput:
    """Sends one prompt while preserving provider-specific error types."""
    try:
      normalized_config = config.copy()

      system_message = ''
      if self.format_type == data.FormatType.JSON:
        system_message = (
            'You are a helpful assistant that responds in JSON format.'
        )
      elif self.format_type == data.FormatType.YAML:
        system_message = (
            'You are a helpful assistant that responds in YAML format.'
        )

      messages = [{'role': 'user', 'content': prompt}]
      if system_message:
        messages.insert(0, {'role': 'system', 'content': system_message})

      api_params = {
          'model': self.model_id,
          'messages': messages,
          'n': 1,
      }

      temp = normalized_config.get('temperature', self.temperature)
      if temp is not None:
        api_params['temperature'] = temp

      runtime_response_format = normalized_config.get('response_format')
      if self.openai_schema and runtime_response_format is None:
        self._validate_schema_config()
        api_params['response_format'] = self.openai_schema.response_format
      elif runtime_response_format is not None:
        if self.openai_schema:
          # Advanced callers may deliberately override response_format at
          # runtime; warn because that bypasses the configured schema.
          warnings.warn(
              'openai_schema is set but a runtime response_format kwarg '
              'was provided; the schema is bypassed for this call.',
              UserWarning,
              stacklevel=3,
          )
        api_params['response_format'] = runtime_response_format
      elif self.format_type == data.FormatType.JSON:
        api_params['response_format'] = {'type': 'json_object'}

      if (v := normalized_config.get('max_output_tokens')) is not None:
        api_params['max_tokens'] = v
      if (v := normalized_config.get('top_p')) is not None:
        api_params['top_p'] = v
      for key in [
          'frequency_penalty',
          'presence_penalty',
          'seed',
          'stop',
          'logprobs',
          'top_logprobs',
          'reasoning_effort',
      ]:
        if (v := normalized_config.get(key)) is not None:
          api_params[key] = v

      response = self._client.chat.completions.create(**api_params)

      output_text = response.choices[0].message.content

      return core_types.ScoredOutput(score=1.0, output=output_text)

    except exceptions.InferenceConfigError:
      raise
    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'OpenAI API error: {str(e)}', original=e
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

    temp = merged_kwargs.get('temperature', self.temperature)
    if temp is not None:
      config['temperature'] = temp
    if 'max_output_tokens' in merged_kwargs:
      config['max_output_tokens'] = merged_kwargs['max_output_tokens']
    if 'top_p' in merged_kwargs:
      config['top_p'] = merged_kwargs['top_p']

    for key in [
        'frequency_penalty',
        'presence_penalty',
        'seed',
        'stop',
        'logprobs',
        'top_logprobs',
        'reasoning_effort',
        'response_format',
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
      # Sequential processing for single prompt or worker
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]  # pylint: disable=duplicate-code
