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

from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import Any, Iterator, Sequence

from langextract import data
from langextract import exceptions
from langextract import inference
from langextract import schema
from langextract.providers import registry


@registry.register(
    r'^gpt-4',  # gpt-4.1, gpt-4o, gpt-4-turbo, etc.
    r'^gpt4\.',  # gpt4.1-mini, gpt4.1-nano, etc.
    priority=10,
)
@dataclasses.dataclass(init=False)
class OpenAILanguageModel(inference.BaseLanguageModel):
  """Language model inference using OpenAI's API with structured output."""

  model_id: str = 'gpt-4o-mini'
  api_key: str | None = None
  base_url: str | None = None
  organization: str | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  max_workers: int = 10
  fence_output: bool = False
  _client: Any = dataclasses.field(default=None, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = 'gpt-4o-mini',
      api_key: str | None = None,
      base_url: str | None = None,
      organization: str | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      fence_output: bool = False,
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
      fence_output: Whether to wrap output in markdown fences.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    # Lazy import: OpenAI package required
    try:
      # pylint: disable=import-outside-toplevel
      import openai
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'OpenAI provider requires openai package. '
          'Install with: pip install langextract[openai]'
      ) from e

    self.model_id = model_id
    self.api_key = api_key
    self.base_url = base_url
    self.organization = organization
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self.fence_output = fence_output
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      raise exceptions.InferenceConfigError('API key not provided for OpenAI.')

    # Initialize the OpenAI client
    self._client = openai.OpenAI(
        api_key=self.api_key,
        base_url=self.base_url,
        organization=self.organization,
    )

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> inference.ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      # Build kwargs for the API call
      kwargs = {}

      # Use response_format for JSON when not using fences
      if self.format_type == data.FormatType.JSON and not self.fence_output:
        kwargs['response_format'] = {'type': 'json_object'}
      response = self._client.chat.completions.create(
          model=self.model_id,
          messages=[
              {'role': 'user', 'content': prompt},
          ],
          temperature=config.get('temperature', self.temperature),
          max_tokens=config.get('max_output_tokens'),
          top_p=config.get('top_p'),
          n=1,
          **kwargs,
      )

      # Extract the response text using the v1.x response format
      output_text = response.choices[0].message.content

      return inference.ScoredOutput(score=1.0, output=output_text)

    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'OpenAI API error: {str(e)}', original=e
      ) from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[inference.ScoredOutput]]:
    """Runs inference on a list of prompts via OpenAI's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    config = {
        'temperature': kwargs.get('temperature', self.temperature),
    }
    if 'max_output_tokens' in kwargs:
      config['max_output_tokens'] = kwargs['max_output_tokens']
    if 'top_p' in kwargs:
      config['top_p'] = kwargs['top_p']

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

        results: list[inference.ScoredOutput | None] = [None] * len(
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
        yield [result]
