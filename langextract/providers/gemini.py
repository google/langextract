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

from absl import logging

from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import gemini_batch
from langextract.providers import patterns
from langextract.providers import router
from langextract.providers import schemas

_DEFAULT_MODEL_ID = 'gemini-2.5-flash'
_DEFAULT_LOCATION = 'us-central1'
_MIME_TYPE_JSON = 'application/json'

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
class GeminiLanguageModel(base_model.BaseLanguageModel):  # pylint: disable=too-many-instance-attributes
  """Language model inference using Google's Gemini API with structured output."""

  model_id: str = _DEFAULT_MODEL_ID
  api_key: str | None = None
  vertexai: bool = False
  credentials: Any | None = None
  project: str | None = None
  location: str | None = None
  http_options: Any | None = None
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
    if isinstance(schema_instance, schemas.gemini.GeminiSchema):
      self.gemini_schema = schema_instance

  def __init__(
      self,
      model_id: str = _DEFAULT_MODEL_ID,
      api_key: str | None = None,
      vertexai: bool = False,
      credentials: Any | None = None,
      project: str | None = None,
      location: str | None = None,
      http_options: Any | None = None,
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
      vertexai: Whether to use Vertex AI instead of API key authentication.
      credentials: Optional Google auth credentials for Vertex AI.
      project: Google Cloud project ID for Vertex AI.
      location: Vertex AI location (e.g., 'global', 'us-central1').
      http_options: Optional HTTP options for the client (e.g., for VPC endpoints).
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
          'google-genai is required for Gemini. Install it with: pip install'
          ' google-genai'
      ) from e

    self.model_id = model_id
    self.api_key = api_key
    self.vertexai = vertexai
    self.credentials = credentials
    self.project = project
    self.location = location
    self.http_options = http_options
    self.gemini_schema = gemini_schema
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self.fence_output = fence_output

    # Extract batch config before we filter kwargs into _extra_kwargs
    batch_cfg_dict = kwargs.pop('batch', None)
    self._batch_cfg = gemini_batch.BatchConfig.from_dict(batch_cfg_dict)

    if not self.api_key and not self.vertexai:
      raise exceptions.InferenceConfigError(
          'Gemini models require either:\n  - An API key via api_key parameter'
          ' or LANGEXTRACT_API_KEY env var\n  - Vertex AI configuration with'
          ' vertexai=True, project, and location'
      )
    if self.vertexai and (not self.project or not self.location):
      raise exceptions.InferenceConfigError(
          'Vertex AI mode requires both project and location parameters'
      )

    if self.api_key and self.vertexai:
      logging.warning(
          'Both API key and Vertex AI configuration provided. '
          'API key will take precedence for authentication.'
      )

    self._client = genai.Client(
        api_key=self.api_key,
        vertexai=vertexai,
        credentials=credentials,
        project=project,
        location=location,
        http_options=http_options,
    )

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )
    self._extra_kwargs = {
        k: v for k, v in (kwargs or {}).items() if k in _API_CONFIG_KEYS
    }

  def _validate_schema_config(self) -> None:
    """Validate that schema configuration is compatible with format type.

    Raises:
      InferenceConfigError: If gemini_schema is set but format_type is not JSON.
    """
    if self.gemini_schema and self.format_type != data.FormatType.JSON:
      raise exceptions.InferenceConfigError(
          'Gemini structured output only supports JSON format. '
          'Set format_type=JSON or use_schema_constraints=False.'
      )

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
        self._validate_schema_config()
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
    for key in ('max_output_tokens', 'top_p', 'top_k'):
      if key in merged_kwargs:
        config[key] = merged_kwargs[key]

    handled_keys = {'temperature', 'max_output_tokens', 'top_p', 'top_k'}
    for key, value in merged_kwargs.items():
      if (
          key not in handled_keys
          and key in _API_CONFIG_KEYS
          and value is not None
      ):
        config[key] = value

    # Use batch API if threshold met
    if self._batch_cfg and self._batch_cfg.enabled:
      if len(batch_prompts) >= self._batch_cfg.threshold:
        try:
          if self.gemini_schema:
            self._validate_schema_config()
          schema_dict = (
              self.gemini_schema.schema_dict if self.gemini_schema else None
          )
          # Remove schema fields from config for batch API - they're handled via schema_dict
          batch_config = dict(config)
          batch_config.pop('response_mime_type', None)
          batch_config.pop('response_schema', None)
          # Extract top-level fields that don't belong in generationConfig
          system_instruction = batch_config.pop('system_instruction', None)
          safety_settings = batch_config.pop('safety_settings', None)
          outputs = gemini_batch.infer_batch(
              client=self._client,
              model_id=self.model_id,
              prompts=batch_prompts,
              schema_dict=schema_dict,
              gen_config=batch_config,
              cfg=self._batch_cfg,
              system_instruction=system_instruction,
              safety_settings=safety_settings,
          )
        except exceptions.InferenceRuntimeError:
          raise
        except Exception as e:
          raise exceptions.InferenceRuntimeError(
              f'Gemini Batch API error: {e}', original=e
          ) from e

        for text in outputs:
          yield [core_types.ScoredOutput(score=1.0, output=text)]
        return
      else:
        logging.info(
            'Gemini batch mode enabled but prompt count (%d) is below the'
            ' threshold (%d); using real-time API. Submit at least %d prompts'
            ' to trigger batch mode.',
            len(batch_prompts),
            self._batch_cfg.threshold,
            self._batch_cfg.threshold,
        )

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
