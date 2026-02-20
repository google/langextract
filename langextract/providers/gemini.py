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
import time
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

# Default retry configuration for transient errors (503, 429, etc.)
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 1.0  # Initial delay in seconds
_DEFAULT_MAX_RETRY_DELAY = 16.0  # Maximum delay in seconds

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
  max_retries: int = _DEFAULT_MAX_RETRIES
  retry_delay: float = _DEFAULT_RETRY_DELAY
  max_retry_delay: float = _DEFAULT_MAX_RETRY_DELAY
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
      max_retries: int = _DEFAULT_MAX_RETRIES,
      retry_delay: float = _DEFAULT_RETRY_DELAY,
      max_retry_delay: float = _DEFAULT_MAX_RETRY_DELAY,
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
      max_retries: Maximum number of retry attempts for transient errors
        (503, 429, network errors). Set to 0 to disable retries.
      retry_delay: Initial delay in seconds before first retry.
        Subsequent delays increase exponentially.
      max_retry_delay: Maximum delay in seconds between retries.
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
    self.max_retries = max_retries
    self.retry_delay = retry_delay
    self.max_retry_delay = max_retry_delay

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

  def _is_retryable_error(self, error: Exception) -> bool:
    """Determine if an error is retryable (transient).

    Retryable errors include:
    - 503 Service Unavailable (model overloaded)
    - 429 Too Many Requests (rate limiting)
    - Network-related errors (connection, timeout, etc.)

    Non-retryable errors include:
    - 400 Bad Request (invalid input)
    - 401 Unauthorized (authentication failure)
    - 403 Forbidden (permission denied)
    - 404 Not Found

    Args:
      error: The exception to check.

    Returns:
      True if the error is retryable, False otherwise.
    """
    error_str = str(error).lower()

    # Check for 503 (service unavailable / model overloaded)
    if '503' in error_str or 'overloaded' in error_str:
      return True

    # Check for 429 (rate limiting)
    if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
      return True

    # Check for 500 (internal server error) - may be transient
    if '500' in error_str and 'internal' in error_str:
      return True

    # Network-related errors are typically transient
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
      return True

    # Check for connection/timeout keywords in error message
    if any(
        keyword in error_str
        for keyword in ['timeout', 'connection', 'reset', 'unavailable']
    ):
      return True

    return False

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> core_types.ScoredOutput:
    """Process a single prompt and return a ScoredOutput.

    Implements exponential backoff retry for transient errors (503, 429, etc.).
    Each chunk is retried independently, allowing other chunks to succeed
    even if one chunk encounters temporary failures.

    Args:
      prompt: The prompt to process.
      config: Configuration dictionary for the API call.

    Returns:
      A ScoredOutput containing the model's response.

    Raises:
      InferenceRuntimeError: If the API call fails after all retry attempts
        or encounters a non-retryable error.
    """
    last_exception: Exception | None = None
    delay = self.retry_delay

    for attempt in range(self.max_retries + 1):
      try:
        # Apply stored kwargs that weren't already set in config
        # Make a copy to avoid mutating on retries
        call_config = dict(config)
        for key, value in self._extra_kwargs.items():
          if key not in call_config and value is not None:
            call_config[key] = value

        if self.gemini_schema:
          self._validate_schema_config()
          call_config.setdefault('response_mime_type', 'application/json')
          call_config.setdefault(
              'response_schema', self.gemini_schema.schema_dict
          )

        response = self._client.models.generate_content(
            model=self.model_id, contents=prompt, config=call_config
        )

        return core_types.ScoredOutput(score=1.0, output=response.text)

      except Exception as e:
        last_exception = e

        # Check if we should retry
        if attempt < self.max_retries and self._is_retryable_error(e):
          logging.warning(
              'Retryable error on attempt %d/%d: %s. Retrying in %.1f'
              ' seconds...',
              attempt + 1,
              self.max_retries + 1,
              str(e),
              delay,
          )
          time.sleep(delay)
          # Exponential backoff with cap
          delay = min(delay * 2, self.max_retry_delay)
          continue

        # Non-retryable error or max retries exhausted
        raise exceptions.InferenceRuntimeError(
            f'Gemini API error: {str(e)}', original=e
        ) from e

    # This should not be reached, but handle it just in case
    raise exceptions.InferenceRuntimeError(
        f'Gemini API error after {self.max_retries + 1} attempts:'
        f' {str(last_exception)}',
        original=last_exception,
    )

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
              project=self.project,
              location=self.location,
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
