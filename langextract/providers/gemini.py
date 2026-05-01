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
import numbers
import random
import re
import time
from typing import Any, Final, Iterator, Sequence

from langextract._logging import get_logger
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import gemini_batch

logger = get_logger(__name__)
from langextract.providers import patterns
from langextract.providers import router
from langextract.providers import schemas

_DEFAULT_MODEL_ID = 'gemini-2.5-flash'
_DEFAULT_LOCATION = 'us-central1'
_MIME_TYPE_JSON = 'application/json'

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 1.0
_DEFAULT_MAX_RETRY_DELAY = 16.0

_RETRYABLE_API_CODES = frozenset({408, 429, 500, 502, 503, 504})

# Phrases are narrow on purpose: bare "quota" or "unavailable" can be permanent
# (denied quota, region out of service); we only match the transient forms.
_RETRYABLE_MESSAGE_RE = re.compile(
    r'503|overloaded|429|rate[ _]limit|quota exceeded|500.*internal'
    r'|temporarily unavailable|timeout|connection reset',
    re.IGNORECASE,
)


def _is_non_bool_integral(value: Any) -> bool:
  """Return True when `value` is an integer-like value, excluding bool."""
  return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_non_bool_real(value: Any) -> bool:
  """Return True when `value` is a real number, excluding bool."""
  return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _has_sdk_retry_options(http_options: Any) -> bool:
  """Return True if http_options enables SDK-level retries.

  Only reports True when SDK retries would *actually* execute: the google-genai
  SDK normalizes `HttpRetryOptions.attempts` of 0 or 1 to `stop_after_attempt(1)`,
  i.e. no retries, so those values do not stack with our provider loop.

  Accepts both HttpOptions (attribute access) and HttpOptionsDict (dict); the
  dict form validates through pydantic camelCase aliases, so both
  `retry_options` and `retryOptions` reach the same field.
  """
  if http_options is None:
    return False
  if isinstance(http_options, dict):
    retry_options = http_options.get('retry_options')
    if retry_options is None:
      retry_options = http_options.get('retryOptions')
  else:
    retry_options = getattr(http_options, 'retry_options', None)
  if retry_options is None:
    return False
  if isinstance(retry_options, dict):
    attempts = retry_options.get('attempts')
  else:
    attempts = getattr(retry_options, 'attempts', None)
  # attempts=None means SDK default (which is >1); 0 or 1 means no retries.
  return attempts is None or attempts > 1


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
      *,
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
    for name, value, ok in (
        (
            'max_retries',
            max_retries,
            _is_non_bool_integral(max_retries) and max_retries >= 0,
        ),
        (
            'retry_delay',
            retry_delay,
            _is_non_bool_real(retry_delay) and retry_delay >= 0,
        ),
        (
            'max_retry_delay',
            max_retry_delay,
            _is_non_bool_real(max_retry_delay) and max_retry_delay > 0,
        ),
    ):
      if not ok:
        raise exceptions.InferenceConfigError(f'{name} invalid: {value}')
    self.max_retries = max_retries
    self.retry_delay = retry_delay
    self.max_retry_delay = max_retry_delay

    # Avoid stacking with SDK-level retries (HttpOptions.retry_options).
    if max_retries > 0 and _has_sdk_retry_options(http_options):
      raise exceptions.InferenceConfigError(
          'http_options.retry_options and max_retries>0 both configured; '
          'retries would stack. Set max_retries=0 or clear retry_options.'
      )

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
      logger.warning(
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
    """Return True if `error` is a transient failure worth retrying."""
    try:
      from google.genai import errors as genai_errors  # pylint: disable=import-outside-toplevel

      if isinstance(error, genai_errors.APIError):
        return error.code in _RETRYABLE_API_CODES
    except ImportError:
      pass

    # httpx transient subclasses only. LocalProtocolError / UnsupportedProtocol
    # are client/config bugs and not included.
    try:
      import httpx  # pylint: disable=import-outside-toplevel

      if isinstance(
          error,
          (
              httpx.TimeoutException,
              httpx.NetworkError,
              httpx.RemoteProtocolError,
              httpx.ProxyError,
          ),
      ):
        return True
    except ImportError:
      pass

    # Specifically ConnectionError / TimeoutError. Bare OSError is excluded:
    # it also covers file/permission errors that won't resolve by retrying.
    if isinstance(error, (ConnectionError, TimeoutError)):
      return True

    return bool(_RETRYABLE_MESSAGE_RE.search(str(error)))

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> core_types.ScoredOutput:
    """Run one Gemini request with per-chunk retries for transient failures."""
    delay = self.retry_delay
    for attempt in range(self.max_retries + 1):
      try:
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
        if attempt < self.max_retries and self._is_retryable_error(e):
          # Cap after jitter so the named maximum applies to the real sleep.
          sleep_for = min(
              delay * random.uniform(0.5, 1.5), self.max_retry_delay
          )
          logging.info(
              'Retryable error on attempt %d/%d: %s. Retrying in %.1fs...',
              attempt + 1,
              self.max_retries + 1,
              e,
              sleep_for,
          )
          time.sleep(sleep_for)
          delay = min(delay * 2, self.max_retry_delay)
          continue
        raise exceptions.InferenceRuntimeError(
            f'Gemini API error: {e}', original=e
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
        logger.info(
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
