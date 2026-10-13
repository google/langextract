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

"""Fallback provider: serve each batch from the first model that works."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import dataclasses
import time
from typing import Any

from absl import logging

from langextract import factory
from langextract.core import base_model
from langextract.core import exceptions
from langextract.core import schema as schema_lib
from langextract.core import types

__all__ = ['FallbackLanguageModel', 'RetryPolicy']

_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_TRANSIENT_ERROR_TYPE_NAMES = frozenset({
    'ReadTimeout',
    'ConnectTimeout',
    'Timeout',
    'TimeoutException',
    'APITimeoutError',
    'APIConnectionError',
    'NetworkError',
})


@dataclasses.dataclass(frozen=True)
class RetryPolicy:
  """How often and how slowly to retry a transient error on one model.

  Attributes:
    max_attempts: Total attempts per model, including the first.
    initial_delay_seconds: Delay before the first retry.
    max_delay_seconds: Upper bound for the delay before any retry.
  """

  max_attempts: int = 3
  initial_delay_seconds: float = 1.0
  max_delay_seconds: float = 30.0

  def __post_init__(self) -> None:
    if self.max_attempts < 1:
      raise exceptions.InferenceConfigError(
          f'max_attempts must be at least 1, got {self.max_attempts}.'
      )
    if self.initial_delay_seconds < 0:
      raise exceptions.InferenceConfigError(
          'initial_delay_seconds must be non-negative, got'
          f' {self.initial_delay_seconds}.'
      )
    if self.max_delay_seconds < self.initial_delay_seconds:
      raise exceptions.InferenceConfigError(
          f'max_delay_seconds ({self.max_delay_seconds}) must be at least'
          f' initial_delay_seconds ({self.initial_delay_seconds}).'
      )

  def delay_before_retry(self, retry_index: int) -> float:
    """Returns the capped exponential delay before a retry.

    Args:
      retry_index: 0-based index of the retry on one model.

    Returns:
      Seconds to sleep before that retry.
    """
    return min(
        self.initial_delay_seconds * 2**retry_index, self.max_delay_seconds
    )


def _is_transient(error: Exception) -> bool:
  """Returns whether `error` is worth retrying; never reads the message."""
  explicit = getattr(error, 'retryable', None)
  if isinstance(explicit, bool):
    return explicit
  candidates = (error, getattr(error, 'original', None))
  return any(
      _looks_transient(candidate)
      for candidate in candidates
      if candidate is not None
  )


def _looks_transient(error: BaseException) -> bool:
  """Returns whether one error is a timeout, connection or 5xx/429 failure."""
  if isinstance(error, (TimeoutError, ConnectionError)):
    return True
  for attribute in ('status_code', 'code'):
    status = getattr(error, attribute, None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS_CODES:
      return True
  # why: SDK timeout classes (requests, httpx, openai) are matched by class
  # name so optional SDKs need not be imported.
  return any(
      klass.__name__ in _TRANSIENT_ERROR_TYPE_NAMES
      for klass in type(error).__mro__
  )


def _model_label(model: base_model.BaseLanguageModel) -> str:
  """Returns the model's `model_id`, else its class name."""
  return getattr(model, 'model_id', None) or type(model).__name__


def _resolve_model(
    model_or_config: factory.ModelConfig | base_model.BaseLanguageModel,
) -> base_model.BaseLanguageModel:
  """Builds a model from a config; returns an already-built model as is."""
  if isinstance(model_or_config, factory.ModelConfig):
    return factory.create_model(model_or_config)
  return model_or_config


class FallbackLanguageModel(base_model.BaseLanguageModel):
  """Serves every batch from the first model that does not fail transiently.

  For each call to `infer` (one batch), the wrapper starts at the first model
  and retries a transient error on it up to `retry_policy.max_attempts` total
  attempts with capped exponential backoff, then moves to the next model. The
  next batch starts at the first model again, so put the cheaper model first.

  A transient error is one marked `retryable=True`, a timeout or connection
  error, or one carrying an HTTP status of 408, 429, 500, 502, 503 or 504 (on
  the error or its `original`). Other errors, such as configuration or
  authentication failures, are raised unchanged without retry or fallback.
  When every model fails transiently, the last error is raised unchanged.
  Outputs of a model that fails part-way through a batch are discarded.

  `served_model_ids` records which model served each successful batch. All
  models must share one output format and cannot be fallback models
  themselves. Inner providers keep their own internal retries (for example
  Gemini `max_retries`), so you may want to set those to 0.

  Select it through the router with `provider='fallback'` and leave
  `model_id` unset::

    factory.ModelConfig(
        provider='fallback',
        provider_kwargs={
            'models': [config_a, config_b],
            'retry_policy': RetryPolicy(max_attempts=3),
            'sleep': time.sleep,
        },
    )
  """

  def __init__(
      self,
      models: Sequence[factory.ModelConfig | base_model.BaseLanguageModel],
      *,
      retry_policy: RetryPolicy = RetryPolicy(),
      sleep: Callable[[float], None] = time.sleep,
  ) -> None:
    """Initializes the fallback chain.

    Args:
      models: Ordered model configs or built models; the first is tried first.
      retry_policy: Retry count and backoff applied to each model.
      sleep: Called with the backoff delay; inject a recorder in tests.
    """
    super().__init__()
    self._models = tuple(_resolve_model(entry) for entry in models)
    self._retry_policy = retry_policy
    self._sleep = sleep
    self._served_model_ids: list[str] = []
    self.model_id = _model_label(self._models[0])

  @property
  def served_model_ids(self) -> tuple[str, ...]:
    """Labels of the models that served each successful batch, in order."""
    return tuple(self._served_model_ids)

  def infer(
      self, batch_prompts: Sequence[str], **kwargs: Any
  ) -> Iterator[Sequence[types.ScoredOutput]]:
    """Runs one batch on the first model that serves it.

    Args:
      batch_prompts: Batch of prompts.
      **kwargs: Passed to the serving model's `infer`.

    Yields:
      The serving model's outputs for the batch.

    Raises:
      Exception: The first non-transient error, or the last model's last
        transient error once every model is exhausted.
    """
    last_error = None
    for model in self._models:
      try:
        outputs = self._infer_with_retries(model, batch_prompts, kwargs)
      except Exception as error:
        # why: only transient errors fall through to the next model.
        if not _is_transient(error):
          raise
        last_error = error
        continue
      self._served_model_ids.append(_model_label(model))
      logging.info(
          'Batch of %d prompts served by %s',
          len(batch_prompts),
          _model_label(model),
      )
      yield from outputs
      return
    raise last_error

  def _infer_with_retries(
      self,
      model: base_model.BaseLanguageModel,
      batch_prompts: Sequence[str],
      kwargs: dict[str, Any],
  ) -> list[list[types.ScoredOutput]]:
    """Runs the whole batch on one model, retrying transient errors.

    Each attempt materializes the full batch, so a partial failure yields
    nothing. The backoff restarts for every model.

    Returns:
      The model's outputs for every prompt.

    Raises:
      Exception: A non-transient error at once, or the last transient error
        after `max_attempts` attempts.
    """
    max_attempts = self._retry_policy.max_attempts
    for attempt_index in range(max_attempts):
      try:
        return [
            list(outputs) for outputs in model.infer(batch_prompts, **kwargs)
        ]
      except Exception as error:
        if not _is_transient(error):
          raise
        logging.warning(
            'Transient %s from %s on attempt %d/%d',
            type(error).__name__,
            _model_label(model),
            attempt_index + 1,
            max_attempts,
        )
        if attempt_index + 1 == max_attempts:
          raise
        self._sleep(self._retry_policy.delay_before_retry(attempt_index))
    raise AssertionError('unreachable: max_attempts is validated >= 1')

  @property
  def schema(self) -> schema_lib.BaseSchema | None:
    """The first model's schema."""
    return None

  @property
  def requires_fence_output(self) -> bool:
    """The first model's fence requirement."""
    return True

  def set_fence_output(self, fence_output: bool | None) -> None:
    """Applies the fence preference to every model."""
    del fence_output

  def apply_output_schema(self, output_schema: types.JsonSchema) -> None:
    """Applies the output schema to every model."""
    del output_schema
