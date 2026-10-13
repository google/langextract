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
import math
import numbers
import time
from typing import Any

from absl import logging

from langextract import factory
from langextract.core import base_model
from langextract.core import exceptions
from langextract.core import schema as schema_lib
from langextract.core import types
from langextract.providers import retryable_http

__all__ = ['FallbackLanguageModel', 'RetryPolicy']

_BACKOFF_MULTIPLIER = 2
_TRANSIENT_STATUS_CODES = retryable_http.RETRYABLE_HTTP_STATUS_CODES
_TRANSIENT_ERROR_TYPE_NAMES = frozenset({
    'ReadTimeout',
    'ConnectTimeout',
    'Timeout',
    'TimeoutException',
    'APITimeoutError',
    'APIConnectionError',
    'ConnectionError',  # requests/urllib3; builtin ConnectionError is isinstance
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
    if not _is_non_bool_integral(self.max_attempts):
      raise exceptions.InferenceConfigError(
          f'max_attempts must be an integer >= 1, got {self.max_attempts!r}.'
      )
    if self.max_attempts < 1:
      raise exceptions.InferenceConfigError(
          f'max_attempts must be at least 1, got {self.max_attempts}.'
      )
    _reject_invalid_delay('initial_delay_seconds', self.initial_delay_seconds)
    _reject_invalid_delay('max_delay_seconds', self.max_delay_seconds)
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
    uncapped_delay = self.initial_delay_seconds * (
        _BACKOFF_MULTIPLIER**retry_index
    )
    return min(uncapped_delay, self.max_delay_seconds)


def _is_non_bool_integral(value: Any) -> bool:
  """Returns True when `value` is an integer-like value, excluding bool."""
  return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _reject_invalid_delay(field_name: str, delay_seconds: float) -> None:
  """Raises InferenceConfigError when a delay is not a finite non-negative number."""
  if isinstance(delay_seconds, bool) or not isinstance(
      delay_seconds, (int, float)
  ):
    raise exceptions.InferenceConfigError(
        f'{field_name} must be a non-negative number, got {delay_seconds!r}.'
    )
  if math.isnan(delay_seconds) or math.isinf(delay_seconds):
    raise exceptions.InferenceConfigError(
        f'{field_name} must be a finite non-negative number, got'
        f' {delay_seconds!r}.'
    )
  if delay_seconds < 0:
    raise exceptions.InferenceConfigError(
        f'{field_name} must be non-negative, got {delay_seconds}.'
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


# (property name, getter) pairs that must match across all models.
_OUTPUT_FORMAT_PROPERTIES = (
    ('requires_fence_output', lambda model: model.requires_fence_output),
    (
        'format_type',
        lambda model: getattr(model, 'format_type', types.FormatType.JSON),
    ),
    ('schema', lambda model: type(model.schema)),
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


def _validate_models(models: Sequence[base_model.BaseLanguageModel]) -> None:
  """Rejects an empty, nested or mixed-output-format model list.

  Raises:
    InferenceConfigError: If `models` is empty, contains a
      `FallbackLanguageModel`, or the models differ in output format.
  """
  if not models:
    raise exceptions.InferenceConfigError(
        'FallbackLanguageModel needs at least one model.'
    )
  for model in models:
    if isinstance(model, FallbackLanguageModel):
      raise exceptions.InferenceConfigError(
          f'Fallback model {_model_label(model)!r} is a'
          ' FallbackLanguageModel, which cannot be nested.'
      )
  _check_same_output_format(models)


def _check_same_output_format(
    models: Sequence[base_model.BaseLanguageModel],
) -> None:
  """Raises if any model's output format differs from the first model's."""
  first = models[0]
  for model in models[1:]:
    for name, getter in _OUTPUT_FORMAT_PROPERTIES:
      if getter(model) != getter(first):
        raise exceptions.InferenceConfigError(
            f'Fallback model {_model_label(model)!r} has'
            f' {name}={getter(model)!r} but first model'
            f' {_model_label(first)!r} has {getter(first)!r}; all models must'
            ' share one output format.'
        )


class FallbackLanguageModel(base_model.BaseLanguageModel):
  """Serves every batch from the first model that does not fail transiently.

  For each call to `infer` (one batch), the wrapper starts at the first model
  and retries a transient error on it up to `retry_policy.max_attempts` total
  attempts with capped exponential backoff, then moves to the next model. The
  next batch starts at the first model again, so put the cheaper model first.

  A transient error is one marked `retryable=True`, a timeout or connection
  error, or one carrying a transient HTTP status (see
  `retryable_http.RETRYABLE_HTTP_STATUS_CODES`) on the error or its
  `original`. Other errors, such as configuration or authentication
  failures, are raised unchanged without retry or fallback. When every model
  fails transiently, the last error is raised unchanged. Outputs of a model
  that fails part-way through a batch are discarded. Each retry or fallback
  re-sends the whole batch of prompts.

  `served_model_ids` is the ordered list of serving-model labels for every
  successful `infer` call on this wrapper instance since construction (it
  accumulates across documents, passes and extract calls; it is not reset
  per run). All models must share one output format and cannot be fallback
  models themselves. Inner providers keep their own internal retries (for
  example Gemini `max_retries`), so you may want to set those to 0.

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
    _validate_models(self._models)
    self._retry_policy = retry_policy
    self._sleep = sleep
    self._served_model_ids: list[str] = []
    self.model_id = _model_label(self._models[0])

  @property
  def served_model_ids(self) -> tuple[str, ...]:
    """Serving-model labels for each successful `infer`, in call order.

    Grows for the lifetime of this wrapper; not cleared between extract runs.
    """
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
        batch_outputs = self._infer_with_retries(model, batch_prompts, kwargs)
      except Exception as error:
        # why: only transient errors fall through to the next model.
        if not _is_transient(error):
          raise
        last_error = error
        continue
      label = _model_label(model)
      self._served_model_ids.append(label)
      logging.info(
          'Batch of %d prompts served by %s', len(batch_prompts), label
      )
      yield from batch_outputs
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
    attempt_index = 0
    while True:
      try:
        return [
            list(prompt_outputs)
            for prompt_outputs in model.infer(batch_prompts, **kwargs)
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
        attempt_index += 1
        if attempt_index >= max_attempts:
          raise
        self._sleep(self._retry_policy.delay_before_retry(attempt_index - 1))

  @property
  def schema(self) -> schema_lib.BaseSchema | None:
    """The first model's schema."""
    return self._models[0].schema

  @property
  def requires_fence_output(self) -> bool:
    """The first model's fence requirement."""
    return self._models[0].requires_fence_output

  def set_fence_output(self, fence_output: bool | None) -> None:
    """Applies the fence preference to every model.

    Args:
      fence_output: True to force fences, False to disable, None for auto.
    """
    for model in self._models:
      model.set_fence_output(fence_output)

  def apply_output_schema(self, output_schema: types.JsonSchema) -> None:
    """Applies the output schema to every model.

    Args:
      output_schema: JSON schema for the raw output envelope.

    Raises:
      InferenceConfigError: If a model rejects the schema or applying it
        leaves the models with different output formats.
    """
    for model in self._models:
      model.apply_output_schema(output_schema)
    _check_same_output_format(self._models)
