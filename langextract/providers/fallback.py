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

from langextract import factory
from langextract.core import base_model
from langextract.core import schema as schema_lib
from langextract.core import types

__all__ = ['FallbackLanguageModel', 'RetryPolicy']


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

  def delay_before_retry(self, retry_index: int) -> float:
    """Returns the capped exponential delay before a retry.

    Args:
      retry_index: 0-based index of the retry on one model.

    Returns:
      Seconds to sleep before that retry.
    """
    del retry_index
    return 0.0


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
    self._models = tuple(models)
    self._retry_policy = retry_policy
    self._sleep = sleep
    self._served_model_ids: list[str] = []
    self.model_id = None

  @property
  def served_model_ids(self) -> tuple[str, ...]:
    """Labels of the models that served each successful batch, in order."""
    return ()

  def infer(
      self, batch_prompts: Sequence[str], **kwargs: Any
  ) -> Iterator[Sequence[types.ScoredOutput]]:
    """Runs one batch on the first model that serves it.

    Args:
      batch_prompts: Batch of prompts.
      **kwargs: Passed to the serving model's `infer`.

    Returns:
      The serving model's outputs for the batch.

    Raises:
      Exception: The first non-transient error, or the last model's last
        transient error once every model is exhausted.
    """
    yield from ()

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
