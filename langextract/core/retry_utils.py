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

"""Retry helpers for transient inference failures."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import random
import time
from typing import Any, TypeVar

from langextract.core import exceptions

T = TypeVar("T")


@dataclasses.dataclass(slots=True, frozen=True)
class RetryConfig:
  """Retry configuration for transient provider failures."""

  enabled: bool = True
  max_attempts: int = 5
  initial_delay_s: float = 1.0
  max_delay_s: float = 30.0
  backoff_multiplier: float = 2.0
  jitter_ratio: float = 0.1

  retryable_status_codes: frozenset[int] = frozenset({429, 500, 502, 503, 504})
  retryable_message_substrings: tuple[str, ...] = (
      "model is overloaded",
      "overloaded",
      "service unavailable",
      "temporarily unavailable",
      "rate limit",
      "too many requests",
      "timeout",
      "timed out",
      "connection reset",
      "connection aborted",
      "connection error",
  )

  def __post_init__(self) -> None:
    if self.max_attempts < 1:
      raise ValueError("retry.max_attempts must be >= 1")
    if self.initial_delay_s < 0:
      raise ValueError("retry.initial_delay_s must be >= 0")
    if self.max_delay_s < 0:
      raise ValueError("retry.max_delay_s must be >= 0")
    if self.backoff_multiplier < 1:
      raise ValueError("retry.backoff_multiplier must be >= 1")
    if not (0.0 <= self.jitter_ratio <= 1.0):
      raise ValueError("retry.jitter_ratio must be in [0, 1]")

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> RetryConfig:
    if d is None:
      return cls()
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in d.items() if k in valid_keys}
    return cls(**filtered)

  def delay_s(self, attempt: int) -> float:
    """Compute delay for a 1-based attempt number."""
    if attempt <= 1 or self.initial_delay_s <= 0:
      return 0.0
    base = self.initial_delay_s * (self.backoff_multiplier ** (attempt - 2))
    base = min(base, self.max_delay_s) if self.max_delay_s > 0 else base
    if self.jitter_ratio <= 0:
      return base
    jitter = random.uniform(1.0 - self.jitter_ratio, 1.0 + self.jitter_ratio)
    return max(0.0, base * jitter)


def call_with_retry(
    fn: Callable[[], T],
    *,
    cfg: RetryConfig,
    is_retryable: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> T:
  """Call `fn`, retrying on transient failures."""
  if sleep_fn is None:
    sleep_fn = time.sleep
  last_exc: BaseException | None = None
  for attempt in range(1, cfg.max_attempts + 1):
    try:
      return fn()
    except BaseException as e:  # pylint: disable=broad-exception-caught
      last_exc = e
      if not cfg.enabled or attempt >= cfg.max_attempts:
        raise
      retryable = (
          is_retryable(e)
          if is_retryable is not None
          else is_retryable_error(e, cfg=cfg)
      )
      if not retryable:
        raise
      delay = cfg.delay_s(attempt + 1)
      if on_retry is not None:
        on_retry(attempt, e, delay)
      if delay > 0:
        sleep_fn(delay)
  assert last_exc is not None
  raise last_exc


def is_retryable_error(exc: BaseException, *, cfg: RetryConfig) -> bool:
  """Heuristic retryability check for common provider exceptions."""
  unwrapped = _unwrap_inference_error(exc)
  code = _status_code(unwrapped)
  if code is not None and code in cfg.retryable_status_codes:
    return True
  msg = str(unwrapped).lower()
  return any(substr in msg for substr in cfg.retryable_message_substrings)


def _unwrap_inference_error(exc: BaseException) -> BaseException:
  if isinstance(exc, exceptions.InferenceRuntimeError) and exc.original is not None:
    return exc.original
  return exc


def _status_code(exc: BaseException) -> int | None:
  """Best-effort extraction of an HTTP-like status code from an exception."""
  # Common shapes:
  # - requests.HTTPError: .response.status_code
  # - openai errors: .status_code
  # - google api_core exceptions: .code() -> int
  for attr in ("status_code", "status", "code"):
    val = getattr(exc, attr, None)
    if val is None:
      continue
    try:
      candidate = val() if callable(val) else val
    except TypeError:
      continue
    if isinstance(candidate, int):
      return candidate

  response = getattr(exc, "response", None)
  if response is not None:
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
      return status
  return None

