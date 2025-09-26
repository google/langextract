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

"""Retry utilities for handling transient errors in LangExtract."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from absl import logging

from langextract.core import exceptions

T = TypeVar("T")

# Transient error patterns that should trigger retries
TRANSIENT_ERROR_PATTERNS = [
    "503",
    "service unavailable",
    "temporarily unavailable",
    "rate limit",
    "429",
    "too many requests",
    "connection reset",
    "timeout",
    "timed out",
    "deadline exceeded",
    "model is overloaded",
    "quota exceeded",
    "resource exhausted",
    "internal server error",
    "502",
    "504",
    "gateway timeout",
    "bad gateway",
]

# Exception types that indicate transient errors
TRANSIENT_EXCEPTION_TYPES = [
    "ServiceUnavailable",
    "RateLimitError",
    "Timeout",
    "ConnectionError",
    "TimeoutError",
    "OSError",
    "RuntimeError",
]


def is_transient_error(error: Exception) -> bool:
  """Check if an error is transient and should be retried.

  Args:
      error: The exception to check

  Returns:
      True if the error is transient and should be retried
  """
  error_str = str(error).lower()
  error_type = type(error).__name__

  # Check for transient error patterns in the error message
  is_transient_pattern = any(
      pattern in error_str for pattern in TRANSIENT_ERROR_PATTERNS
  )

  # Check for transient exception types
  is_transient_type = error_type in TRANSIENT_EXCEPTION_TYPES

  return is_transient_pattern or is_transient_type


def retry_on_transient_errors(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
  """Decorator to retry functions on transient errors with exponential backoff.

  Args:
      max_retries: Maximum number of retry attempts (default: 3)
      initial_delay: Initial delay in seconds (default: 1.0)
      backoff_factor: Multiplier for exponential backoff (default: 2.0)
      max_delay: Maximum delay between retries in seconds (default: 60.0)
      jitter: Whether to add random jitter to prevent thundering herd (default: True)

  Returns:
      Decorated function with retry logic
  """

  def decorator(func: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
      last_exception = None
      delay = initial_delay

      for attempt in range(max_retries + 1):
        try:
          return func(*args, **kwargs)
        except Exception as e:
          last_exception = e

          # If this is a transient error
          if not is_transient_error(e):
            logging.debug(
                "Non-transient error encountered, not retrying: %s", str(e)
            )
            raise

          # If we've exhausted retries
          if attempt >= max_retries:
            logging.warning(
                "Max retries (%d) exceeded for transient error: %s",
                max_retries,
                str(e),
            )
            raise

          # Calculate delay with exponential backoff.
          current_delay = min(delay, max_delay)

          # Add jitter to prevent thundering herd.
          if jitter:
            import random

            jitter_amount = current_delay * 0.1 * random.random()
            current_delay += jitter_amount

          logging.info(
              "Transient error on attempt %d/%d: %s. Retrying in %.2f"
              " seconds...",
              attempt + 1,
              max_retries + 1,
              str(e),
              current_delay,
          )

          time.sleep(current_delay)
          delay = min(delay * backoff_factor, max_delay)

      # This should never be reached, but just in case.
      if last_exception:
        raise last_exception
      raise RuntimeError("Retry logic failed unexpectedly")

    return wrapper

  return decorator


def retry_chunk_processing(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    enabled: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
  """Specialized retry decorator for chunk processing with chunk-specific logging.

  This is optimized for the annotation process where individual chunks may fail
  due to transient errors while other chunks in the same batch succeed.

  Args:
      max_retries: Maximum number of retry attempts (default: 3)
      initial_delay: Initial delay in seconds (default: 1.0)
      backoff_factor: Multiplier for exponential backoff (default: 2.0)
      max_delay: Maximum delay between retries in seconds (default: 60.0)
      enabled: Whether retry is enabled (default: True)

  Returns:
      Decorated function with chunk-specific retry logic
  """

  def decorator(func: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
      # Check if retry is disabled.
      if not enabled:
        return func(*args, **kwargs)

      last_exception = None
      delay = initial_delay

      for attempt in range(max_retries + 1):
        try:
          return func(*args, **kwargs)
        except Exception as e:
          last_exception = e

          # Check if this is a transient error
          if not is_transient_error(e):
            logging.debug(
                "Non-transient error in chunk processing, not retrying: %s",
                str(e),
            )
            raise

          # If we've exhausted retries, raise the exception
          if attempt >= max_retries:
            logging.error(
                "Chunk processing failed after %d retries: %s",
                max_retries,
                str(e),
            )
            raise

          # Calculate delay with exponential backoff
          current_delay = min(delay, max_delay)

          # Add jitter to prevent thundering herd
          import random

          jitter_amount = current_delay * 0.1 * random.random()
          current_delay += jitter_amount

          logging.warning(
              "Chunk processing failed on attempt %d/%d due to transient error:"
              " %s. Retrying in %.2f seconds...",
              attempt + 1,
              max_retries + 1,
              str(e),
              current_delay,
          )

          time.sleep(current_delay)
          delay = min(delay * backoff_factor, max_delay)

      # This should never be reached, but just in case
      if last_exception:
        raise last_exception
      raise RuntimeError("Chunk retry logic failed unexpectedly")

    return wrapper

  return decorator
