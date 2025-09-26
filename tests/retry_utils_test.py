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


import time
from unittest import mock

from absl.testing import absltest

from langextract import retry_utils
from langextract.core import exceptions


class RetryUtilsTest(absltest.TestCase):
  """Test retry utility functions."""

  def test_is_transient_error_503(self):
    """Test that 503 errors are identified as transient."""
    error = exceptions.InferenceRuntimeError("503 The model is overloaded")
    self.assertTrue(retry_utils.is_transient_error(error))

    error = exceptions.InferenceRuntimeError("Service temporarily unavailable")
    self.assertTrue(retry_utils.is_transient_error(error))

  def test_is_transient_error_429(self):
    """Test that 429 rate limit errors are identified as transient."""
    error = exceptions.InferenceRuntimeError("429 Too Many Requests")
    self.assertTrue(retry_utils.is_transient_error(error))

    error = exceptions.InferenceRuntimeError("Rate limit exceeded")
    self.assertTrue(retry_utils.is_transient_error(error))

  def test_is_transient_error_timeout(self):
    """Test that timeout errors are identified as transient."""
    error = TimeoutError("Request timed out")
    self.assertTrue(retry_utils.is_transient_error(error))

    error = exceptions.InferenceRuntimeError("Connection timeout")
    self.assertTrue(retry_utils.is_transient_error(error))

  def test_is_transient_error_non_transient(self):
    """Test that non-transient errors are not retried."""
    error = exceptions.InferenceConfigError("Invalid API key")
    self.assertFalse(retry_utils.is_transient_error(error))

    error = exceptions.InferenceRuntimeError("Invalid model ID")
    self.assertFalse(retry_utils.is_transient_error(error))

  def test_retry_decorator_success(self):
    """Test that retry decorator works for successful calls."""

    @retry_utils.retry_on_transient_errors(max_retries=2, initial_delay=0.01)
    def successful_function():
      return "success"

    result = successful_function()
    self.assertEqual(result, "success")

  def test_retry_decorator_transient_error_success(self):
    """Test that retry decorator retries on transient errors and succeeds."""
    call_count = 0

    @retry_utils.retry_on_transient_errors(max_retries=3, initial_delay=0.01)
    def failing_then_successful_function():
      nonlocal call_count
      call_count += 1
      if call_count < 3:
        raise exceptions.InferenceRuntimeError("503 The model is overloaded")
      return "success"

    result = failing_then_successful_function()
    self.assertEqual(result, "success")
    self.assertEqual(call_count, 3)

  def test_retry_decorator_non_transient_error(self):
    """Test that retry decorator doesn't retry non-transient errors."""
    call_count = 0

    @retry_utils.retry_on_transient_errors(max_retries=3, initial_delay=0.01)
    def non_transient_failing_function():
      nonlocal call_count
      call_count += 1
      raise exceptions.InferenceConfigError("Invalid API key")

    with self.assertRaises(exceptions.InferenceConfigError):
      non_transient_failing_function()

    self.assertEqual(call_count, 1)

  def test_retry_decorator_max_retries_exceeded(self):
    """Test that retry decorator gives up after max retries."""
    call_count = 0

    @retry_utils.retry_on_transient_errors(max_retries=2, initial_delay=0.01)
    def always_failing_function():
      nonlocal call_count
      call_count += 1
      raise exceptions.InferenceRuntimeError("503 The model is overloaded")

    with self.assertRaises(exceptions.InferenceRuntimeError):
      always_failing_function()

    self.assertEqual(call_count, 3)

  def test_retry_chunk_processing_disabled(self):
    """Test that retry can be disabled."""
    call_count = 0

    @retry_utils.retry_chunk_processing(enabled=False)
    def failing_function():
      nonlocal call_count
      call_count += 1
      raise exceptions.InferenceRuntimeError("503 The model is overloaded")

    with self.assertRaises(exceptions.InferenceRuntimeError):
      failing_function()

    self.assertEqual(call_count, 1)

  def test_retry_chunk_processing_enabled(self):
    """Test that retry works when enabled."""
    call_count = 0

    @retry_utils.retry_chunk_processing(max_retries=2, initial_delay=0.01)
    def failing_then_successful_function():
      nonlocal call_count
      call_count += 1
      if call_count < 3:
        raise exceptions.InferenceRuntimeError("503 The model is overloaded")
      return "success"

    result = failing_then_successful_function()
    self.assertEqual(result, "success")
    self.assertEqual(call_count, 3)

  def test_retry_backoff_timing(self):
    """Test that retry uses exponential backoff."""
    call_times = []

    @retry_utils.retry_on_transient_errors(
        max_retries=2, initial_delay=0.1, backoff_factor=2.0, max_delay=1.0
    )
    def timing_test_function():
      call_times.append(time.time())
      if len(call_times) < 3:
        raise exceptions.InferenceRuntimeError("503 The model is overloaded")
      return "success"

    start_time = time.time()
    result = timing_test_function()
    end_time = time.time()

    self.assertEqual(result, "success")
    self.assertEqual(len(call_times), 3)

    if len(call_times) >= 3:
      delay1 = call_times[1] - call_times[0]
      delay2 = call_times[2] - call_times[1]

      self.assertGreater(delay1, 0.05)
      self.assertGreater(delay2, 0.15)
      self.assertGreater(delay2, delay1)

  def test_retry_with_jitter(self):
    """Test that retry adds jitter to prevent thundering herd."""
    call_times = []

    @retry_utils.retry_on_transient_errors(
        max_retries=2, initial_delay=0.1, jitter=True
    )
    def jitter_test_function():
      call_times.append(time.time())
      if len(call_times) < 3:
        raise exceptions.InferenceRuntimeError("503 The model is overloaded")
      return "success"

    result = jitter_test_function()
    self.assertEqual(result, "success")
    self.assertEqual(len(call_times), 3)

    if len(call_times) >= 2:
      delay1 = call_times[1] - call_times[0]
      self.assertGreater(delay1, 0.05)
      self.assertLess(delay1, 0.15)

  def test_retry_max_delay_cap(self):
    """Test that retry respects max_delay cap."""
    call_times = []

    @retry_utils.retry_on_transient_errors(
        max_retries=2, initial_delay=0.1, backoff_factor=10.0, max_delay=0.2
    )
    def max_delay_test_function():
      call_times.append(time.time())
      if len(call_times) < 3:
        raise exceptions.InferenceRuntimeError("503 The model is overloaded")
      return "success"

    result = max_delay_test_function()
    self.assertEqual(result, "success")
    self.assertEqual(len(call_times), 3)

    if len(call_times) >= 2:
      delay1 = call_times[1] - call_times[0]
      delay2 = call_times[2] - call_times[1]

      self.assertLess(delay1, 0.3)
      self.assertLess(delay2, 0.3)

  def test_error_message_detection(self):
    """Test that various error messages are properly detected."""
    error_messages_503 = [
        "503 The model is overloaded",
        "503 Service Unavailable",
        "The model is overloaded",
        "Service temporarily unavailable",
    ]

    for msg in error_messages_503:
      error = exceptions.InferenceRuntimeError(msg)
      self.assertTrue(
          retry_utils.is_transient_error(error),
          f"Error message '{msg}' should be transient",
      )

    error_messages_429 = [
        "429 Too Many Requests",
        "Rate limit exceeded",
        "Too many requests",
    ]

    for msg in error_messages_429:
      error = exceptions.InferenceRuntimeError(msg)
      self.assertTrue(
          retry_utils.is_transient_error(error),
          f"Error message '{msg}' should be transient",
      )

    error_messages_timeout = [
        "Request timed out",
        "Connection timeout",
        "Timeout",
        "Deadline exceeded",
    ]

    for msg in error_messages_timeout:
      error = exceptions.InferenceRuntimeError(msg)
      self.assertTrue(
          retry_utils.is_transient_error(error),
          f"Error message '{msg}' should be transient",
      )

    error_messages_non_transient = [
        "Invalid API key",
        "Invalid model ID",
        "Authentication failed",
    ]

    for msg in error_messages_non_transient:
      error = exceptions.InferenceRuntimeError(msg)
      self.assertFalse(
          retry_utils.is_transient_error(error),
          f"Error message '{msg}' should not be transient",
      )

  def test_retry_decorator_preserves_function_metadata(self):
    """Test that retry decorator preserves function metadata."""

    @retry_utils.retry_on_transient_errors(max_retries=2)
    def test_function():
      """Test function docstring."""
      return "test"

    self.assertEqual(test_function.__name__, "test_function")
    self.assertEqual(test_function.__doc__, "Test function docstring.")

  def test_retry_chunk_processing_preserves_function_metadata(self):
    """Test that chunk retry decorator preserves function metadata."""

    @retry_utils.retry_chunk_processing(max_retries=2)
    def test_chunk_function():
      """Test chunk function docstring."""
      return "test"

    self.assertEqual(test_chunk_function.__name__, "test_chunk_function")
    self.assertEqual(
        test_chunk_function.__doc__, "Test chunk function docstring."
    )


if __name__ == "__main__":
  absltest.main()
