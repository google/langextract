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

"""OpenAI Batch API helper module for LangExtract.

This module is intentionally written to be testable without importing the
`openai` package: it accepts a generic client object with the expected
`files.*` and `batches.*` methods.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import io
import json
import time
from typing import Any

from langextract.core import exceptions

_DEFAULT_ENDPOINT = "/v1/chat/completions"


@dataclasses.dataclass(slots=True, frozen=True)
class BatchConfig:
  """Define and validate OpenAI Batch API configuration.

  Attributes:
    enabled: Whether batch mode is enabled.
    threshold: Minimum prompts to trigger batch processing.
    completion_window: Optional OpenAI completion window string (e.g., "24h").
      If unset, LangExtract will omit it from the batch-create call.
    poll_interval: Seconds between status checks.
    timeout: Maximum seconds to wait for completion.
    max_requests_per_job: Safety cap on the number of requests per batch job.
    metadata: Optional metadata dict attached to the batch job.
    on_job_create: Optional hook invoked with the created job object.
  """

  enabled: bool = False
  threshold: int = 50
  completion_window: str | None = None
  poll_interval: int = 10
  timeout: int = 3600
  max_requests_per_job: int = 50000
  metadata: Mapping[str, Any] | None = None
  on_job_create: Callable[[Any], None] | None = None

  def __post_init__(self):
    validations = [
        (self.threshold >= 1, "batch.threshold must be >= 1"),
        (self.poll_interval > 0, "batch.poll_interval must be > 0"),
        (self.timeout > 0, "batch.timeout must be > 0"),
        (
            self.max_requests_per_job > 0,
            "batch.max_requests_per_job must be > 0",
        ),
    ]
    for is_valid, msg in validations:
      if not is_valid:
        raise ValueError(msg)

    if self.completion_window is not None and not self.completion_window:
      raise ValueError(
          "batch.completion_window must be a non-empty string when set"
      )

  @classmethod
  def from_dict(cls, d: dict | None) -> BatchConfig:
    """Create BatchConfig from dictionary, using defaults for missing keys."""
    if not d:
      return cls(enabled=False)

    # Allow either {enabled: true, ...} or a truthy dict without enabled.
    enabled = bool(d.get("enabled", True))
    return cls(
        enabled=enabled,
        threshold=int(d.get("threshold", cls.threshold)),
        completion_window=d.get("completion_window"),
        poll_interval=int(d.get("poll_interval", cls.poll_interval)),
        timeout=int(d.get("timeout", cls.timeout)),
        max_requests_per_job=int(
            d.get("max_requests_per_job", cls.max_requests_per_job)
        ),
        metadata=d.get("metadata"),
    )


def _custom_id(idx: int) -> str:
  return f"idx-{idx:06d}"


def _extract_text_from_response_body(body: Mapping[str, Any]) -> str:
  try:
    choices = body.get("choices")
    if not choices:
      raise KeyError("choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
      raise KeyError("message.content")
    return content
  except Exception as e:
    raise exceptions.InferenceRuntimeError(
        f"Failed to extract text from OpenAI batch response body: {e}",
        original=e,
        provider="OpenAI",
    ) from e


def _content_to_text(content: Any) -> str:
  """Best-effort conversion of OpenAI SDK file content responses to text."""
  if content is None:
    return ""
  if isinstance(content, str):
    return content
  if isinstance(content, bytes):
    return content.decode("utf-8")

  text = getattr(content, "text", None)
  if isinstance(text, str):
    return text

  read = getattr(content, "read", None)
  if callable(read):
    data = read()
    if isinstance(data, bytes):
      return data.decode("utf-8")
    if isinstance(data, str):
      return data

  # Fall back to stringification.
  return str(content)


def infer_batch(
    *,
    client: Any,
    model_id: str,
    prompts: Sequence[str],
    cfg: BatchConfig,
    request_builder: Callable[[str], Mapping[str, Any]],
    endpoint: str = _DEFAULT_ENDPOINT,
    batch_size: int | None = None,
) -> list[str]:
  """Execute batch inference on multiple prompts using OpenAI Batch API.

  Args:
    client: OpenAI client instance (or compatible fake for testing).
    model_id: OpenAI model id.
    prompts: Prompt strings.
    cfg: Batch configuration.
    request_builder: Callable that produces the request body for one prompt.
    endpoint: The OpenAI endpoint string for the batch (default chat completions).
    batch_size: Optional per-call limit that caps requests per batch job.

  Returns:
    List of output texts aligned with prompts.

  Raises:
    InferenceRuntimeError: On job failure, timeout, or per-item errors.
  """
  if not prompts:
    return []

  if not cfg.enabled:
    raise exceptions.InferenceConfigError(
        "OpenAI batch mode is not enabled (cfg.enabled=False)"
    )

  if batch_size is not None and batch_size <= 0:
    raise ValueError("batch_size must be > 0")

  per_job_limit = cfg.max_requests_per_job
  if batch_size is not None:
    per_job_limit = min(per_job_limit, batch_size)

  outputs: list[str] = [""] * len(prompts)

  # Submit in chunks to avoid huge jobs and to honor batch_size.
  for offset in range(0, len(prompts), per_job_limit):
    chunk = list(prompts[offset : offset + per_job_limit])
    chunk_outputs = _infer_batch_one_job(
        client=client,
        model_id=model_id,
        prompts=chunk,
        cfg=cfg,
        request_builder=request_builder,
        endpoint=endpoint,
        base_index=offset,
    )
    outputs[offset : offset + len(chunk_outputs)] = chunk_outputs

  return outputs


def _infer_batch_one_job(
    *,
    client: Any,
    model_id: str,
    prompts: Sequence[str],
    cfg: BatchConfig,
    request_builder: Callable[[str], Mapping[str, Any]],
    endpoint: str,
    base_index: int,
) -> list[str]:
  lines: list[str] = []
  for i, prompt in enumerate(prompts):
    idx = base_index + i
    body = dict(request_builder(prompt))
    body.setdefault("model", model_id)

    req = {
        "custom_id": _custom_id(idx),
        "method": "POST",
        "url": endpoint,
        "body": body,
    }
    lines.append(json.dumps(req, ensure_ascii=False))

  jsonl = "\n".join(lines) + "\n"

  # Use an in-memory buffer with a name attribute for broad compatibility.
  buf = io.BytesIO(jsonl.encode("utf-8"))
  buf.name = "langextract_openai_batch_input.jsonl"  # type: ignore[attr-defined]

  try:
    input_file = client.files.create(file=buf, purpose="batch")
    input_file_id = getattr(input_file, "id", None) or input_file.get("id")
  except Exception as e:
    raise exceptions.InferenceRuntimeError(
        f"OpenAI Batch API input file upload failed: {e}",
        original=e,
        provider="OpenAI",
    ) from e

  try:
    create_kwargs: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "metadata": dict(cfg.metadata or {}),
    }
    if cfg.completion_window:
      create_kwargs["completion_window"] = cfg.completion_window

    job = client.batches.create(**create_kwargs)
    if cfg.on_job_create:
      cfg.on_job_create(job)
    batch_id = getattr(job, "id", None) or job.get("id")
  except Exception as e:
    raise exceptions.InferenceRuntimeError(
        f"OpenAI Batch API job create failed: {e}",
        original=e,
        provider="OpenAI",
    ) from e

  start = time.time()
  last_status = None
  while True:
    if time.time() - start > cfg.timeout:
      raise exceptions.InferenceRuntimeError(
          f"OpenAI Batch API job timed out after {cfg.timeout}s"
          f" (last_status={last_status})",
          provider="OpenAI",
      )

    try:
      job = client.batches.retrieve(batch_id)
    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f"OpenAI Batch API job retrieve failed: {e}",
          original=e,
          provider="OpenAI",
      ) from e

    status = getattr(job, "status", None) or job.get("status")
    last_status = status

    if status in ("completed", "failed", "expired", "cancelled"):
      break

    time.sleep(cfg.poll_interval)

  if status != "completed":
    err = getattr(job, "error", None) or job.get("error")
    raise exceptions.InferenceRuntimeError(
        f"OpenAI Batch API job did not complete (status={status}, error={err})",
        provider="OpenAI",
    )

  output_file_id = (
      getattr(job, "output_file_id", None)
      or job.get("output_file_id")
      or getattr(job, "output_file", None)
      or job.get("output_file")
  )
  if not output_file_id:
    raise exceptions.InferenceRuntimeError(
        "OpenAI Batch API job completed but has no output_file_id",
        provider="OpenAI",
    )

  try:
    content = client.files.content(output_file_id)
    text = _content_to_text(content)
  except Exception as e:
    raise exceptions.InferenceRuntimeError(
        f"OpenAI Batch API output download failed: {e}",
        original=e,
        provider="OpenAI",
    ) from e

  # Parse output JSONL.
  outputs_by_idx: dict[int, str] = {}
  errors: list[str] = []
  for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line:
      continue
    try:
      obj = json.loads(line)
    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f"OpenAI Batch API output JSONL parse error: {e}",
          original=e,
          provider="OpenAI",
      ) from e

    cid = obj.get("custom_id")
    if not cid or not isinstance(cid, str) or not cid.startswith("idx-"):
      continue

    try:
      idx = int(cid.split("-", 1)[1])
    except ValueError:
      continue

    item_error = obj.get("error")
    if item_error:
      errors.append(f"{cid}: {item_error}")
      continue

    response = obj.get("response") or {}
    body = response.get("body") or {}
    try:
      outputs_by_idx[idx] = _extract_text_from_response_body(body)
    except exceptions.InferenceRuntimeError as e:
      errors.append(f"{cid}: {e}")

  if errors:
    raise exceptions.InferenceRuntimeError(
        "OpenAI Batch API per-item errors: " + "; ".join(errors),
        provider="OpenAI",
    )

  # Ensure we have every prompt.
  chunk_outputs: list[str] = []
  for i in range(base_index, base_index + len(prompts)):
    if i not in outputs_by_idx:
      raise exceptions.InferenceRuntimeError(
          f"OpenAI Batch API missing output for custom_id={_custom_id(i)}",
          provider="OpenAI",
      )
    chunk_outputs.append(outputs_by_idx[i])

  return chunk_outputs
