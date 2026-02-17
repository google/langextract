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

"""OpenAI Batch API support.

Implements file-based batch submission and result extraction for the
`/v1/chat/completions` endpoint.

The Batch API returns output lines in arbitrary order. We preserve the input
ordering by assigning a stable `custom_id` per request.

Reference: https://platform.openai.com/docs/guides/batch
"""

from __future__ import annotations

import dataclasses
import io
import json
import time
from typing import Any, Sequence

from absl import logging

from langextract.core import data
from langextract.core import exceptions


_ENDPOINT_CHAT_COMPLETIONS = "/v1/chat/completions"
_COMPLETION_WINDOW = "24h"
_CUSTOM_ID_PREFIX = "idx-"

_TERMINAL_OK = frozenset({"completed"})
_TERMINAL_FAIL = frozenset({"failed"})
_TERMINAL_PARTIAL = frozenset({"expired", "cancelled"})
_TERMINAL_ALL = _TERMINAL_OK | _TERMINAL_FAIL | _TERMINAL_PARTIAL


@dataclasses.dataclass(slots=True, frozen=True)
class BatchConfig:
  """Define and validate OpenAI Batch API configuration.

  Attributes:
    enabled: Whether batch mode is enabled.
    threshold: Minimum prompts to trigger batch processing.
    poll_interval: Seconds between status checks.
    timeout: Maximum seconds to wait for completion.
    max_prompts_per_job: Max prompts allowed in one batch job (OpenAI limit is
      50,000 requests per batch file).
    ignore_item_errors: If True, fill item failures with a fallback output.
    metadata: Optional OpenAI metadata dict attached to the batch job.
  """

  enabled: bool = False
  threshold: int = 50
  poll_interval: int = 30
  timeout: int = 3600
  max_prompts_per_job: int = 50000
  ignore_item_errors: bool = False
  metadata: dict[str, str] | None = None

  def __post_init__(self):
    validations = [
        (self.threshold >= 1, "batch.threshold must be >= 1"),
        (self.poll_interval > 0, "batch.poll_interval must be > 0"),
        (self.timeout > 0, "batch.timeout must be > 0"),
        (self.max_prompts_per_job > 0, "batch.max_prompts_per_job must be > 0"),
    ]
    for is_valid, msg in validations:
      if not is_valid:
        raise ValueError(msg)
    if self.max_prompts_per_job > 50000:
      raise ValueError("batch.max_prompts_per_job must be <= 50000")

  @classmethod
  def from_dict(cls, d: dict | None) -> BatchConfig:
    """Create BatchConfig from dictionary, using defaults for missing keys."""
    if d is None:
      return cls()
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered_dict = {k: v for k, v in d.items() if k in valid_keys}

    unknown = sorted(set(d.keys()) - valid_keys)
    if unknown:
      logging.warning(
          "Ignoring unknown batch config keys: %s", ", ".join(unknown)
      )
    return cls(**filtered_dict)


def empty_extractions_output(
    *, format_type: data.FormatType, use_fences: bool
) -> str:
  """Return a minimal valid empty extraction payload for the format."""
  if format_type == data.FormatType.JSON:
    content = '{"extractions": []}'
    if not use_fences:
      return content
    return f"```json\n{content}\n```"

  content = "extractions: []"
  if not use_fences:
    return content
  return f"```yaml\n{content}\n```"


def infer_batch(
    *,
    client: Any,
    request_bodies: Sequence[dict[str, Any]],
    cfg: BatchConfig,
    endpoint: str = _ENDPOINT_CHAT_COMPLETIONS,
    fallback_output: str | None = None,
) -> list[str]:
  """Execute OpenAI batch inference for many chat.completions requests.

  Args:
    client: `openai.OpenAI` client instance.
    request_bodies: Per-request `body` payloads for `/v1/chat/completions`.
    cfg: BatchConfig settings.
    endpoint: Batch endpoint (defaults to chat completions).
    fallback_output: Output to use for per-item errors when
      cfg.ignore_item_errors=True.

  Returns:
    A list of output strings in the same order as request_bodies.
  """
  if cfg.ignore_item_errors and fallback_output is None:
    raise ValueError(
        "fallback_output must be provided when ignore_item_errors=True"
    )

  results: list[str] = []
  for start in range(0, len(request_bodies), cfg.max_prompts_per_job):
    chunk = request_bodies[start : start + cfg.max_prompts_per_job]
    results.extend(
        _infer_batch_single_job(
            client=client,
            request_bodies=chunk,
            cfg=cfg,
            endpoint=endpoint,
            fallback_output=fallback_output,
        )
    )
  return results


def _infer_batch_single_job(
    *,
    client: Any,
    request_bodies: Sequence[dict[str, Any]],
    cfg: BatchConfig,
    endpoint: str,
    fallback_output: str | None,
) -> list[str]:
  """Submit one batch job and return outputs in request order."""
  jsonl = io.BytesIO()
  # The OpenAI SDK uses the file object's name for multipart uploads.
  jsonl.name = "langextract_openai_batch.jsonl"  # type: ignore[attr-defined]

  for idx, body in enumerate(request_bodies):
    line = {
        "custom_id": f"{_CUSTOM_ID_PREFIX}{idx}",
        "method": "POST",
        "url": endpoint,
        "body": body,
    }
    jsonl.write(
        (json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
        .encode("utf-8")
    )

  jsonl.seek(0)

  try:
    file_obj = client.files.create(file=jsonl, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window=_COMPLETION_WINDOW,
        metadata=cfg.metadata,
    )
  except Exception as e:
    raise exceptions.InferenceRuntimeError(
        f"OpenAI Batch API submission error: {e}", original=e
    ) from e

  batch_id = batch.id

  completed = _poll_batch(client=client, batch_id=batch_id, cfg=cfg)

  output_file_id = getattr(completed, "output_file_id", None)
  error_file_id = getattr(completed, "error_file_id", None)

  outputs_by_idx: dict[int, str] = {}
  errors_by_idx: dict[int, str] = {}

  if output_file_id:
    _parse_batch_file(
        client=client,
        file_id=output_file_id,
        outputs_by_idx=outputs_by_idx,
        errors_by_idx=errors_by_idx,
    )

  if error_file_id:
    _parse_batch_file(
        client=client,
        file_id=error_file_id,
        outputs_by_idx=outputs_by_idx,
        errors_by_idx=errors_by_idx,
    )

  # Rebuild ordered outputs (and surface errors if configured).
  ordered: list[str] = []
  for idx in range(len(request_bodies)):
    if idx in outputs_by_idx:
      ordered.append(outputs_by_idx[idx])
      continue
    err = errors_by_idx.get(idx)
    if err is None:
      err = f"Missing batch output for custom_id={_CUSTOM_ID_PREFIX}{idx}"
    if cfg.ignore_item_errors:
      ordered.append(fallback_output or "")
    else:
      raise exceptions.InferenceRuntimeError(
          f"OpenAI Batch item failed: custom_id={_CUSTOM_ID_PREFIX}{idx}: {err}"
      )
  return ordered


def _poll_batch(*, client: Any, batch_id: str, cfg: BatchConfig) -> Any:
  """Poll batch until terminal state or timeout."""
  deadline = time.time() + cfg.timeout
  last_status = None
  while True:
    try:
      batch = client.batches.retrieve(batch_id)
    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f"OpenAI Batch API retrieve error: {e}", original=e
      ) from e

    status = getattr(batch, "status", None)
    if status != last_status:
      logging.info("OpenAI batch %s status=%s", batch_id, status)
      last_status = status

    if status in _TERMINAL_ALL:
      if status in _TERMINAL_FAIL:
        raise exceptions.InferenceRuntimeError(
            f"OpenAI batch failed: batch_id={batch_id}"
        )
      return batch

    if time.time() > deadline:
      try:
        client.batches.cancel(batch_id)
      except Exception as e:
        logging.warning("Failed to cancel timed-out OpenAI batch %s: %s", batch_id, e)
      raise exceptions.InferenceRuntimeError(
          f"OpenAI batch timed out after {cfg.timeout}s: batch_id={batch_id}"
      )

    time.sleep(cfg.poll_interval)


def _download_file_text(*, client: Any, file_id: str) -> str:
  """Download a file as UTF-8 text from OpenAI."""
  resp = client.files.content(file_id)
  if hasattr(resp, "read"):
    raw = resp.read()
    if isinstance(raw, bytes):
      return raw.decode("utf-8", errors="replace")
    return str(raw)
  # Fallback for alternate response shapes
  return str(resp)


def _parse_custom_id(custom_id: str) -> int:
  if not isinstance(custom_id, str) or not custom_id.startswith(_CUSTOM_ID_PREFIX):
    raise ValueError(f"Invalid custom_id: {custom_id!r}")
  return int(custom_id[len(_CUSTOM_ID_PREFIX) :])


def _parse_batch_file(
    *,
    client: Any,
    file_id: str,
    outputs_by_idx: dict[int, str],
    errors_by_idx: dict[int, str],
) -> None:
  """Parse a batch output/error file and update output/error mappings."""
  content = _download_file_text(client=client, file_id=file_id)
  for raw in content.splitlines():
    line = raw.strip()
    if not line:
      continue
    try:
      item = json.loads(line)
      idx = _parse_custom_id(item.get("custom_id", ""))
      if idx in outputs_by_idx or idx in errors_by_idx:
        raise ValueError(f"Duplicate custom_id: {item.get('custom_id')}")

      err = item.get("error")
      if err:
        errors_by_idx[idx] = err.get("message", str(err))
        continue

      resp = item.get("response")
      if not resp:
        errors_by_idx[idx] = "Missing response"
        continue

      status_code = resp.get("status_code")
      if status_code != 200:
        errors_by_idx[idx] = f"HTTP {status_code}"
        continue

      body = resp.get("body") or {}
      # Chat Completions shape: body.choices[0].message.content
      choices = body.get("choices") or []
      if not choices:
        errors_by_idx[idx] = "Missing choices in response body"
        continue
      message = (choices[0] or {}).get("message") or {}
      content_text = message.get("content")
      if not isinstance(content_text, str):
        errors_by_idx[idx] = "Missing message.content in response body"
        continue

      outputs_by_idx[idx] = content_text

    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f"Failed to parse OpenAI batch file line: {e}", original=e
      ) from e

