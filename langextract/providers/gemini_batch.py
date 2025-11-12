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

"""Gemini Batch API helper module for LangExtract.

This module provides batch inference support using the google-genai SDK.
It handles:
- File-based batch submission for all batch sizes
- Job polling and result extraction
- Schema-based structured output
- Order preservation across batch processing
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import concurrent.futures
import dataclasses
import enum
import hashlib
import json
import os
import re
import tempfile
import time
from typing import Any, Protocol
import uuid

from absl import logging
from google import genai
from google.api_core import exceptions as google_exceptions
from google.cloud import storage

from langextract.core import exceptions

_MIME_TYPE_JSON = "application/json"
_DEFAULT_LOCATION = "us-central1"
_EXT_JSON = ".json"
_EXT_JSONL = ".jsonl"
_KEY_IDX = "idx-"
_CACHE_PREFIX = "cache"


@dataclasses.dataclass(slots=True, frozen=True)
class BatchConfig:
  """Define and validate Gemini Batch API configuration.

  Attributes:
    enabled: Whether batch mode is enabled.
    threshold: Minimum prompts to trigger batch processing.
    poll_interval: Seconds between job status checks.
    timeout: Maximum seconds to wait for job completion.
    max_prompts_per_job: Max prompts allowed in one batch job.
    ignore_item_errors: If True, continue on per-item errors.
    enable_caching: If True, use GCS-based caching for inference results.
  """

  enabled: bool = False
  threshold: int = 50
  poll_interval: int = 30
  timeout: int = 3600
  max_prompts_per_job: int = 20000
  ignore_item_errors: bool = False
  enable_caching: bool = True

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

  def __post_init__(self):
    """Validate numeric knobs early."""
    validations = [
        (self.threshold >= 1, "batch.threshold must be >= 1"),
        (self.poll_interval > 0, "batch.poll_interval must be > 0"),
        (self.timeout > 0, "batch.timeout must be > 0"),
        (self.max_prompts_per_job > 0, "batch.max_prompts_per_job must be > 0"),
    ]
    for is_valid, error_msg in validations:
      if not is_valid:
        raise ValueError(error_msg)


_TERMINAL_FAIL = frozenset({
    genai.types.JobState.JOB_STATE_FAILED,
    genai.types.JobState.JOB_STATE_CANCELLED,
    genai.types.JobState.JOB_STATE_EXPIRED,
})
_TERMINAL_OK = frozenset({
    genai.types.JobState.JOB_STATE_SUCCEEDED,
    genai.types.JobState.JOB_STATE_PAUSED,
})


def _snake_to_camel(key: str) -> str:
  """Convert snake_case to camelCase for REST API compatibility."""
  parts = key.split("_")
  return parts[0] + "".join(p.title() for p in parts[1:])


def _is_vertexai_client(client) -> bool:
  """Check if client is configured for Vertex AI with explicit identity check.

  Args:
    client: The genai.Client instance to check.

  Returns:
    True if client.vertexai is explicitly True, False otherwise.
  """
  return getattr(client, "vertexai", False) is True


def _get_project_location(client: genai.Client) -> tuple[str | None, str]:
  """Resolve project and location from client or environment."""
  project = getattr(client, "project", None)
  if not isinstance(project, str):
    project = None
  project = project or os.getenv("GOOGLE_CLOUD_PROJECT")

  location = getattr(client, "location", None)
  if not isinstance(location, str):
    location = None
  location = location or _DEFAULT_LOCATION
  return project, location


def _get_bucket_name(project: str | None, location: str) -> str:
  """Generate consistent GCS bucket name for batch operations."""
  base = f"langextract-{project}-{location}-batch".lower()
  return re.sub(r"[^a-z0-9._-]", "-", base)


def _build_request(
    prompt: str,
    schema_dict: dict | None,
    gen_config: dict | None,
    system_instruction: str | None = None,
    safety_settings: Sequence[Any] | None = None,
) -> dict:
  """Build a batch request in REST format for file-based submission.

  Constructs a properly formatted request dictionary for batch processing.
  Per the Gemini Batch API documentation, each request in the JSONL file
  can include its own generationConfig with schema and generation parameters,
  as well as top-level systemInstruction and safetySettings.

  Args:
    prompt: The text prompt to send to the model.
    schema_dict: Optional JSON schema for structured output.
    gen_config: Optional generation configuration parameters.
    system_instruction: Optional system instruction text.
    safety_settings: Optional safety settings sequence.

  Returns:
    A dictionary formatted for REST API file-based submission, containing:
      * contents: The prompt content.
      * systemInstruction: Optional system instructions.
      * safetySettings: Optional safety settings.
      * generationConfig: Optional generation configuration and schema.
  """
  request = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

  if system_instruction:
    request["systemInstruction"] = {"parts": [{"text": system_instruction}]}

  if safety_settings:
    request["safetySettings"] = safety_settings

  if schema_dict or gen_config:
    generation_config = {}
    if schema_dict:
      generation_config["responseMimeType"] = _MIME_TYPE_JSON
      generation_config["responseSchema"] = schema_dict
    if gen_config:
      for k, v in gen_config.items():
        generation_config[_snake_to_camel(k)] = v
    request["generationConfig"] = generation_config

  return request


def _submit_file(
    client: genai.Client,
    model_id: str,
    requests: Sequence[dict],
    display: str,
) -> genai.types.BatchJob:
  """Submit a file-based batch job to Vertex AI using GCS storage.

  Batch processing is only supported with Vertex AI because it requires
  GCS for file upload. Creates JSONL file, uploads to auto-created bucket,
  and submits job for async processing.

  Args:
    client: google.genai.Client instance configured for Vertex AI
        (must have client.vertexai=True).
    model_id: Model identifier (e.g., "gemini-2.5-flash").
    requests: List of request dictionaries with embedded configuration.
        Each request contains contents and optional generationConfig
        (including schema and generation parameters).
    display: Display name for the batch job, used for identification and
        as part of the GCS blob name.

  Returns:
    BatchJob object that can be polled for completion status.

  Raises:
    ValueError: If client is not configured for Vertex AI.
  """
  path = None
  try:
    with tempfile.NamedTemporaryFile(
        "w", suffix=_EXT_JSONL, delete=False, encoding="utf-8"
    ) as f:
      path = f.name
      for idx, req in enumerate(requests):
        # We use a simple "idx-{N}" key format to track the original order
        # of prompts, as batch processing may return results out of order.
        line = {"key": f"{_KEY_IDX}{idx}", "request": req}
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

    project, location = _get_project_location(client)
    bucket_name = _get_bucket_name(project, location)
    blob_name = f"batch-input/{display}-{uuid.uuid4().hex}.jsonl"

    storage_client = storage.Client(project=project)
    try:
      bucket = storage_client.create_bucket(bucket_name, location=location)
      logging.info("Created GCS bucket: %s", bucket_name)
    except google_exceptions.Conflict:
      bucket = storage_client.bucket(bucket_name)
      logging.info("Using existing GCS bucket: %s", bucket_name)

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path)

    gcs_uri = f"gs://{bucket.name}/{blob.name}"

    # Create batch job (config and schema are in per-request generationConfig)
    job = client.batches.create(
        model=model_id, src=gcs_uri, config={"display_name": display}
    )
    return job
  finally:
    if path:
      try:
        os.unlink(path)
      except OSError:
        pass


class GCSBatchCache:
  """GCS-based cache for batch inference results."""

  def __init__(self, bucket_name: str, project: str | None = None):
    self.bucket_name = bucket_name
    self.project = project
    self._client = storage.Client(project=project)
    self._bucket = self._client.bucket(bucket_name)

  def _compute_hash(self, key_data: dict) -> str:
    """Compute SHA256 hash of the canonicalized request data."""
    canonical_json = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

  def _get_single(self, key_hash: str) -> str | None:
    """Fetch single item from GCS."""
    blob = self._bucket.blob(f"{_CACHE_PREFIX}/{key_hash}{_EXT_JSON}")
    try:
      data = json.loads(blob.download_as_text())
      return data.get("text")
    except google_exceptions.NotFound:
      return None
    except Exception as e:
      logging.warning("Cache read error for %s: %s", key_hash, e)
    return None

  def get_multi(self, key_data_list: Sequence[dict]) -> dict[int, str]:
    """Fetch multiple items from GCS in parallel.

    Returns:
      Dict mapping index in key_data_list to cached text.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
      future_to_idx = {}
      for idx, key_data in enumerate(key_data_list):
        key_hash = self._compute_hash(key_data)
        future = executor.submit(self._get_single, key_hash)
        future_to_idx[future] = idx

      for future in concurrent.futures.as_completed(future_to_idx):
        idx = future_to_idx[future]
        text = future.result()
        if text is not None:
          results[idx] = text
    return results

  def set_multi(self, items: Sequence[tuple[dict, str]]) -> None:
    """Upload multiple items to GCS in parallel.

    Args:
      items: List of (key_data, result_text) tuples.
    """

    def _upload(text: str, key_data: dict):
      key_hash = self._compute_hash(key_data)
      blob = self._bucket.blob(f"{_CACHE_PREFIX}/{key_hash}{_EXT_JSON}")
      try:
        blob.upload_from_string(
            json.dumps({"text": text}, ensure_ascii=False),
            content_type=_MIME_TYPE_JSON,
        )
      except Exception as e:
        logging.warning(
            "Cache write error for %s: %s", key_hash, e, exc_info=True
        )

    def _json_default(obj):
      if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
      if isinstance(obj, enum.Enum):
        return obj.value
      raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
      for key_data, text in items:
        # If text is not a string, try to serialize it
        if not isinstance(text, str):
          try:
            text = json.dumps(text, default=_json_default, ensure_ascii=False)
          except Exception as e:
            logging.warning("Serialization error: %s", e)
            continue

        executor.submit(_upload, text, key_data)

  def iter_items(self) -> Iterator[tuple[str, str]]:
    """Iterate over all items in the cache.

    Yields:
      Tuple of (key_hash, text_content).
    """
    blobs = self._bucket.list_blobs(prefix=f"{_CACHE_PREFIX}/")
    for blob in blobs:
      if not blob.name.endswith(_EXT_JSON):
        continue
      try:
        key_hash = blob.name.split("/")[-1].replace(_EXT_JSON, "")
        data = json.loads(blob.download_as_text())
        text = data.get("text")
        if text is not None:
          yield key_hash, text
      except (json.JSONDecodeError, Exception) as e:
        logging.warning("Failed to read cache item %s: %s", blob.name, e)


class _TextResponse(Protocol):
  """Protocol for inline response objects with text attribute."""

  text: str


def _safe_get_nested(data: dict, *keys) -> Any:
  """Safely traverse nested dictionaries/lists.

  Args:
    data: The dict to traverse.
    *keys: Keys/indices to access. Use integers for list indices.

  Returns:
    The value at the path, or None if any key doesn't exist.
  """
  current = data
  for key in keys:
    if current is None:
      return None
    if isinstance(key, int):
      if not isinstance(current, list) or len(current) <= key:
        return None
      current = current[key]
    else:
      if not isinstance(current, dict):
        return None
      current = current.get(key)
  return current


def _extract_text(resp: _TextResponse | dict[str, Any] | None) -> str | None:
  """Extract text from Vertex AI batch API response.

  Args:
    resp: Response object (inline) or dict (file) containing text.

  Returns:
    Extracted text string, or None if not found or invalid.
  """
  if resp is None:
    return None

  if hasattr(resp, "text"):
    text = getattr(resp, "text", None)
    return text if isinstance(text, str) else None

  if not isinstance(resp, dict):
    return None

  # Vertex AI format: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
  text = _safe_get_nested(resp, "candidates", 0, "content", "parts", 0, "text")
  return text if isinstance(text, str) else None


def _poll_completion(
    client: genai.Client, job: genai.types.BatchJob, cfg: BatchConfig
) -> genai.types.BatchJob:
  """Poll batch job until completion or timeout.

  Args:
    client: google.genai.Client instance for polling job status.
    job: Batch job object returned from client.batches.create().
    cfg: Batch configuration including timeout and poll_interval.

  Returns:
    Completed batch job object.

  Raises:
    RuntimeError: If the job enters a failed terminal state.
    TimeoutError: If the job does not complete within cfg.timeout.
  """
  start = time.time()
  name = job.name

  while True:
    job = client.batches.get(name=name)
    state = job.state

    if state in _TERMINAL_OK:
      return job

    if state in _TERMINAL_FAIL:
      error_details = job.error or "(no error details)"
      raise exceptions.InferenceRuntimeError(
          f"Batch job failed: state={state.name}, name={name}, "
          f"error={error_details}"
      )

    if time.time() - start > cfg.timeout:
      try:
        client.batches.cancel(name=name)
      except Exception as e:
        logging.warning("Failed to cancel timed-out batch job %s: %s", name, e)
      raise exceptions.InferenceRuntimeError(
          f"Batch job timed out after {cfg.timeout}s: {name}"
      )

    time.sleep(cfg.poll_interval)


def _parse_batch_line(
    line: str, outputs: dict[int, str], cfg: BatchConfig
) -> None:
  """Parse a single line from batch output JSONL."""
  try:
    obj = json.loads(line)
  except json.JSONDecodeError:
    return

  error = obj.get("error")
  if error and not cfg.ignore_item_errors:
    code = error.get("code") if isinstance(error, dict) else None
    if code not in (None, 0):
      raise exceptions.InferenceRuntimeError(f"Batch item error: {error}")

  resp = obj.get("response", {})
  text = _extract_text(resp) or ""

  key = obj.get("key", "")
  try:
    # Extract the original index from the key (e.g., "idx-5" -> 5)
    idx = int(str(key).rsplit(_KEY_IDX, maxsplit=1)[-1])
  except (ValueError, IndexError):
    idx = max(outputs.keys(), default=-1) + 1
  outputs[idx] = text


def _extract_from_file(
    client: genai.Client,
    job: genai.types.BatchJob,
    cfg: BatchConfig,
    expected_count: int,
) -> list[str]:
  """Extract text outputs from file-based batch results, preserving order.

  Reads results from GCS output directory.

  Args:
    client: google.genai.Client instance for downloading result file.
    job: Completed batch job object with result location.
    cfg: Batch configuration including error handling settings.
    expected_count: Number of prompts submitted (for order preservation).

  Returns:
    List of text outputs corresponding 1:1 to input prompts. Missing results
    are padded with empty strings.

  Raises:
    RuntimeError: If job is missing result location or item has error.
  """
  if not _is_vertexai_client(client):
    raise ValueError("Batch API is only supported with Vertex AI.")

  outputs_by_idx: dict[int, str] = {}

  if not job.dest:
    raise exceptions.InferenceRuntimeError("Vertex AI batch job missing dest")
  gcs_uri = getattr(job.dest, "gcs_uri", None) or getattr(
      job.dest, "gcs_output_directory", None
  )
  if not gcs_uri:
    raise exceptions.InferenceRuntimeError(
        "Vertex AI batch job missing output GCS URI"
    )

  if not gcs_uri.startswith("gs://"):
    raise exceptions.InferenceRuntimeError(f"Invalid GCS URI format: {gcs_uri}")

  bucket_name, _, prefix = gcs_uri[5:].partition("/")

  project = getattr(client, "project", None) or os.getenv(
      "GOOGLE_CLOUD_PROJECT"
  )
  storage_client = storage.Client(project=project)
  bucket = storage_client.bucket(bucket_name)

  # Vertex AI may write multiple output files.
  blobs = list(bucket.list_blobs(prefix=prefix))
  if not blobs:
    raise exceptions.InferenceRuntimeError(
        f"No output files found in {gcs_uri}"
    )

  for blob in blobs:
    if not blob.name.endswith(_EXT_JSONL):
      continue

    # Stream file line by line to avoid loading entire file into memory
    with blob.open("r", encoding="utf-8") as f:
      for line in f:
        if not line.strip():
          continue
        _parse_batch_line(line, outputs_by_idx, cfg)

  return [outputs_by_idx.get(i, "") for i in range(expected_count)]


def infer_batch(
    client: genai.Client,
    model_id: str,
    prompts: Sequence[str],
    schema_dict: dict | None,
    gen_config: dict,
    cfg: BatchConfig,
    system_instruction: str | None = None,
    safety_settings: Sequence[Any] | None = None,
) -> list[str]:
  """Execute batch inference on multiple prompts using the Vertex AI Batch API.

  This function provides file-based batch processing via Vertex AI. It:
  - Uploads prompts to GCS (Google Cloud Storage)
  - Submits batch job to Vertex AI
  - Polls for job completion
  - Extracts and returns results

  Args:
    client: google.genai.Client instance configured for Vertex AI
        (must have client.vertexai=True).
    model_id: Model identifier (e.g., "gemini-2.5-flash").
    prompts: Sequence of prompts to process in batch.
    schema_dict: Optional JSON schema for structured output. When provided,
        enables JSON mode with the specified schema constraints.
    gen_config: Generation configuration parameters (temperature, top_p, etc.).
    cfg: Batch configuration including thresholds, timeouts, and error handling.
    system_instruction: Optional system instruction text.
    safety_settings: Optional safety settings sequence.

  Returns:
    List of text outputs corresponding 1:1 to input prompts. Missing results
    are padded with empty strings.

  Raises:
    RuntimeError: If batch job fails or individual items have errors
        (when cfg.ignore_item_errors is False).
    TimeoutError: If batch job doesn't complete within cfg.timeout seconds.
  """
  if not prompts:
    return []

  if not _is_vertexai_client(client):
    raise ValueError(
        "Batch API is only supported with Vertex AI. To use batch mode, create"
        " your client with: genai.Client(vertexai=True, project='YOUR_PROJECT',"
        " location='us-central1'). For Google AI API keys, batch mode is not"
        " currently supported."
    )

  logging.info("Batch API: Processing %d prompts", len(prompts))
  display_base = f"langextract-batch-{int(time.time())}"

  project, location = _get_project_location(client)
  bucket_name = _get_bucket_name(project, location)

  cache = GCSBatchCache(bucket_name, project) if cfg.enable_caching else None

  prompts_to_process: list[tuple[int, str]] = []
  cached_results: dict[int, str] = {}

  if cache:

    key_data_list = []
    for prompt in prompts:
      key_data_list.append({
          "model_id": model_id,
          "prompt": prompt,
          "system_instruction": system_instruction,
          "gen_config": gen_config,
          "safety_settings": safety_settings,
          "schema": schema_dict,
      })

    cached_results = cache.get_multi(key_data_list)

    for idx, prompt in enumerate(prompts):
      if idx not in cached_results:
        prompts_to_process.append((idx, prompt))
  else:
    prompts_to_process = list(enumerate(prompts))

  if not prompts_to_process:
    logging.info("Batch API: All %d prompts found in cache", len(prompts))
    return [cached_results[i] for i in range(len(prompts))]

  logging.info(
      "Batch API: %d cached, %d to submit",
      len(cached_results),
      len(prompts_to_process),
  )

  def _process_batch(
      batch_items: Sequence[tuple[int, str]], display: str
  ) -> dict[int, str]:
    """Submit batch job, poll completion, and extract results.

    Returns:
      Dict mapping original index to result text.
    """
    batch_prompts = [p for _, p in batch_items]
    requests = [
        _build_request(
            p, schema_dict, gen_config, system_instruction, safety_settings
        )
        for p in batch_prompts
    ]
    job = _submit_file(client, model_id, requests, display)
    job = _poll_completion(client, job, cfg)
    results = _extract_from_file(
        client, job, cfg, expected_count=len(batch_prompts)
    )

    # Map results back to original indices
    mapped_results = {}
    for (orig_idx, _), result in zip(batch_items, results):
      mapped_results[orig_idx] = result

    return mapped_results

  new_results: dict[int, str] = {}

  if (
      cfg.max_prompts_per_job
      and len(prompts_to_process) > cfg.max_prompts_per_job
  ):
    chunk_size = cfg.max_prompts_per_job
    for chunk_num, i in enumerate(
        range(0, len(prompts_to_process), chunk_size)
    ):
      chunk_items = prompts_to_process[i : i + chunk_size]
      chunk_results = _process_batch(
          chunk_items, f"{display_base}-part-{chunk_num}"
      )
      new_results.update(chunk_results)
  else:
    new_results = _process_batch(prompts_to_process, display_base)

  if cache:
    upload_list = []
    for idx, text in new_results.items():
      prompt = prompts[idx]
      key_data = {
          "model_id": model_id,
          "prompt": prompt,
          "system_instruction": system_instruction,
          "gen_config": gen_config,
          "safety_settings": safety_settings,
          "schema": schema_dict,
      }
      upload_list.append((key_data, text))

    cache.set_multi(upload_list)

  final_outputs = []
  for i in range(len(prompts)):
    if i in cached_results:
      final_outputs.append(cached_results[i])
    else:
      final_outputs.append(new_results.get(i, ""))

  return final_outputs
