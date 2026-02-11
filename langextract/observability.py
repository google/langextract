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

"""Optional observability integrations for LangExtract.

Observability is fully optional and fail-open:
- extraction must continue if telemetry is disabled or misconfigured
- observer errors are logged and never raised into extraction flow
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import inspect
import os
from typing import Any

from absl import logging

try:
  from langfuse import Langfuse as _Langfuse
except ImportError:
  _Langfuse = None

try:
  from langfuse import propagate_attributes as _propagate_attributes
except ImportError:
  _propagate_attributes = None

__all__ = [
    "create_observer",
    "NoOpObserver",
    "LangfuseObserver",
]


def _as_mapping(value: Any) -> dict[str, Any] | None:
  """Convert common payload types to a mutable mapping."""
  if value is None:
    return None
  if isinstance(value, Mapping):
    return dict(value)
  if dataclasses.is_dataclass(value):
    return dataclasses.asdict(value)
  return None


def _filter_none(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Drop None values from payload dictionaries."""
  return {k: v for k, v in payload.items() if v is not None}


def _filter_kwargs_for_callable(
    func: Callable[..., Any], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
  """Filter kwargs to what a callable accepts when signature is explicit."""
  try:
    signature = inspect.signature(func)
  except (TypeError, ValueError):
    return dict(kwargs)

  parameters = signature.parameters.values()
  if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
    return dict(kwargs)

  allowed = {p.name for p in parameters}
  return {k: v for k, v in kwargs.items() if k in allowed}


def _normalize_usage_payload(
    usage: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """Convert LangExtract usage payload to Langfuse v3 usage_details schema."""
  usage_payload = _as_mapping(usage)
  if usage_payload is None:
    return None, None

  usage_details: dict[str, Any] = {}
  existing_usage_details = _as_mapping(usage_payload.get("usage_details"))
  if existing_usage_details:
    usage_details.update(_filter_none(existing_usage_details))

  prompt_tokens = usage_payload.get("prompt_tokens")
  if prompt_tokens is None:
    prompt_tokens = usage_payload.get("input_tokens")
  if prompt_tokens is None:
    prompt_tokens = usage_payload.get("input")

  completion_tokens = usage_payload.get("completion_tokens")
  if completion_tokens is None:
    completion_tokens = usage_payload.get("output_tokens")
  if completion_tokens is None:
    completion_tokens = usage_payload.get("output")

  total_tokens = usage_payload.get("total_tokens")
  if total_tokens is None:
    total_tokens = usage_payload.get("total")

  if prompt_tokens is not None:
    usage_details["prompt_tokens"] = prompt_tokens
  if completion_tokens is not None:
    usage_details["completion_tokens"] = completion_tokens
  if total_tokens is not None:
    usage_details["total_tokens"] = total_tokens

  for detail_key in ("prompt_tokens_details", "completion_tokens_details"):
    detail_payload = _as_mapping(usage_payload.get(detail_key))
    if detail_payload:
      usage_details[detail_key] = _filter_none(detail_payload)

  provider_details = _as_mapping(usage_payload.get("provider_details"))
  return usage_details or None, provider_details


def _normalize_observation_payload(kwargs: Mapping[str, Any]) -> dict[str, Any]:
  """Normalize observation payload and map usage -> usage_details."""
  payload = _filter_none(kwargs)
  metadata = _as_mapping(payload.get("metadata")) or {}

  usage_details: dict[str, Any] = {}
  explicit_usage_details = _as_mapping(payload.get("usage_details"))
  if explicit_usage_details:
    usage_details.update(_filter_none(explicit_usage_details))

  mapped_usage_details, provider_usage_details = _normalize_usage_payload(
      payload.get("usage")
  )
  if mapped_usage_details:
    usage_details.update(mapped_usage_details)

  if usage_details:
    payload["usage_details"] = usage_details
  payload.pop("usage", None)

  if provider_usage_details:
    metadata["provider_usage"] = provider_usage_details

  if metadata:
    payload["metadata"] = metadata

  return _filter_none(payload)


def _safe_exit_context(context_manager: Any | None) -> None:
  """Exit context manager while swallowing observability errors."""
  if context_manager is None:
    return
  try:
    context_manager.__exit__(None, None, None)
  except Exception as e:  # pragma: no cover - observer failures are non-fatal
    logging.warning("Observer context exit failed: %s", e)


class _NoOpHandle:
  """Observation handle used when telemetry is disabled."""

  def update(self, **_: Any) -> None:
    return None

  def end(self, **_: Any) -> None:
    return None


class _ObservationHandle:
  """Wraps a Langfuse observation so callers use update/end consistently."""

  def __init__(
      self,
      *,
      observation: Any | None,
      observation_context: Any | None,
      propagate_context: Any | None = None,
  ) -> None:
    self._observation = observation
    self._observation_context = observation_context
    self._propagate_context = propagate_context
    self._ended = False

  def update(self, **kwargs: Any) -> None:
    if self._ended:
      return
    if self._observation is None or not hasattr(self._observation, "update"):
      return
    payload = _normalize_observation_payload(kwargs)
    if not payload:
      return
    try:
      update_fn = self._observation.update
      update_kwargs = _filter_kwargs_for_callable(update_fn, payload)
      update_fn(**update_kwargs)
    except Exception as e:  # pragma: no cover - observer failures are non-fatal
      logging.warning("Observer update failed: %s", e)

  def end(self, **kwargs: Any) -> None:
    if self._ended:
      return

    payload = _normalize_observation_payload(kwargs)

    try:
      # For context-managed observations, update first then exit to close.
      if self._observation_context is not None:
        if payload:
          self.update(**payload)
      elif self._observation is not None and hasattr(self._observation, "end"):
        end_fn = self._observation.end
        end_kwargs = _filter_kwargs_for_callable(end_fn, payload)
        end_fn(**end_kwargs)
      elif payload:
        self.update(**payload)
    except Exception as e:  # pragma: no cover - observer failures are non-fatal
      logging.warning("Observer end failed: %s", e)
    finally:
      _safe_exit_context(self._propagate_context)
      _safe_exit_context(self._observation_context)
      self._ended = True


class NoOpObserver:
  """No-op observer implementation."""

  enabled = False
  provider = "none"

  def generation(self, **_: Any) -> _NoOpHandle:
    return _NoOpHandle()

  def span(self, **_: Any) -> _NoOpHandle:
    return _NoOpHandle()

  def flush(self) -> None:
    return None


class LangfuseObserver(NoOpObserver):
  """Langfuse observer implementation (optional, fail-open)."""

  provider = "langfuse"

  def __init__(
      self,
      public_key: str | None = None,
      secret_key: str | None = None,
      base_url: str | None = None,
      host: str | None = None,
      enabled: bool = True,
      **kwargs: Any,
  ) -> None:
    self.enabled = False
    self._client = None

    if not enabled:
      return

    public_key = (public_key or os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (secret_key or os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    resolved_base_url = (
        base_url
        or host
        or os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
        or None
    )
    if isinstance(resolved_base_url, str):
      resolved_base_url = resolved_base_url.strip() or None

    if _Langfuse is None:
      logging.warning(
          "Langfuse observer requested but 'langfuse' is not installed. "
          'Install optional deps with: pip install "langextract[langfuse]"'
      )
      return

    if not public_key or not secret_key:
      logging.info(
          "Langfuse keys not set; observability remains disabled (optional)."
      )
      return

    client_kwargs = dict(kwargs)
    client_kwargs.update({
        "public_key": public_key,
        "secret_key": secret_key,
    })
    if resolved_base_url:
      client_kwargs["base_url"] = resolved_base_url

    try:
      self._client = _Langfuse(**client_kwargs)
      self.enabled = True
    except Exception as e:  # pragma: no cover - optional dependency runtime
      logging.warning("Langfuse initialization failed: %s", e)
      self._client = None
      self.enabled = False

  def _apply_session_id(
      self, observation: Any | None, session_id: str | None
  ) -> Any | None:
    """Attach a document/session identifier to the trace when possible."""
    if not session_id:
      return None

    # Preferred: set the trace attribute directly from observation handle.
    if observation is not None and hasattr(observation, "update_trace"):
      try:
        update_trace_fn = observation.update_trace
        kwargs = _filter_kwargs_for_callable(
            update_trace_fn, {"session_id": session_id}
        )
        update_trace_fn(**kwargs)
        return None
      except (
          Exception
      ) as e:  # pragma: no cover - observer failures are non-fatal
        logging.warning("Langfuse update_trace failed: %s", e)

    # Fallback: update currently active trace on client.
    if self._client is not None and hasattr(
        self._client, "update_current_trace"
    ):
      try:
        update_current_trace_fn = self._client.update_current_trace
        kwargs = _filter_kwargs_for_callable(
            update_current_trace_fn, {"session_id": session_id}
        )
        update_current_trace_fn(**kwargs)
        return None
      except (
          Exception
      ) as e:  # pragma: no cover - observer failures are non-fatal
        logging.warning("Langfuse update_current_trace failed: %s", e)

    # Compatibility fallback for SDK variants exposing propagate_attributes.
    propagate_fn = None
    if self._client is not None and hasattr(
        self._client, "propagate_attributes"
    ):
      propagate_fn = self._client.propagate_attributes
    elif callable(_propagate_attributes):
      propagate_fn = _propagate_attributes

    if propagate_fn is None:
      return None

    try:
      propagate_context = propagate_fn(session_id=session_id)
      if hasattr(propagate_context, "__enter__"):
        propagate_context.__enter__()
        return propagate_context
    except Exception as e:  # pragma: no cover - observer failures are non-fatal
      logging.warning("Langfuse propagate_attributes failed: %s", e)

    return None

  def _start_observation(
      self, observation_type: str, name: str, **kwargs: Any
  ) -> _ObservationHandle | _NoOpHandle:
    """Start an observation with safe compatibility handling."""
    if not self.enabled or self._client is None:
      return _NoOpHandle()

    payload = _normalize_observation_payload(kwargs)
    session_id = payload.pop("session_id", None)
    payload["as_type"] = observation_type
    payload["name"] = name

    start_fn = getattr(self._client, "start_as_current_observation", None)
    if not callable(start_fn):
      start_fn = getattr(self._client, "start_observation", None)

    if not callable(start_fn):
      logging.warning(
          "Langfuse client has no observation start method; telemetry disabled."
      )
      return _NoOpHandle()

    observation = None
    observation_context = None
    try:
      start_kwargs = _filter_kwargs_for_callable(start_fn, payload)
      result = start_fn(**start_kwargs)
      if hasattr(result, "__enter__"):
        observation_context = result
        observation = observation_context.__enter__()
      else:
        observation = result
    except Exception as e:  # pragma: no cover - observer failures are non-fatal
      logging.warning(
          "Langfuse observation start failed; continuing without tracing: %s", e
      )
      return _NoOpHandle()

    if observation is None:
      logging.warning(
          "Langfuse observation start returned no handle; continuing without"
          " tracing."
      )
      return _NoOpHandle()

    propagate_context = self._apply_session_id(observation, session_id)
    return _ObservationHandle(
        observation=observation,
        observation_context=observation_context,
        propagate_context=propagate_context,
    )

  def span(
      self,
      *,
      name: str,
      input: Mapping[str, Any] | None = None,
      output: Mapping[str, Any] | None = None,
      metadata: Mapping[str, Any] | None = None,
      session_id: str | None = None,
      **kwargs: Any,
  ) -> _ObservationHandle | _NoOpHandle:
    return self._start_observation(
        "span",
        name=name,
        input=input or {},
        output=output,
        metadata=metadata or {},
        session_id=session_id,
        **kwargs,
    )

  def generation(
      self,
      *,
      name: str,
      input: Mapping[str, Any] | None = None,
      output: Mapping[str, Any] | None = None,
      model: str | None = None,
      metadata: Mapping[str, Any] | None = None,
      session_id: str | None = None,
      usage: Mapping[str, Any] | None = None,
      usage_details: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> _ObservationHandle | _NoOpHandle:
    return self._start_observation(
        "generation",
        name=name,
        input=input or {},
        output=output,
        model=model,
        metadata=metadata or {},
        session_id=session_id,
        usage=usage,
        usage_details=usage_details,
        **kwargs,
    )

  def flush(self) -> None:
    if (
        not self.enabled
        or self._client is None
        or not hasattr(self._client, "flush")
    ):
      return
    try:
      self._client.flush()
    except Exception as e:  # pragma: no cover - observer failures are non-fatal
      logging.warning("Langfuse flush failed: %s", e)


def create_observer(
    provider: str = "none", provider_kwargs: Mapping[str, Any] | None = None
) -> NoOpObserver | LangfuseObserver:
  """Factory for optional observability providers."""
  normalized = (provider or "none").strip().lower()
  kwargs = dict(provider_kwargs or {})

  if normalized in {"", "none", "noop", "disabled", "off"}:
    return NoOpObserver()

  if normalized == "langfuse":
    return LangfuseObserver(**kwargs)

  raise ValueError(
      f"Unknown observer provider '{provider}'. Supported providers: none,"
      " langfuse."
  )
