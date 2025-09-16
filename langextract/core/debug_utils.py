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

"""Debug utilities for LangExtract."""
from __future__ import annotations

import functools
import inspect
import logging
import reprlib
import time
from typing import Any, Callable, Mapping

from absl import logging as absl_logging

_LOG = logging.getLogger("langextract.debug")

# Add NullHandler to prevent "No handler found" warnings
_langextract_logger = logging.getLogger("langextract")
if not _langextract_logger.handlers:
    _langextract_logger.addHandler(logging.NullHandler())

# Sensitive keys to redact
_REDACT_KEYS = {
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "bearer",
    "jwt",
}
_MAX_STR = 500
_MAX_SEQ = 20


def _safe_repr(obj: Any) -> str:
    """Truncate object repr for safe logging."""
    r = reprlib.Repr()
    r.maxstring = _MAX_STR
    r.maxlist = r.maxtuple = r.maxset = r.maxdict = _MAX_SEQ
    return r.repr(obj)


def _redact_value(name: str, value: Any) -> str:
    """Redact sensitive values based on parameter name."""
    if isinstance(name, str) and name.lower() in _REDACT_KEYS:
        return "<REDACTED>"
    # If a nested mapping, redact its sensitive keys too
    if isinstance(value, Mapping):
        redacted = {}
        for k, v in value.items():
            if isinstance(k, str) and k.lower() in _REDACT_KEYS:
                redacted[k] = "<REDACTED>"
            else:
                redacted[k] = _safe_repr(v)
        return _safe_repr(redacted)
    return _safe_repr(value)


def _redact_mapping(mapping: Mapping[str, Any]) -> dict[str, str]:
    """Replace sensitive values with <REDACTED>."""
    out = {}
    for k, v in mapping.items():
        out[k] = _redact_value(k, v)
    return out


def _format_bound_args(
    fn: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    """Format function arguments using signature inspection."""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except Exception:
        # Fallback (no names) if binding fails
        parts = [_safe_repr(a) for a in args]
        if kwargs:
            red = _redact_mapping(kwargs)
            parts += [f"{k}={v}" for k, v in sorted(red.items())]
        return ", ".join(parts)

    parts: list[str] = []
    for name, value in bound.arguments.items():
        if name in ("self", "cls"):
            parts.append(f"{name}=<{type(value).__name__}>")
        else:
            parts.append(f"{name}={_redact_value(name, value)}")
    return ", ".join(parts)


def debug_log_calls(fn: Callable) -> Callable:
    """Log function calls with redacted sensitive data and timing.

    Automatically redacts api_key, token, etc. and truncates large outputs.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        logger = _LOG
        if not logger.isEnabledFor(logging.DEBUG):
            return fn(*args, **kwargs)

        fn_qual = getattr(fn, "__qualname__", fn.__name__)
        mod = getattr(fn, "__module__", "")

        # Format arguments using signature inspection
        arg_str = _format_bound_args(fn, args, kwargs)

        logger.debug("[%s] CALL: %s(%s)", mod, fn_qual, arg_str, stacklevel=2)

        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception:
            dur_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "[%s] EXCEPTION: %s (%.1f ms)", mod, fn_qual, dur_ms, stacklevel=2
            )
            raise

        dur_ms = (time.perf_counter() - start) * 1000
        result_repr = _safe_repr(result)
        logger.debug(
            "[%s] RETURN: %s -> %s (%.1f ms)",
            mod,
            fn_qual,
            result_repr,
            dur_ms,
            stacklevel=2,
        )
        return result

    return wrapper


def configure_debug_logging() -> None:
    """Enable debug logging for the 'langextract' namespace only."""
    logger = logging.getLogger("langextract")

    # Skip if we already added our handler
    our_handler_exists = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "langextract_debug", False)
        for h in logger.handlers
    )
    if our_handler_exists:
        return

    # Respect host handlers - only set level if they exist
    non_null_handlers = [
        h for h in logger.handlers if not isinstance(h, logging.NullHandler)
    ]

    if non_null_handlers:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        handler.langextract_debug = True
        logger.addHandler(handler)
        logger.propagate = False

    # Best-effort absl configuration
    try:
        absl_logging.set_verbosity(absl_logging.DEBUG)
    except Exception:
        pass
