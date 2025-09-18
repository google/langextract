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

"""Compatibility shim for langextract.inference imports."""

from __future__ import annotations

import enum
import warnings


class InferenceType(enum.Enum):
    """Enum for inference types - kept for backward compatibility."""

    ITERATIVE = "iterative"
    MULTIPROCESS = "multiprocess"


def __getattr__(name: str):
    moved = {
        "BaseLanguageModel": ("langextract.core.base_model", "BaseLanguageModel"),
        "ScoredOutput": ("langextract.core.types", "ScoredOutput"),
        "InferenceOutputError": (
            "langextract.core.exceptions",
            "InferenceOutputError",
        ),
        "GeminiLanguageModel": (
            "langextract.providers.gemini",
            "GeminiLanguageModel",
        ),
        "OpenAILanguageModel": (
            "langextract.providers.openai",
            "OpenAILanguageModel",
        ),
        "OllamaLanguageModel": (
            "langextract.providers.ollama",
            "OllamaLanguageModel",
        ),
    }
    if name in moved:
        mod, attr = moved[name]
        warnings.warn(
            f"`langextract.inference.{name}` is deprecated and will be removed in"
            f" v2.0.0; use `{mod}.{attr}` instead.",
            FutureWarning,
            stacklevel=2,
        )
        module = __import__(mod, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(name)
