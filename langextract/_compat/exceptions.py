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

"""Compatibility shim for langextract.exceptions imports."""
# pylint: disable=duplicate-code

from __future__ import annotations

import warnings

from langextract.core import exceptions


# Re-export exceptions from core.exceptions with a warning-on-first-access
def __getattr__(name: str):
    allowed = {
        "LangExtractError",
        "InferenceError",
        "InferenceConfigError",
        "InferenceRuntimeError",
        "InferenceOutputError",
        "ProviderError",
        "SchemaError",
    }
    if name in allowed:
        warnings.warn(
            "`langextract.exceptions` is deprecated; import from"
            " `langextract.core.exceptions`.",
            FutureWarning,
            stacklevel=2,
        )
        return getattr(exceptions, name)
    raise AttributeError(name)
