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

"""Public exceptions API for LangExtract.

This module re-exports exceptions from core.exceptions for backward compatibility.
All new code should import directly from langextract.core.exceptions.
"""
# pylint: disable=duplicate-code

from __future__ import annotations

from langextract.core import exceptions as core_exceptions

# Backward compat re-exports
InferenceConfigError = core_exceptions.InferenceConfigError
InferenceError = core_exceptions.InferenceError
InferenceOutputError = core_exceptions.InferenceOutputError
InferenceRuntimeError = core_exceptions.InferenceRuntimeError
LangExtractError = core_exceptions.LangExtractError
ProviderError = core_exceptions.ProviderError
SchemaError = core_exceptions.SchemaError

__all__ = [
    "LangExtractError",
    "InferenceError",
    "InferenceConfigError",
    "InferenceRuntimeError",
    "InferenceOutputError",
    "ProviderError",
    "SchemaError",
]
