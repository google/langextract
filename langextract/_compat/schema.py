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

"""Compatibility shim for langextract.schema imports."""
# pylint: disable=duplicate-code

from __future__ import annotations

import warnings


def __getattr__(name: str):
  moved = {
      "BaseSchema": ("langextract.core.schema", "BaseSchema"),
      "Constraint": ("langextract.core.schema", "Constraint"),
      "ConstraintType": ("langextract.core.schema", "ConstraintType"),
      "EXTRACTIONS_KEY": ("langextract.core.schema", "EXTRACTIONS_KEY"),
      "GeminiSchema": ("langextract.providers.schemas.gemini", "GeminiSchema"),
  }
  if name in moved:
    mod, attr = moved[name]
    warnings.warn(
        f"`langextract.schema.{name}` is deprecated and will be removed in"
        f" v2.0.0; use `{mod}.{attr}` instead.",
        FutureWarning,
        stacklevel=2,
    )
    module = __import__(mod, fromlist=[attr])
    return getattr(module, attr)
  raise AttributeError(name)
