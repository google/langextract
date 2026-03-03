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

"""Novita provider for LangExtract using OpenAI-compatible API."""

from __future__ import annotations

import dataclasses

from langextract.providers import openai
from langextract.providers import patterns
from langextract.providers import router

_DEFAULT_MODEL_ID = "deepseek/deepseek-v3.2"
_DEFAULT_BASE_URL = "https://api.novita.ai/openai"


@router.register(
    *patterns.NOVITA_PATTERNS,
    priority=patterns.NOVITA_PRIORITY,
)
@dataclasses.dataclass(init=False)
class NovitaLanguageModel(openai.OpenAILanguageModel):
  """Language model inference using Novita's OpenAI-compatible API."""

  def __init__(
      self,
      model_id: str = _DEFAULT_MODEL_ID,
      api_key: str | None = None,
      base_url: str | None = _DEFAULT_BASE_URL,
      **kwargs,
  ) -> None:
    """Initialize Novita provider with a Novita default base URL."""
    super().__init__(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
