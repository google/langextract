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

"""xAI (Grok) provider for LangExtract.

Grok is OpenAI-API-compatible at https://api.x.ai/v1. This provider auto-routes
`grok-*` model IDs and injects XAI_API_KEY + the xAI base URL.
"""

from __future__ import annotations

import os

from langextract.providers import latest_models
from langextract.providers import openai as openai_provider
from langextract.providers import patterns
from langextract.providers import router


@router.register(
    *patterns.XAI_PATTERNS,
    priority=patterns.XAI_PRIORITY,
)
class XAILanguageModel(openai_provider.OpenAILanguageModel):
  """Grok models via xAI's OpenAI-compatible Chat Completions API."""

  model_id: str = latest_models.GROK_DEFAULT

  def __init__(
      self,
      model_id: str = latest_models.GROK_DEFAULT,
      api_key: str | None = None,
      base_url: str | None = None,
      **kwargs,
  ) -> None:
    resolved_key = api_key or os.environ.get('XAI_API_KEY')
    resolved_url = (
        base_url or os.environ.get('XAI_BASE_URL') or latest_models.XAI_BASE_URL
    )
    super().__init__(
        model_id=model_id or latest_models.GROK_DEFAULT,
        api_key=resolved_key,
        base_url=resolved_url,
        **kwargs,
    )
