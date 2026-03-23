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

"""MiniMax provider for LangExtract.

This provider uses MiniMax's OpenAI-compatible API to extract structured
information from text.

Usage:
    # Using factory
    from langextract.factory import ModelConfig, create_model

    config = ModelConfig(
        model_id="MiniMax-M2.5",
        provider="MiniMaxLanguageModel",
        provider_kwargs={
            "api_key": "your-minimax-api-key"
        }
    )
    model = create_model(config)

    result = lx.extract(
        text_or_documents=text,
        prompt_description=instructions,
        model=model
    )
"""

from __future__ import annotations

import dataclasses
from typing import Any

from langextract.core import base_model
from langextract.core import data
from langextract.providers import patterns
from langextract.providers import router

_DEFAULT_MODEL_ID = "MiniMax-M2.5"
_DEFAULT_BASE_URL = "https://api.minimax.io/v1"


@router.register(
    *patterns.MINIMAX_PATTERNS,
    priority=patterns.MINIMAX_PRIORITY,
)
@dataclasses.dataclass(init=False)
class MiniMaxLanguageModel(base_model.BaseLanguageModel):
  """Language model inference using MiniMax's OpenAI-compatible API."""

  model_id: str = _DEFAULT_MODEL_ID
  api_key: str | None = None
  base_url: str = _DEFAULT_BASE_URL
  organization: str | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float | None = None
  max_workers: int = 10
  _client: Any = dataclasses.field(default=None, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @property
  def requires_fence_output(self) -> bool:
    """MiniMax returns raw JSON without fences."""
    if self.format_type == data.FormatType.JSON:
      return False
    return super().requires_fence_output

  def __post_init__(self):
    """Initialize the OpenAI client with MiniMax configuration."""
    try:
      from openai import AsyncOpenAI
    except ImportError as e:
      raise ImportError(
          "OpenAI package is required for MiniMax provider. "
          "Install with: pip install langextract[openai]"
      ) from e

    if self._client is None:
      self._client = AsyncOpenAI(
          api_key=self.api_key,
          base_url=self.base_url,
          organization=self.organization,
          **self._extra_kwargs,
      )

  async def _generate(
      self,
      texts: list[str],
      prompt_description: str,
      extra_params: dict[str, Any] | None = None,
  ) -> list[list[base_model.ExtractionCandidate]]:
    """Generate extractions for the given texts."""
    import asyncio

    extra_params = extra_params or {}

    async def process_single(text: str) -> list[base_model.ExtractionCandidate]:
      response = await self._client.chat.completions.create(
          model=self.model_id,
          messages=[
              {
                  "role": "system",
                  "content": (
                      "You are a helpful assistant that extracts structured"
                      " information from text."
                  ),
              },
              {
                  "role": "user",
                  "content": f"{prompt_description}\n\nText: {text}",
              },
          ],
          response_format={"type": "json_object"}
          if self.format_type == data.FormatType.JSON
          else None,
          temperature=self.temperature,
          **extra_params,
      )

      content = response.choices[0].message.content
      if not content:
        return []

      try:
        import json

        data = json.loads(content)
        # Wrap in ExtractionCandidate format
        if isinstance(data, list):
          return [
              base_model.ExtractionCandidate(
                  extraction_text=item.get("text", str(item)),
                  extraction_class=item.get("class", "unknown"),
                  extraction_index=i,
              )
              for i, item in enumerate(data)
          ]
        elif isinstance(data, dict):
          # For single object extractions
          return [
              base_model.ExtractionCandidate(
                  extraction_text=str(v),
                  extraction_class=k,
                  extraction_index=i,
              )
              for i, (k, v) in enumerate(data.items())
          ]
      except (json.JSONDecodeError, AttributeError):
        pass

      return [
          base_model.ExtractionCandidate(
              extraction_text=content,
              extraction_class="extracted",
              extraction_index=0,
          )
      ]

    # Process texts in parallel
    tasks = [process_single(text) for text in texts]
    results = await asyncio.gather(*tasks)
    return results

  def _generate_sync(
      self,
      texts: list[str],
      prompt_description: str,
      extra_params: dict[str, Any] | None = None,
  ) -> list[list[base_model.ExtractionCandidate]]:
    """Synchronous wrapper for generation."""
    import asyncio

    try:
      loop = asyncio.get_event_loop()
      if loop.is_running():
        # If we're in an async context, we need to create a new loop
        # This is a simplified sync wrapper - for production use async directly
        import concurrent.futures

        def run_in_executor():
          return asyncio.run(
              self._generate(texts, prompt_description, extra_params)
          )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
          future = executor.submit(run_in_executor)
          return future.result()
    except RuntimeError:
      # No event loop, run directly
      return asyncio.run(
          self._generate(texts, prompt_description, extra_params)
      )

  def __call__(
      self,
      texts: Sequence[str],
      prompt_description: str,
      extra_params: dict[str, Any] | None = None,
  ) -> list[list[base_model.ExtractionCandidate]]:
    """Synchronous interface for the model."""
    return self._generate_sync(list(texts), prompt_description, extra_params)

  async def _call_async(
      self,
      texts: Sequence[str],
      prompt_description: str,
      extra_params: dict[str, Any] | None = None,
  ) -> list[list[base_model.ExtractionCandidate]]:
    """Asynchronous interface for the model."""
    return await self._generate(list(texts), prompt_description, extra_params)

  def close(self):
    """Close the client connection."""
    # AsyncOpenAI doesn't need explicit close
    pass
