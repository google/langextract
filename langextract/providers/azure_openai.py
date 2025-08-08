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

"""Azure OpenAI provider for LangExtract.

Inherits from OpenAILanguageModel and only overrides client initialization
to use AzureOpenAI instead of OpenAI. All inference logic is reused.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from langextract import data
from langextract import exceptions
from langextract.providers import registry
from langextract.providers.openai import OpenAILanguageModel


@registry.register(
    r"^azure:",  # e.g., azure:<deployment>
    r"^aoai:",  # e.g., aoai:<deployment>
    r"^AzureOpenAILanguageModel$",  # explicit provider selection
    priority=20,
)
@dataclasses.dataclass(init=False)
class AzureOpenAILanguageModel(OpenAILanguageModel):
  """Azure OpenAI language model that inherits OpenAI inference logic.

  Only differs from OpenAILanguageModel in client initialization - uses
  AzureOpenAI instead of OpenAI client.
  """

  azure_endpoint: str | None = None
  api_version: str = "2024-07-01-preview"
  _deployment: str = dataclasses.field(default="", repr=False, compare=False)

  def __init__(
      self,
      model_id: str = "azure:deployment",
      azure_endpoint: str | None = None,
      api_key: str | None = None,
      api_version: str = "2024-07-01-preview",
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      **kwargs: Any,
  ) -> None:
    # Lazy import to avoid hard dependency unless provider is used
    try:
      # pylint: disable=import-outside-toplevel
      from openai import AzureOpenAI  # type: ignore
    except Exception as exc:
      raise exceptions.InferenceConfigError(
          "Azure OpenAI provider requires the 'openai' package. "
          "Install with: pip install langextract[openai]"
      ) from exc

    # Set Azure-specific attributes
    self.azure_endpoint = azure_endpoint
    self.api_version = api_version

    # Determine deployment name from model_id
    if model_id.startswith("azure:"):
      self._deployment = model_id.split(":", 1)[1]
    elif model_id.startswith("aoai:"):
      self._deployment = model_id.split(":", 1)[1]
    else:
      self._deployment = model_id

    # Validate required parameters
    if not api_key:
      raise exceptions.InferenceConfigError(
          "API key not provided for Azure OpenAI."
      )
    if not self.azure_endpoint:
      raise exceptions.InferenceConfigError(
          "Azure endpoint not provided for Azure OpenAI."
      )

    # Initialize parent class with OpenAI-compatible parameters
    # We'll override the client creation below
    super().__init__(
        model_id=self._deployment,  # Use deployment as model_id for parent
        api_key=api_key,
        base_url=None,
        organization=None,
        format_type=format_type,
        temperature=temperature,
        max_workers=max_workers,
        **kwargs,
    )

    # Override client with Azure-specific client
    self._client = AzureOpenAI(
        api_key=api_key,
        api_version=self.api_version,
        azure_endpoint=self.azure_endpoint,
    )

  # All other methods (infer, _process_single_prompt) are inherited from OpenAILanguageModel
