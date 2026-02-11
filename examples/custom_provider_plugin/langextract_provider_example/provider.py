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

"""Minimal example of a custom provider plugin for LangExtract."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, Sequence

from langextract_provider_example import schema as custom_schema

import langextract as lx


@lx.providers.registry.register(
    r'^gemini',  # Matches Gemini model IDs (same as default provider)
)
@dataclasses.dataclass(init=False)
class CustomGeminiProvider(lx.inference.BaseLanguageModel):
  """Example custom LangExtract provider implementation.

  This demonstrates how to create a custom provider for LangExtract
  that can intercept and handle model requests. This example wraps
  the actual Gemini API to show how custom schemas integrate, but you
  would replace the Gemini calls with your own API or model implementation.

  Note: Since this registers the same pattern as the default Gemini provider,
  you must explicitly specify this provider when creating a model:

  config = lx.factory.ModelConfig(
      model_id="gemini-2.5-flash",
      provider="CustomGeminiProvider"
  )
  model = lx.factory.create_model(config)
  """

  model_id: str
  api_key: str | None
  temperature: float
  response_schema: dict[str, Any] | None = None
  enable_structured_output: bool = False
  _client: Any = dataclasses.field(repr=False, compare=False)

  def __init__(
      self,
      model_id: str = 'gemini-2.5-flash',
      api_key: str | None = None,
      temperature: float = 0.0,
      **kwargs: Any,
  ) -> None:
    """Initialize the custom provider.

    Args:
      model_id: The model ID.
      api_key: API key for the service.
      temperature: Sampling temperature.
      **kwargs: Additional parameters.
    """
    super().__init__()

    # TODO: Replace with your own client initialization
    try:
      from google import genai  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise lx.exceptions.InferenceConfigError(
          'This example requires google-genai package. '
          'Install with: pip install google-genai'
      ) from e

    self.model_id = model_id
    self.api_key = api_key
    self.temperature = temperature

    # Schema kwargs from CustomProviderSchema.to_provider_config()
    self.response_schema = kwargs.get('response_schema')
    self.enable_structured_output = kwargs.get(
        'enable_structured_output', False
    )

    # Store any additional kwargs for potential use
    self._extra_kwargs = kwargs

    if not self.api_key:
      raise lx.exceptions.InferenceConfigError(
          'API key required. Set GEMINI_API_KEY or pass api_key parameter.'
      )

    self._client = genai.Client(api_key=self.api_key)

  @classmethod
  def get_schema_class(cls) -> type[lx.schema.BaseSchema] | None:
    """Return our custom schema class.

    This allows LangExtract to use our custom schema implementation
    when use_schema_constraints=True is specified.

    Returns:
      Our custom schema class that will be used to generate constraints.
    """
    return custom_schema.CustomProviderSchema

  def apply_schema(self, schema_instance: lx.schema.BaseSchema | None) -> None:
    """Apply or clear schema configuration.

    This method is called by LangExtract to dynamically apply schema
    constraints after the provider is instantiated. It's important to
    handle both the application of a new schema and clearing (None).

    Args:
      schema_instance: The schema to apply, or None to clear existing schema.
    """
    super().apply_schema(schema_instance)

    if schema_instance:
      # Apply the new schema configuration
      config = schema_instance.to_provider_config()
      self.response_schema = config.get('response_schema')
      self.enable_structured_output = config.get(
          'enable_structured_output', False
      )
    else:
      # Clear the schema configuration
      self.response_schema = None
      self.enable_structured_output = False

  def infer(
      self, batch_prompts: Sequence[str], **kwargs: Any
  ) -> Iterator[Sequence[lx.inference.ScoredOutput]]:
    """Run inference on a batch of prompts.

    Args:
      batch_prompts: Input prompts to process.
      **kwargs: Additional generation parameters.

    Yields:
      Lists of ScoredOutputs, one per prompt.
    """
    config = {
        'temperature': kwargs.get('temperature', self.temperature),
    }

    # Add other parameters if provided
    for key in ['max_output_tokens', 'top_p', 'top_k']:
      if key in kwargs:
        config[key] = kwargs[key]

    # Apply schema constraints if configured
    if self.response_schema and self.enable_structured_output:
      # For Gemini, this ensures the model outputs JSON matching our schema
      # Adapt this section based on your actual provider's API requirements
      config['response_schema'] = self.response_schema
      config['response_mime_type'] = 'application/json'

    for prompt in batch_prompts:
      try:
        # TODO: Replace this with your own API/model calls
        response = self._client.models.generate_content(
            model=self.model_id, contents=prompt, config=config
        )
        output = response.text.strip()
        usage = self._extract_usage(response)
        yield [
            lx.inference.ScoredOutput(
                score=1.0,
                output=output,
                usage=usage,
            )
        ]

      except Exception as e:
        raise lx.exceptions.InferenceRuntimeError(
            f'API error: {str(e)}', original=e
        ) from e

  @staticmethod
  def _extract_usage(response: Any) -> dict[str, Any] | None:
    """Extract usage metadata from Gemini response payload.

    Returns a plain mapping so the example stays decoupled from internal types.
    """
    usage_obj = getattr(response, 'usage_metadata', None) or getattr(
        response, 'usageMetadata', None
    )
    if usage_obj is None:
      return None

    def _get_value(obj: Any, snake: str, camel: str) -> Any:
      if isinstance(obj, dict):
        if snake in obj:
          return obj[snake]
        return obj.get(camel)
      value = getattr(obj, snake, None)
      if value is not None:
        return value
      return getattr(obj, camel, None)

    def _int_or_none(value: Any) -> int | None:
      if isinstance(value, bool):
        return None
      return value if isinstance(value, int) else None

    input_tokens = _int_or_none(
        _get_value(usage_obj, 'prompt_token_count', 'promptTokenCount')
    )
    output_tokens = _int_or_none(
        _get_value(usage_obj, 'candidates_token_count', 'candidatesTokenCount')
    )
    total_tokens = _int_or_none(
        _get_value(usage_obj, 'total_token_count', 'totalTokenCount')
    )

    provider_details = {}
    for snake, camel in (
        ('cached_content_token_count', 'cachedContentTokenCount'),
        ('tool_use_prompt_token_count', 'toolUsePromptTokenCount'),
        ('thoughts_token_count', 'thoughtsTokenCount'),
    ):
      value = _int_or_none(_get_value(usage_obj, snake, camel))
      if value is not None:
        provider_details[snake] = value

    if (
        input_tokens is None
        and output_tokens is None
        and total_tokens is None
        and not provider_details
    ):
      return None

    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'provider_details': provider_details or None,
    }
