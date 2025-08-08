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

"""Simple library for performing language model inference."""

import abc
from collections.abc import Iterator, Sequence
import dataclasses
import enum
import json
import textwrap
from typing import Any

from absl import logging
from typing_extensions import deprecated
import yaml

from langextract import data
from langextract import exceptions
from langextract import schema

_OLLAMA_DEFAULT_MODEL_URL = 'http://localhost:11434'


@dataclasses.dataclass(frozen=True)
class ScoredOutput:
  """Scored output."""

  score: float | None = None
  output: str | None = None

  def __str__(self) -> str:
    score_str = '-' if self.score is None else f'{self.score:.2f}'
    if self.output is None:
      return f'Score: {score_str}\nOutput: None'
    formatted_lines = textwrap.indent(self.output, prefix='  ')
    return f'Score: {score_str}\nOutput:\n{formatted_lines}'


class InferenceOutputError(exceptions.LangExtractError):
  """Exception raised when no scored outputs are available from the language model."""

  def __init__(self, message: str):
    self.message = message
    super().__init__(self.message)


class BaseLanguageModel(abc.ABC):
  """An abstract inference class for managing LLM inference.

  Attributes:
    _constraint: A `Constraint` object specifying constraints for model output.
  """

  def __init__(self, constraint: schema.Constraint = schema.Constraint()):
    """Initializes the BaseLanguageModel with an optional constraint.

    Args:
      constraint: Applies constraints when decoding the output. Defaults to no
        constraint.
    """
    self._constraint = constraint

  @abc.abstractmethod
  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Implements language model inference.

    Args:
      batch_prompts: Batch of inputs for inference. Single element list can be
        used for a single input.
      **kwargs: Additional arguments for inference, like temperature and
        max_decode_steps.

    Returns: Batch of Sequence of probable output text outputs, sorted by
      descending
      score.
    """

  def infer_batch(
      self, prompts: Sequence[str], batch_size: int = 32  # pylint: disable=unused-argument
  ) -> list[list[ScoredOutput]]:
    """Batch inference with configurable batch size.

    This is a convenience method that collects all results from infer().

    Args:
      prompts: List of prompts to process.
      batch_size: Batch size (currently unused, for future optimization).

    Returns:
      List of lists of ScoredOutput objects.
    """
    results = []
    for output in self.infer(prompts):
      results.append(list(output))
    return results

  def parse_output(self, output: str) -> Any:
    """Parses model output as JSON or YAML.

    Note: This expects raw JSON/YAML without code fences.
    Code fence extraction is handled by resolver.py.

    Args:
      output: Raw output string from the model.

    Returns:
      Parsed Python object (dict or list).

    Raises:
      ValueError: If output cannot be parsed as JSON or YAML.
    """
    # Check if we have a format_type attribute (providers should set this)
    format_type = getattr(self, 'format_type', data.FormatType.JSON)

    try:
      if format_type == data.FormatType.JSON:
        return json.loads(output)
      else:
        return yaml.safe_load(output)
    except Exception as e:
      raise ValueError(
          f'Failed to parse output as {format_type.name}: {str(e)}'
      ) from e


class InferenceType(enum.Enum):
  ITERATIVE = 'iterative'
  MULTIPROCESS = 'multiprocess'


@deprecated(
    'Use langextract.providers.ollama.OllamaLanguageModel instead. '
    'Will be removed in v2.0.0.'
)
class OllamaLanguageModel(BaseLanguageModel):
  """Language model inference class using Ollama based host.

  DEPRECATED: Use langextract.providers.ollama.OllamaLanguageModel instead.
  This class is kept for backward compatibility only.
  """

  def __init__(self, **kwargs):
    """Initialize the Ollama language model (deprecated)."""
    logging.warning(
        'OllamaLanguageModel from langextract.inference is deprecated. '
        'Use langextract.providers.ollama.OllamaLanguageModel instead.'
    )

    # pylint: disable=import-outside-toplevel
    from langextract.providers import ollama  # Avoid circular import

    # Convert old parameter names to new ones
    if 'model' in kwargs:
      kwargs['model_id'] = kwargs.pop('model')

    if 'structured_output_format' in kwargs:
      format_str = kwargs.pop('structured_output_format')
      kwargs['format_type'] = (
          data.FormatType.JSON if format_str == 'json' else data.FormatType.YAML
      )

    self._impl = ollama.OllamaLanguageModel(**kwargs)
    self._model = self._impl._model
    self._model_url = self._impl._model_url
    self.format_type = (
        self._impl.format_type
    )  # Changed from _structured_output_format
    self._constraint = self._impl._constraint
    self._extra_kwargs = self._impl._extra_kwargs

    super().__init__(constraint=self._impl._constraint)

  def _ollama_query(self, **kwargs):
    """Backward compatibility method."""
    return self._impl._ollama_query(**kwargs)  # pylint: disable=protected-access

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Delegate to new provider."""
    return self._impl.infer(batch_prompts, **kwargs)

  def parse_output(self, output: str) -> Any:
    """Delegate to new provider."""
    return self._impl.parse_output(output)


@deprecated(
    'Use langextract.providers.gemini.GeminiLanguageModel instead. '
    'Will be removed in v2.0.0.'
)
class GeminiLanguageModel(BaseLanguageModel):
  """Language model inference using Google's Gemini API with structured output.

  DEPRECATED: Use langextract.providers.gemini.GeminiLanguageModel instead.
  This class is kept for backward compatibility only.
  """

  def __init__(self, **kwargs):
    """Initialize the Gemini language model (deprecated)."""
    logging.warning(
        'GeminiLanguageModel from langextract.inference is deprecated. '
        'Use langextract.providers.gemini.GeminiLanguageModel instead.'
    )

    # pylint: disable=import-outside-toplevel
    from langextract.providers import gemini  # Avoid circular import

    self._impl = gemini.GeminiLanguageModel(**kwargs)
    self.model_id = self._impl.model_id
    self.api_key = self._impl.api_key
    self.gemini_schema = self._impl.gemini_schema
    self.format_type = self._impl.format_type
    self.temperature = self._impl.temperature
    self.max_workers = self._impl.max_workers

    super().__init__(constraint=self._impl._constraint)

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Delegate to new provider."""
    return self._impl.infer(batch_prompts, **kwargs)

  def parse_output(self, output: str) -> Any:
    """Delegate to new provider."""
    return self._impl.parse_output(output)


@deprecated(
    'Use langextract.providers.openai.OpenAILanguageModel instead. '
    'Will be removed in v2.0.0.'
)
class OpenAILanguageModel(BaseLanguageModel):
  """Language model inference using OpenAI's API with structured output.

  DEPRECATED: Use langextract.providers.openai.OpenAILanguageModel instead.
  This class is kept for backward compatibility only.
  """

  def __init__(self, **kwargs):
    """Initialize the OpenAI language model (deprecated)."""
    logging.warning(
        'OpenAILanguageModel from langextract.inference is deprecated. '
        'Use langextract.providers.openai.OpenAILanguageModel instead.'
    )

    # pylint: disable=import-outside-toplevel
    from langextract.providers import openai  # Avoid circular import

    try:
      self._impl = openai.OpenAILanguageModel(**kwargs)
    except exceptions.InferenceConfigError as e:
      # Convert to ValueError for backward compatibility
      raise ValueError(
          str(e).replace(
              'API key not provided for OpenAI.', 'API key not provided.'
          )
      ) from e
    self.model_id = self._impl.model_id
    self.api_key = self._impl.api_key
    self.base_url = self._impl.base_url
    self.organization = self._impl.organization
    self.format_type = self._impl.format_type
    self.temperature = self._impl.temperature
    self.max_workers = self._impl.max_workers
    self._client = self._impl._client

    self._process_single_prompt = (
        self._impl._process_single_prompt
    )  # For test compatibility

    super().__init__(constraint=self._impl._constraint)

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[ScoredOutput]]:
    """Delegate to new provider."""
    return self._impl.infer(batch_prompts, **kwargs)

  def parse_output(self, output: str) -> Any:
    """Delegate to new provider."""
    return self._impl.parse_output(output)
