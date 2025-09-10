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

"""Ollama provider for LangExtract.

This provider enables using local Ollama models with LangExtract's extract() function.
No API key is required since Ollama runs locally on your machine.

Usage with extract():
    import langextract as lx
    from langextract.data import ExampleData, Extraction

    # Create an example for few-shot learning
    example = ExampleData(
        text="Marie Curie was a pioneering physicist and chemist.",
        extractions=[
            Extraction(
                extraction_class="person",
                extraction_text="Marie Curie",
                attributes={"name": "Marie Curie", "field": "physics and chemistry"}
            )
        ]
    )

    # Basic usage with Ollama
    result = lx.extract(
        text_or_documents="Isaac Asimov was a prolific science fiction writer.",
        model_id="gemma2:2b",
        prompt_description="Extract the person's name and field",
        examples=[example],
    )

Direct provider instantiation (when model ID conflicts with other providers):
    from langextract.providers.ollama import OllamaLanguageModel

    # Create Ollama provider directly
    model = OllamaLanguageModel(
        model_id="gemma2:2b",
        model_url="http://localhost:11434",  # optional, uses default if not specified
    )

    # Use with extract by passing the model instance
    result = lx.extract(
        text_or_documents="Your text here",
        model=model,  # Pass the model instance directly
        prompt_description="Extract information",
        examples=[example],
    )

Using pre-configured FormatHandler for manual control:
    from langextract.providers.ollama import OLLAMA_FORMAT_HANDLER

    # Use the pre-configured Ollama FormatHandler
    result = lx.extract(
        text_or_documents="Your text here",
        model_id="gemma2:2b",
        prompt_description="Extract information",
        examples=[example],
        resolver_params={'format_handler': OLLAMA_FORMAT_HANDLER}
    )

Supported model ID formats:
    - Standard Ollama: llama3.2:1b, gemma2:2b, mistral:7b, qwen2.5:7b, etc.
    - Hugging Face style: meta-llama/Llama-3.2-1B-Instruct, google/gemma-2b, etc.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull the model: ollama pull gemma2:2b
    3. Ollama server will start automatically when you use extract()
"""
# pylint: disable=duplicate-code

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urljoin
from urllib.parse import urlparse
import warnings

import requests

# Import from core modules directly
from langextract.core import base_model
from langextract.core import exceptions
from langextract.core import format_handler as fh
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import patterns
from langextract.providers import router

# Ollama defaults
_OLLAMA_DEFAULT_MODEL_URL = 'http://localhost:11434'
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_TIMEOUT = 120
_DEFAULT_KEEP_ALIVE = 5 * 60  # 5 minutes
_DEFAULT_NUM_CTX = 2048

# Pre-configured FormatHandler for consistent Ollama configuration
# use_wrapper=True creates {"extractions": [...]} vs just [...] 
# Ollama's JSON mode expects a dictionary root, not a bare list
OLLAMA_FORMAT_HANDLER = fh.FormatHandler(
    format_type=data.FormatType.JSON,
    use_wrapper=True,
    wrapper_key=None,
    use_fences=False,
    strict_fences=False,
)


@router.register(
    *patterns.OLLAMA_PATTERNS,
    priority=patterns.OLLAMA_PRIORITY,
)
@dataclasses.dataclass(init=False)
class OllamaLanguageModel(base_model.BaseLanguageModel):
  """Language model inference class using Ollama based host.

  Timeout can be set via constructor or passed through lx.extract():
    lx.extract(..., language_model_params={"timeout": 300})

  Authentication is supported for proxied Ollama instances:
    lx.extract(..., language_model_params={"api_key": "sk-..."})
  """

  _model: str
  _model_url: str
  format_type: core_types.FormatType = core_types.FormatType.JSON
  _constraint: schema.Constraint = dataclasses.field(
      default_factory=schema.Constraint, repr=False, compare=False
  )
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )
  # Authentication
  _api_key: str | None = None
  _auth_scheme: str = 'Bearer'
  _auth_header: str = 'Authorization'

  @classmethod
  def get_schema_class(cls) -> type[schema.BaseSchema] | None:
    """Return the FormatModeSchema class for JSON output support.

    Returns:
      The FormatModeSchema class that enables JSON mode (non-strict).
    """
    return schema.FormatModeSchema

  def __repr__(self) -> str:
    """Return string representation with redacted API key."""
    api_key_display = '[REDACTED]' if self._api_key else None
    return (
        f'{self.__class__.__name__}('
        f'model={self._model!r}, '
        f'model_url={self._model_url!r}, '
        f'format_type={self.format_type!r}, '
        f'api_key={api_key_display})'
    )

  def __init__(
      self,
      model_id: str,
      model_url: str = _OLLAMA_DEFAULT_MODEL_URL,
      base_url: str | None = None,  # Alias for model_url
      format_type: core_types.FormatType | None = None,
      structured_output_format: str | None = None,  # Deprecated
      constraint: schema.Constraint = schema.Constraint(),
      timeout: int | None = None,
      **kwargs,
  ) -> None:
    """Initialize the Ollama language model.

    Args:
      model_id: The Ollama model ID to use.
      model_url: URL for Ollama server (legacy parameter).
      base_url: Alternative parameter name for Ollama server URL.
      format_type: Output format (JSON or YAML). Defaults to JSON.
      structured_output_format: DEPRECATED - use format_type instead.
      constraint: Schema constraints.
      timeout: Request timeout in seconds. Defaults to 120.
      **kwargs: Additional parameters.
    """
    self._requests = requests

    # Handle deprecated structured_output_format parameter
    if structured_output_format is not None:
      warnings.warn(
          "'structured_output_format' is deprecated and will be removed in "
          "v2.0.0. Use 'format_type' instead.",
          FutureWarning,
          stacklevel=2,
      )
      if format_type is None:
        format_type = (
            core_types.FormatType.JSON
            if structured_output_format == 'json'
            else core_types.FormatType.YAML
        )

    fmt = kwargs.pop('format', None)
    if format_type is None and fmt in ('json', 'yaml'):
      format_type = (
          core_types.FormatType.JSON
          if fmt == 'json'
          else core_types.FormatType.YAML
      )

    if format_type is None:
      format_type = core_types.FormatType.JSON

    self._model = model_id
    self._model_url = base_url or model_url or _OLLAMA_DEFAULT_MODEL_URL
    self.format_type = format_type
    self._constraint = constraint

    self._api_key = kwargs.pop('api_key', None)
    self._auth_scheme = kwargs.pop('auth_scheme', 'Bearer')
    self._auth_header = kwargs.pop('auth_header', 'Authorization')

    if self._api_key:
      host = urlparse(self._model_url).hostname
      if host in ('localhost', '127.0.0.1', '::1'):
        warnings.warn(
            'API key provided for localhost Ollama instance. '
            "Native Ollama doesn't require authentication. "
            'This is typically only needed for proxied instances.',
            UserWarning,
        )

    super().__init__(constraint=constraint)
    if timeout is not None:
      kwargs['timeout'] = timeout
    self._extra_kwargs = kwargs or {}

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[core_types.ScoredOutput]]:
    """Runs inference on a list of prompts via Ollama's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params.

    Yields:
      Lists of ScoredOutputs.
    """
    combined_kwargs = self.merge_kwargs(kwargs)

    for prompt in batch_prompts:
      try:
        response = self._ollama_query(
            prompt=prompt,
            model=self._model,
            structured_output_format='json'
            if self.format_type == core_types.FormatType.JSON
            else 'yaml',
            model_url=self._model_url,
            **combined_kwargs,
        )
        yield [core_types.ScoredOutput(score=1.0, output=response['response'])]
      except Exception as e:
        raise exceptions.InferenceRuntimeError(
            f'Ollama API error: {str(e)}', original=e
        ) from e

  def _ollama_query(
      self,
      prompt: str,
      model: str | None = None,
      temperature: float | None = None,
      seed: int | None = None,
      top_k: int | None = None,
      top_p: float | None = None,
      max_output_tokens: int | None = None,
      structured_output_format: str | None = None,
      system: str = '',
      raw: bool = False,
      model_url: str | None = None,
      timeout: int | None = None,
      keep_alive: int | None = None,
      num_threads: int | None = None,
      num_ctx: int | None = None,
      stop: str | list[str] | None = None,
      **kwargs,
  ) -> Mapping[str, Any]:
    """Sends a prompt to an Ollama model and returns the generated response.

    Note: This is a low-level method. Constructor timeout is only used when
    calling through infer(). Direct calls use the timeout parameter here.

    This function makes an HTTP POST request to the `/api/generate` endpoint of
    an Ollama server. It can optionally load the specified model first, generate
    a response (with or without streaming), then return a parsed JSON response.

    Args:
      prompt: The text prompt to send to the model.
      model: The name of the model to use. Defaults to self._model.
      temperature: Sampling temperature. Higher values produce more diverse
        output.
      seed: Seed for reproducible generation. If None, random seed is used.
      top_k: The top-K parameter for sampling.
      top_p: The top-P (nucleus) sampling parameter.
      max_output_tokens: Maximum tokens to generate. If None, the model's
        default is used.
      structured_output_format: If set to "json" or a JSON schema dict, requests
        structured outputs from the model. See Ollama documentation for details.
      system: A system prompt to override any system-level instructions.
      raw: If True, bypasses any internal prompt templating; you provide the
        entire raw prompt.
      model_url: The base URL for the Ollama server. Defaults to self._model_url.
      timeout: Timeout (in seconds) for the HTTP request. Defaults to 120.
      keep_alive: How long (in seconds) the model remains loaded after
        generation completes.
      num_threads: Number of CPU threads to use. If None, Ollama uses a default
        heuristic.
      num_ctx: Number of context tokens allowed. If None, uses model's default
        or config.
      stop: Stop sequences to halt generation. Can be a string or list of strings.
      **kwargs: Additional parameters passed through.

    Returns:
      A mapping (dictionary-like) containing the server's JSON response. For
      non-streaming calls, the `"response"` key typically contains the entire
      generated text.

    Raises:
      InferenceConfigError: If the server returns a 404 (model not found).
      InferenceRuntimeError: For any other HTTP errors, timeouts, or request
        exceptions.
    """
    model = model or self._model
    model_url = model_url or self._model_url
    if structured_output_format is None and self.format_type is not None:
      structured_output_format = (
          'json' if self.format_type == core_types.FormatType.JSON else 'yaml'
      )

    options: dict[str, Any] = {}
    if keep_alive is not None:
      options['keep_alive'] = keep_alive
    else:
      options['keep_alive'] = _DEFAULT_KEEP_ALIVE

    if seed is not None:
      options['seed'] = seed
    if temperature is not None:
      options['temperature'] = temperature
    else:
      options['temperature'] = _DEFAULT_TEMPERATURE
    if top_k is not None:
      options['top_k'] = top_k
    if top_p is not None:
      options['top_p'] = top_p
    if num_threads is not None:
      options['num_thread'] = num_threads
    if max_output_tokens is not None:
      options['num_predict'] = max_output_tokens
    if num_ctx is not None:
      options['num_ctx'] = num_ctx
    else:
      options['num_ctx'] = _DEFAULT_NUM_CTX

    reserved_top_level = {
        'model',
        'prompt',
        'system',
        'stop',
        'format',
        'stream',
        'raw',
    }
    for key, value in kwargs.items():
      if value is None:
        continue
      if key in reserved_top_level:
        continue
      if key not in options:
        options[key] = value

    api_url = urljoin(
        model_url if model_url.endswith('/') else model_url + '/',
        'api/generate',
    )

    payload: dict[str, Any] = {
        'model': model,
        'prompt': prompt,
        'system': system,
        'stream': False,
        'raw': raw,
        'options': options,
    }

    if structured_output_format is not None:
      payload['format'] = structured_output_format

    if stop is not None:
      payload['stop'] = stop

    request_timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }

    if self._api_key:
      if self._auth_scheme:
        headers[self._auth_header] = f'{self._auth_scheme} {self._api_key}'
      else:
        headers[self._auth_header] = self._api_key

    try:
      response = self._requests.post(
          api_url,
          headers=headers,
          json=payload,
          timeout=request_timeout,
      )
    except self._requests.exceptions.RequestException as e:
      if isinstance(e, self._requests.exceptions.ReadTimeout):
        msg = (
            f'Ollama Model timed out (timeout={request_timeout},'
            f' num_threads={num_threads})'
        )
        raise exceptions.InferenceRuntimeError(
            msg, original=e, provider='Ollama'
        ) from e
      raise exceptions.InferenceRuntimeError(
          f'Ollama request failed: {str(e)}', original=e, provider='Ollama'
      ) from e

    response.encoding = 'utf-8'
    if response.status_code == 200:
      return response.json()
    if response.status_code == 404:
      raise exceptions.InferenceConfigError(
          f"Can't find Ollama {model}. Try: ollama run {model}"
      )
    else:
      msg = f'Bad status code from Ollama: {response.status_code}'
      raise exceptions.InferenceRuntimeError(msg, provider='Ollama')
