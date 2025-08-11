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

"""LangExtract: A Python library for extracting structured and grounded information from text using LLMs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast, Type, TypeVar
import warnings

import dotenv

from langextract import annotation
from langextract import data
from langextract import exceptions
from langextract import factory
from langextract import inference
from langextract import io
from langextract import prompting
from langextract import providers
from langextract import resolver
from langextract import schema
from langextract import visualization

__all__ = [
    "extract",
    "visualize",
    "annotation",
    "data",
    "exceptions",
    "factory",
    "inference",
    "io",
    "prompting",
    "providers",
    "resolver",
    "schema",
    "visualization",
]

LanguageModelT = TypeVar("LanguageModelT", bound=inference.BaseLanguageModel)

# Set up visualization helper at the top level (lx.visualize).
visualize = visualization.visualize

# Load environment variables from .env file
# NOTE: This behavior will be changed to opt-in in v2.0.0
# Libraries typically should not auto-load .env files, but this is kept
# for backward compatibility. Users can set environment variables directly
# or use python-dotenv explicitly in their own code.
dotenv.load_dotenv()


def extract(
    text_or_documents: str | data.Document | Iterable[data.Document],
    prompt_description: str | None = None,
    examples: Sequence[data.ExampleData] | None = None,
    model_id: str = "gemini-2.5-flash",
    api_key: str | None = None,
    language_model_type: Type[LanguageModelT] = inference.GeminiLanguageModel,
    format_type: data.FormatType = data.FormatType.JSON,
    max_char_buffer: int = 1000,
    temperature: float = 0.5,
    fence_output: bool = False,
    use_schema_constraints: bool = True,
    batch_length: int = 10,
    max_workers: int = 10,
    additional_context: str | None = None,
    resolver_params: dict | None = None,
    language_model_params: dict | None = None,
    debug: bool = True,
    model_url: str | None = None,
    extraction_passes: int = 1,
) -> data.AnnotatedDocument | Iterable[data.AnnotatedDocument]:
  """Extracts structured information from text.

  Retrieves structured information from the provided text or documents using a
  language model based on the instructions in prompt_description and guided by
  examples. Supports sequential extraction passes to improve recall at the cost
  of additional API calls.

  Args:
      text_or_documents: The source text to extract information from, a URL to
        download text from (starting with http:// or https://), or an iterable
        of Document objects.
      prompt_description: Instructions for what to extract from the text.
      examples: List of ExampleData objects to guide the extraction.
      api_key: API key for Gemini or other LLM services (can also use
        environment variable LANGEXTRACT_API_KEY). Cost considerations: Most
        APIs charge by token volume. Smaller max_char_buffer values increase the
        number of API calls, while extraction_passes > 1 reprocesses tokens
        multiple times. Note that max_workers improves processing speed without
        additional token costs. Refer to your API provider's pricing details and
        monitor usage with small test runs to estimate costs.
      model_id: The model ID to use for extraction.
      language_model_type: The type of language model to use for inference.
      format_type: The format type for the output (JSON or YAML).
      max_char_buffer: Max number of characters for inference.
      temperature: The sampling temperature for generation. Higher values (e.g.,
        0.5) can improve performance with schema constraints on some models by
        reducing repetitive outputs. Defaults to 0.5.
      fence_output: Whether to expect/generate fenced output (```json or
        ```yaml). When True, the model is prompted to generate fenced output and
        the resolver expects it. When False, raw JSON/YAML is expected. If your
        model utilizes schema constraints, this can generally be set to False
        unless the constraint also accounts for code fence delimiters.
      use_schema_constraints: Whether to generate schema constraints for models.
        For supported models, this enables structured outputs. Defaults to True.
      batch_length: Number of text chunks processed per batch. Higher values
        enable greater parallelization when batch_length >= max_workers.
        Defaults to 10.
      max_workers: Maximum parallel workers for concurrent processing. Effective
        parallelization is limited by min(batch_length, max_workers). Supported
        by Gemini models. Defaults to 10.
      additional_context: Additional context to be added to the prompt during
        inference.
      resolver_params: Parameters for the `resolver.Resolver`, which parses the
        raw language model output string (e.g., extracting JSON from ```json ...
        ``` blocks) into structured `data.Extraction` objects. This dictionary
        overrides default settings. Keys include: - 'fence_output' (bool):
        Whether to expect fenced output. - 'extraction_index_suffix' (str |
        None): Suffix for keys indicating extraction order. Default is None
        (order by appearance). - 'extraction_attributes_suffix' (str | None):
        Suffix for keys containing extraction attributes. Default is
        "_attributes".
      language_model_params: Additional parameters for the language model.
      debug: Whether to populate debug fields.
      model_url: Endpoint URL for self-hosted or on-prem models. Only forwarded
        when the selected `language_model_type` accepts this argument.
      extraction_passes: Number of sequential extraction attempts to improve
        recall and find additional entities. Defaults to 1 (standard single
        extraction). When > 1, the system performs multiple independent
        extractions and merges non-overlapping results (first extraction wins
        for overlaps). WARNING: Each additional pass reprocesses tokens,
        potentially increasing API costs. For example, extraction_passes=3
        reprocesses tokens 3x.

  Returns:
      An AnnotatedDocument with the extracted information when input is a
      string or URL, or an iterable of AnnotatedDocuments when input is an
      iterable of Documents.

  Raises:
      ValueError: If examples is None or empty.
      ValueError: If no API key is provided or found in environment variables.
      requests.RequestException: If URL download fails.
  """
  if not examples:
    raise ValueError(
        "Examples are required for reliable extraction. Please provide at least"
        " one ExampleData object with sample extractions."
    )

  if use_schema_constraints and fence_output:
    warnings.warn(
        "When `use_schema_constraints` is True and `fence_output` is True, "
        "ensure that your schema constraint includes the code fence "
        "delimiters, or set `fence_output` to False.",
        UserWarning,
    )

  if max_workers is not None and batch_length < max_workers:
    warnings.warn(
        f"batch_length ({batch_length}) is less than max_workers"
        f" ({max_workers}). Only {batch_length} workers will be used. For"
        " optimal parallelization, set batch_length >= max_workers.",
        UserWarning,
    )

  if isinstance(text_or_documents, str) and io.is_url(text_or_documents):
    text_or_documents = io.download_text_from_url(text_or_documents)

  prompt_template = prompting.PromptTemplateStructured(
      description=prompt_description
  )
  prompt_template.examples.extend(examples)

  # Generate schema constraints if enabled
  model_schema = None
  schema_constraint = None

  # TODO: Unify schema generation.
  if (
      use_schema_constraints
      and language_model_type == inference.GeminiLanguageModel
  ):
    model_schema = schema.GeminiSchema.from_examples(prompt_template.examples)

  # Handle backward compatibility for language_model_type parameter
  if language_model_type != inference.GeminiLanguageModel:
    warnings.warn(
        "The 'language_model_type' parameter is deprecated and will be removed"
        " in a future version. The provider is now automatically selected based"
        " on the model_id.",
        DeprecationWarning,
        stacklevel=2,
    )

  # Use factory to create the language model
  base_lm_kwargs: dict[str, Any] = {
      "api_key": api_key,
      "gemini_schema": model_schema,
      "format_type": format_type,
      "temperature": temperature,
      "model_url": model_url,
      "base_url": model_url,  # Support both parameter names for Ollama
      "constraint": schema_constraint,
      "max_workers": max_workers,
  }

  # Merge user-provided params which have precedence over defaults.
  base_lm_kwargs.update(language_model_params or {})

  # Filter out None values
  filtered_kwargs = {k: v for k, v in base_lm_kwargs.items() if v is not None}

  # Create model using factory
  # Providers are loaded lazily by the registry on first resolve
  config = factory.ModelConfig(
      model_id=model_id, provider_kwargs=filtered_kwargs
  )
  language_model = factory.create_model(config)

  resolver_defaults = {
      "fence_output": fence_output,
      "format_type": format_type,
      "extraction_attributes_suffix": "_attributes",
      "extraction_index_suffix": None,
  }
  resolver_defaults.update(resolver_params or {})

  res = resolver.Resolver(**resolver_defaults)

  annotator = annotation.Annotator(
      language_model=language_model,
      prompt_template=prompt_template,
      format_type=format_type,
      fence_output=fence_output,
  )

  if isinstance(text_or_documents, str):
    return annotator.annotate_text(
        text=text_or_documents,
        resolver=res,
        max_char_buffer=max_char_buffer,
        batch_length=batch_length,
        additional_context=additional_context,
        debug=debug,
        extraction_passes=extraction_passes,
    )
  else:
    documents = cast(Iterable[data.Document], text_or_documents)
    return annotator.annotate_documents(
        documents=documents,
        resolver=res,
        max_char_buffer=max_char_buffer,
        batch_length=batch_length,
        debug=debug,
        extraction_passes=extraction_passes,
    )
