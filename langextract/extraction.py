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

"""Main extraction API for LangExtract."""

from __future__ import annotations

from collections.abc import Iterable
import typing
from typing import cast
import warnings

from langextract import annotation
from langextract import factory
from langextract import io
from langextract import prompting
from langextract import resolver
from langextract.core import data


def extract(
    text_or_documents: typing.Any,
    prompt_description: str | None = None,
    examples: typing.Sequence[typing.Any] | None = None,
    model_id: str = "gemini-2.5-flash",
    api_key: str | None = None,
    language_model_type: typing.Type[typing.Any] | None = None,
    format_type: typing.Any = None,
    max_char_buffer: int = 1000,
    temperature: float | None = None,
    fence_output: bool | None = None,
    use_schema_constraints: bool = True,
    batch_length: int = 10,
    max_workers: int = 10,
    additional_context: str | None = None,
    resolver_params: dict | None = None,
    language_model_params: dict | None = None,
    debug: bool = True,
    model_url: str | None = None,
    extraction_passes: int = 1,
    config: typing.Any = None,
    model: typing.Any = None,
    *,
    fetch_urls: bool = True,
) -> typing.Any:
  """Extracts structured information from text.

  Retrieves structured information from the provided text or documents using a
  language model based on the instructions in prompt_description and guided by
  examples. Supports sequential extraction passes to improve recall at the cost
  of additional API calls.

  Args:
      text_or_documents: The source text to extract information from, a URL to
        download text from (starting with http:// or https:// when fetch_urls
        is True), or an iterable of Document objects.
      prompt_description: Instructions for what to extract from the text.
      examples: List of ExampleData objects to guide the extraction.
      api_key: API key for Gemini or other LLM services (can also use
        environment variable LANGEXTRACT_API_KEY). Cost considerations: Most
        APIs charge by token volume. Smaller max_char_buffer values increase the
        number of API calls, while extraction_passes > 1 reprocesses tokens
        multiple times. Note that max_workers improves processing speed without
        additional token costs. Refer to your API provider's pricing details and
        monitor usage with small test runs to estimate costs.
      model_id: The model ID to use for extraction (e.g., 'gemini-2.5-flash').
        If your model ID is not recognized or you need to use a custom provider,
        use the 'config' parameter with factory.ModelConfig to specify the
        provider explicitly.
      language_model_type: [DEPRECATED] The type of language model to use for
        inference. Warning triggers when value differs from the legacy default
        (GeminiLanguageModel). This parameter will be removed in v2.0.0. Use
        the model, config, or model_id parameters instead.
      format_type: The format type for the output (JSON or YAML).
      max_char_buffer: Max number of characters for inference.
      temperature: The sampling temperature for generation. When None (default),
        uses the model's default temperature. Set to 0.0 for deterministic output
        or higher values for more variation.
      fence_output: Whether to expect/generate fenced output (```json or
        ```yaml). When True, the model is prompted to generate fenced output and
        the resolver expects it. When False, raw JSON/YAML is expected. When None,
        automatically determined based on provider schema capabilities: if a schema
        is applied and supports_strict_mode is True, defaults to False; otherwise
        True. If your model utilizes schema constraints, this can generally be set
        to False unless the constraint also accounts for code fence delimiters.
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
      debug: Whether to enable debug logging. When True, enables detailed logging
        of function calls, arguments, return values, and timing for the langextract
        namespace. Note: Debug logging remains enabled for the process once activated.
      model_url: Endpoint URL for self-hosted or on-prem models. Only forwarded
        when the selected `language_model_type` accepts this argument.
      extraction_passes: Number of sequential extraction attempts to improve
        recall and find additional entities. Defaults to 1 (standard single
        extraction). When > 1, the system performs multiple independent
        extractions and merges non-overlapping results (first extraction wins
        for overlaps). WARNING: Each additional pass reprocesses tokens,
        potentially increasing API costs. For example, extraction_passes=3
        reprocesses tokens 3x.
      config: Model configuration to use for extraction. Takes precedence over
        model_id, api_key, and language_model_type parameters. When both model
        and config are provided, model takes precedence.
      model: Pre-configured language model to use for extraction. Takes
        precedence over all other parameters including config.
      fetch_urls: Whether to automatically download content when the input is a
        URL string. When True (default), strings starting with http:// or
        https:// are fetched. When False, all strings are treated as literal
        text to analyze. This is a keyword-only parameter.

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

  if debug:
    # pylint: disable=import-outside-toplevel
    from langextract.core import debug_utils

    debug_utils.configure_debug_logging()

  if format_type is None:
    format_type = data.FormatType.JSON

  if max_workers is not None and batch_length < max_workers:
    warnings.warn(
        f"batch_length ({batch_length}) < max_workers ({max_workers}). "
        f"Only {batch_length} workers will be used. "
        "Set batch_length >= max_workers for optimal parallelization.",
        UserWarning,
    )

  if (
      fetch_urls
      and isinstance(text_or_documents, str)
      and io.is_url(text_or_documents)
  ):
    text_or_documents = io.download_text_from_url(text_or_documents)

  prompt_template = prompting.PromptTemplateStructured(
      description=prompt_description
  )
  prompt_template.examples.extend(examples)

  language_model = None

  if model:
    language_model = model
    if fence_output is not None:
      language_model.set_fence_output(fence_output)
    if use_schema_constraints:
      warnings.warn(
          "'use_schema_constraints' is ignored when 'model' is provided. "
          "The model should already be configured with schema constraints.",
          UserWarning,
          stacklevel=2,
      )
  elif config:
    if use_schema_constraints:
      warnings.warn(
          "With 'config', schema constraints are still applied via examples. "
          "Or pass explicit schema in config.provider_kwargs.",
          UserWarning,
          stacklevel=2,
      )

    language_model = factory.create_model(
        config=config,
        examples=prompt_template.examples if use_schema_constraints else None,
        use_schema_constraints=use_schema_constraints,
        fence_output=fence_output,
    )
  else:
    if language_model_type is not None:
      warnings.warn(
          "'language_model_type' is deprecated and will be removed in v2.0.0. "
          "Use model, config, or model_id parameters instead.",
          FutureWarning,
          stacklevel=2,
      )

    base_lm_kwargs: dict[str, typing.Any] = {
        "api_key": api_key,
        "format_type": format_type,
        "temperature": temperature,
        "model_url": model_url,
        "base_url": model_url,
        "max_workers": max_workers,
    }

    # TODO(v2.0.0): Remove gemini_schema parameter
    if "gemini_schema" in (language_model_params or {}):
      warnings.warn(
          "'gemini_schema' is deprecated. Schema constraints are now "
          "automatically handled. This parameter will be ignored.",
          FutureWarning,
          stacklevel=2,
      )
      language_model_params = dict(language_model_params or {})
      language_model_params.pop("gemini_schema", None)

    base_lm_kwargs.update(language_model_params or {})
    filtered_kwargs = {k: v for k, v in base_lm_kwargs.items() if v is not None}

    config = factory.ModelConfig(
        model_id=model_id, provider_kwargs=filtered_kwargs
    )

    language_model = factory.create_model(
        config=config,
        examples=prompt_template.examples if use_schema_constraints else None,
        use_schema_constraints=use_schema_constraints,
        fence_output=fence_output,
    )

  fence_output = language_model.requires_fence_output

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
        max_workers=max_workers,
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
        max_workers=max_workers,
    )
