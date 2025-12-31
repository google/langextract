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

"""Library for building prompts."""
from __future__ import annotations

import dataclasses
import json
import pathlib

import pydantic
from typing_extensions import override
import yaml

from langextract.core import data
from langextract.core import exceptions
from langextract.core import format_handler


class PromptBuilderError(exceptions.LangExtractError):
  """Failure to build prompt."""


class ParseError(PromptBuilderError):
  """Prompt template cannot be parsed."""


@dataclasses.dataclass
class PromptTemplateStructured:
  """A structured prompt template for few-shot examples.

  Attributes:
    description: Instructions or guidelines for the LLM.
    examples: ExampleData objects demonstrating expected input→output behavior.
  """

  description: str
  examples: list[data.ExampleData] = dataclasses.field(default_factory=list)


def read_prompt_template_structured_from_file(
    prompt_path: str,
    format_type: data.FormatType = data.FormatType.YAML,
) -> PromptTemplateStructured:
  """Reads a structured prompt template from a file.

  Args:
    prompt_path: Path to a file containing PromptTemplateStructured data.
    format_type: The format of the file; YAML or JSON.

  Returns:
    A PromptTemplateStructured object loaded from the file.

  Raises:
    ParseError: If the file cannot be parsed successfully.
  """
  adapter = pydantic.TypeAdapter(PromptTemplateStructured)
  try:
    with pathlib.Path(prompt_path).open("rt") as f:
      data_dict = {}
      prompt_content = f.read()
      if format_type == data.FormatType.YAML:
        data_dict = yaml.safe_load(prompt_content)
      elif format_type == data.FormatType.JSON:
        data_dict = json.loads(prompt_content)
      return adapter.validate_python(data_dict)
  except Exception as e:
    raise ParseError(
        f"Failed to parse prompt template from file: {prompt_path}"
    ) from e


@dataclasses.dataclass
class QAPromptGenerator:
  """Generates question-answer prompts from the provided template."""

  template: PromptTemplateStructured
  format_handler: format_handler.FormatHandler
  examples_heading: str = "Examples"
  question_prefix: str = "Q: "
  answer_prefix: str = "A: "

  def __str__(self) -> str:
    """Returns a string representation of the prompt with an empty question."""
    return self.render("")

  def format_example_as_text(self, example: data.ExampleData) -> str:
    """Formats a single example for the prompt.

    Args:
      example: The example data to format.

    Returns:
      A string representation of the example, including the question and answer.
    """
    question = example.text
    answer = self.format_handler.format_extraction_example(example.extractions)

    return "\n".join([
        f"{self.question_prefix}{question}",
        f"{self.answer_prefix}{answer}\n",
    ])

  def render(self, question: str, additional_context: str | None = None) -> str:
    """Generate a text representation of the prompt.

    Args:
      question: That will be presented to the model.
      additional_context: Additional context to include in the prompt. An empty
        string is ignored.

    Returns:
      Text prompt with a question to be presented to a language model.
    """
    prompt_lines: list[str] = [f"{self.template.description}\n"]

    if additional_context:
      prompt_lines.append(f"{additional_context}\n")

    if self.template.examples:
      prompt_lines.append(self.examples_heading)
      for ex in self.template.examples:
        prompt_lines.append(self.format_example_as_text(ex))

    prompt_lines.append(f"{self.question_prefix}{question}")
    prompt_lines.append(self.answer_prefix)
    return "\n".join(prompt_lines)


class PromptBuilder:
  """Builds prompts for text chunks using a QAPromptGenerator.

  This base class provides a simple interface for prompt generation. Subclasses
  can extend this to add stateful behavior like cross-chunk context tracking.
  """

  def __init__(self, generator: QAPromptGenerator):
    """Initializes the builder with the given prompt generator.

    Args:
      generator: The underlying prompt generator to use.
    """
    self._generator = generator

  def build_prompt(
      self,
      chunk_text: str,
      document_id: str,
      additional_context: str | None = None,
  ) -> str:
    """Builds a prompt for the given chunk.

    Args:
      chunk_text: The text of the current chunk to process.
      document_id: Identifier for the source document.
      additional_context: Optional additional context from the document.

    Returns:
      The rendered prompt string ready for the language model.
    """
    del document_id  # Unused in base class.
    return self._generator.render(
        question=chunk_text,
        additional_context=additional_context,
    )


class ContextAwarePromptBuilder(PromptBuilder):
  """Prompt builder with cross-chunk context tracking.

  Extends PromptBuilder to inject text from the previous chunk into each
  prompt. This helps language models resolve coreferences across chunk
  boundaries (e.g., connecting "She" to "Dr. Sarah Johnson" from the
  previous chunk).

  Context is tracked per document_id, so multiple documents can be processed
  without context bleeding between them.
  """

  _CONTEXT_PREFIX = "[Previous text]: ..."

  def __init__(
      self,
      generator: QAPromptGenerator,
      context_window_chars: int | None = None,
  ):
    """Initializes the builder with context tracking configuration.

    Args:
      generator: The underlying prompt generator to use.
      context_window_chars: Number of characters from the previous chunk's
          tail to include as context. Defaults to None (disabled).
    """
    super().__init__(generator)
    self._context_window_chars = context_window_chars
    self._prev_chunk_by_doc_id: dict[str, str] = {}

  @property
  def context_window_chars(self) -> int | None:
    """Number of trailing characters from previous chunk to include."""
    return self._context_window_chars

  @override
  def build_prompt(
      self,
      chunk_text: str,
      document_id: str,
      additional_context: str | None = None,
  ) -> str:
    """Builds a prompt, injecting previous chunk context if enabled.

    Args:
      chunk_text: The text of the current chunk to process.
      document_id: Identifier for the source document (used to track context
          per document).
      additional_context: Optional additional context from the document.

    Returns:
      The rendered prompt string ready for the language model.
    """
    effective_context = self._build_effective_context(
        document_id, additional_context
    )
    prompt = self._generator.render(
        question=chunk_text,
        additional_context=effective_context,
    )
    self._update_state(document_id, chunk_text)
    return prompt

  def _build_effective_context(
      self,
      document_id: str,
      additional_context: str | None,
  ) -> str | None:
    """Combines previous chunk context with any additional context.

    Args:
      document_id: Identifier for the source document.
      additional_context: Optional additional context from the document.

    Returns:
      Combined context string, or None if no context is available.
    """
    context_parts: list[str] = []

    if self._context_window_chars and document_id in self._prev_chunk_by_doc_id:
      prev_text = self._prev_chunk_by_doc_id[document_id]
      window = prev_text[-self._context_window_chars :]
      context_parts.append(f"{self._CONTEXT_PREFIX}{window}")

    if additional_context:
      context_parts.append(additional_context)

    return "\n\n".join(context_parts) if context_parts else None

  def _update_state(self, document_id: str, chunk_text: str) -> None:
    """Stores current chunk as context for the next chunk in this document.

    Args:
      document_id: Identifier for the source document.
      chunk_text: The current chunk text to store.
    """
    if self._context_window_chars:
      self._prev_chunk_by_doc_id[document_id] = chunk_text
