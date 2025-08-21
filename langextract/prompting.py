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
import yaml

from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema


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
  format_type: data.FormatType = data.FormatType.YAML
  attribute_suffix: str = "_attributes"
  examples_heading: str = "Examples"
  question_prefix: str = "Q: "
  answer_prefix: str = "A: "
  fence_output: bool = True  # whether to wrap answers in ```json/```yaml fences

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

    # Build a dictionary for serialization
    data_dict: dict[str, list] = {schema.EXTRACTIONS_KEY: []}
    for extraction in example.extractions:
      data_entry = {
          f"{extraction.extraction_class}": extraction.extraction_text,
          f"{extraction.extraction_class}{self.attribute_suffix}": (
              extraction.attributes or {}
          ),
      }
      data_dict[schema.EXTRACTIONS_KEY].append(data_entry)

    if self.format_type == data.FormatType.YAML:
      formatted_content = yaml.dump(
          data_dict, default_flow_style=False, sort_keys=False
      )
      if self.fence_output:
        answer = f"```yaml\n{formatted_content.strip()}\n```"
      else:
        answer = formatted_content.strip()
    elif self.format_type == data.FormatType.JSON:
      formatted_content = json.dumps(data_dict, indent=2, ensure_ascii=False)
      if self.fence_output:
        answer = f"```json\n{formatted_content.strip()}\n```"
      else:
        answer = formatted_content.strip()
    else:
      raise ValueError(f"Unsupported format type: {self.format_type}")

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
