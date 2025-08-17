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

"""Core data types for LangExtract."""
from __future__ import annotations

import dataclasses
import enum
import textwrap

__all__ = ['ScoredOutput', 'FormatType', 'ConstraintType', 'Constraint']


class FormatType(enum.Enum):
  """Enumeration of prompt output formats."""

  YAML = 'yaml'
  JSON = 'json'


class ConstraintType(enum.Enum):
  """Enumeration of constraint types."""

  NONE = 'none'


@dataclasses.dataclass
class Constraint:
  """Represents a constraint for model output decoding.

  Attributes:
    constraint_type: The type of constraint applied.
  """

  constraint_type: ConstraintType = ConstraintType.NONE


@dataclasses.dataclass(frozen=True)
class ScoredOutput:
  """Scored output from language model inference."""

  score: float | None = None
  output: str | None = None

  def __str__(self) -> str:
    score_str = '-' if self.score is None else f'{self.score:.2f}'
    if self.output is None:
      return f'Score: {score_str}\nOutput: None'
    formatted_lines = textwrap.indent(self.output, prefix='  ')
    return f'Score: {score_str}\nOutput:\n{formatted_lines}'
