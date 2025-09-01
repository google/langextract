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

"""Prompt validation for alignment checks on few-shot examples."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import dataclasses
import enum

from absl import logging

from langextract import resolver
from langextract.core import data

__all__ = [
    "PromptValidationLevel",
    "ValidationIssue",
    "ValidationReport",
    "PromptAlignmentError",
    "AlignmentPolicy",
    "validate_prompt_alignment",
    "handle_alignment_report",
]


_FUZZY_ALIGNMENT_MIN_THRESHOLD = 0.75


class PromptValidationLevel(enum.Enum):
  """Validation levels for prompt alignment checks."""

  OFF = "off"
  WARNING = "warning"
  ERROR = "error"


class _IssueKind(enum.Enum):
  """Internal categorization of alignment issues."""

  FAILED = "failed"  # alignment_status is None
  NON_EXACT = "non_exact"  # MATCH_FUZZY or MATCH_LESSER


@dataclasses.dataclass(frozen=True)
class ValidationIssue:
  """Represents a single validation issue found during alignment."""

  example_index: int
  example_id: str | None
  extraction_class: str
  extraction_text_preview: str
  alignment_status: data.AlignmentStatus | None
  issue_kind: _IssueKind
  char_interval: tuple[int, int] | None = None
  token_interval: tuple[int, int] | None = None

  def short_msg(self) -> str:
    """Returns a concise message describing the issue."""
    ex_id = f" id={self.example_id}" if self.example_id else ""
    span = ""
    if self.char_interval:
      span = f" char_span={self.char_interval}"
    return (
        f"[example#{self.example_index}{ex_id}] "
        f"class='{self.extraction_class}' "
        f"status={self.alignment_status} "
        f"text='{self.extraction_text_preview}'{span}"
    )


@dataclasses.dataclass
class ValidationReport:
  """Collection of validation issues from prompt alignment checks."""

  issues: list[ValidationIssue]

  @property
  def has_failed(self) -> bool:
    """Returns True if any extraction failed to align."""
    return any(i.issue_kind is _IssueKind.FAILED for i in self.issues)

  @property
  def has_non_exact(self) -> bool:
    """Returns True if any extraction has non-exact alignment."""
    return any(i.issue_kind is _IssueKind.NON_EXACT for i in self.issues)


class PromptAlignmentError(RuntimeError):
  """Raised when prompt alignment validation fails under ERROR mode."""


@dataclasses.dataclass(frozen=True)
class AlignmentPolicy:
  """Configuration for alignment validation behavior."""

  enable_fuzzy_alignment: bool = True
  fuzzy_alignment_threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD
  accept_match_lesser: bool = True


def _preview(s: str, n: int = 120) -> str:
  """Creates a preview of text for logging, collapsing whitespace."""
  s = " ".join(s.split())  # Collapse whitespace for logs
  return s if len(s) <= n else s[: n - 1] + "…"


def validate_prompt_alignment(
    examples: Sequence[data.ExampleData],
    aligner: resolver.WordAligner | None = None,
    policy: AlignmentPolicy | None = None,
) -> ValidationReport:
  """Align extractions to their own example text and collect issues.

  Args:
    examples: The few-shot examples to validate.
    aligner: WordAligner instance to use (creates new if None).
    policy: Alignment configuration (uses defaults if None).

  Returns:
    ValidationReport containing any alignment issues found.
  """
  if not examples:
    return ValidationReport(issues=[])

  aligner = aligner or resolver.WordAligner()
  policy = policy or AlignmentPolicy()

  issues: list[ValidationIssue] = []

  for idx, ex in enumerate(examples):
    # Defensive copy so validation never mutates user examples.
    copied_extractions = [[copy.deepcopy(e) for e in ex.extractions]]
    aligned_groups = aligner.align_extractions(
        extraction_groups=copied_extractions,
        source_text=ex.text,
        token_offset=0,
        char_offset=0,
        enable_fuzzy_alignment=policy.enable_fuzzy_alignment,
        fuzzy_alignment_threshold=policy.fuzzy_alignment_threshold,
        accept_match_lesser=policy.accept_match_lesser,
    )

    for aligned in aligned_groups[0]:
      status = getattr(aligned, "alignment_status", None)
      char_interval = getattr(aligned, "char_interval", None)
      token_interval = getattr(aligned, "token_interval", None)
      klass = getattr(aligned, "extraction_class", "<unknown>")
      text = getattr(aligned, "extraction_text", "")

      if status is None:
        issues.append(
            ValidationIssue(
                example_index=idx,
                example_id=getattr(ex, "example_id", None),
                extraction_class=klass,
                extraction_text_preview=_preview(text),
                alignment_status=None,
                issue_kind=_IssueKind.FAILED,
                char_interval=None,
                token_interval=None,
            )
        )
      elif status in (
          data.AlignmentStatus.MATCH_FUZZY,
          data.AlignmentStatus.MATCH_LESSER,
      ):
        char_interval_tuple = None
        token_interval_tuple = None
        if char_interval:
          char_interval_tuple = (char_interval.start_pos, char_interval.end_pos)
        if token_interval:
          token_interval_tuple = (
              token_interval.start_index,
              token_interval.end_index,
          )

        issues.append(
            ValidationIssue(
                example_index=idx,
                example_id=getattr(ex, "example_id", None),
                extraction_class=klass,
                extraction_text_preview=_preview(text),
                alignment_status=status,
                issue_kind=_IssueKind.NON_EXACT,
                char_interval=char_interval_tuple,
                token_interval=token_interval_tuple,
            )
        )

  return ValidationReport(issues=issues)


def handle_alignment_report(
    report: ValidationReport,
    level: PromptValidationLevel,
    *,
    strict_non_exact: bool = False,
) -> None:
  """Log or raise based on validation level.

  Args:
    report: The validation report to handle.
    level: The validation level determining behavior.
    strict_non_exact: If True, treat non-exact matches as errors in ERROR mode.

  Raises:
    PromptAlignmentError: If validation fails in ERROR mode.
  """
  if level is PromptValidationLevel.OFF:
    return

  for issue in report.issues:
    if issue.issue_kind is _IssueKind.NON_EXACT:
      logging.warning(
          "Prompt alignment: non-exact match: %s", issue.short_msg()
      )
    else:
      logging.warning(
          "Prompt alignment: FAILED to align: %s", issue.short_msg()
      )

  if level is PromptValidationLevel.ERROR:
    failed = [i for i in report.issues if i.issue_kind is _IssueKind.FAILED]
    non_exact = [
        i for i in report.issues if i.issue_kind is _IssueKind.NON_EXACT
    ]

    if failed:
      sample = failed[0].short_msg()
      raise PromptAlignmentError(
          f"Prompt alignment validation failed: {len(failed)} extraction(s) "
          f"could not be aligned (e.g., {sample})"
      )
    if strict_non_exact and non_exact:
      sample = non_exact[0].short_msg()
      raise PromptAlignmentError(
          "Prompt alignment validation failed under strict mode: "
          f"{len(non_exact)} non-exact match(es) found (e.g., {sample})"
      )
