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

"""Tests for prompt validation module."""

from absl.testing import absltest
from absl.testing import parameterized

from langextract import extraction
from langextract import prompt_validation
from langextract.core import data


class PromptAlignmentValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="exact_alignment",
          text="Patient takes lisinopril.",
          extraction_class="Medication",
          extraction_text="lisinopril",
          expected_issues=0,
          expected_has_failed=False,
          expected_has_non_exact=False,
          expected_alignment_status=None,
      ),
      dict(
          testcase_name="fuzzy_match_lesser",
          text="Type 2 diabetes.",
          extraction_class="Diagnosis",
          extraction_text="type-2 diabetes",
          expected_issues=1,
          expected_has_failed=False,
          expected_has_non_exact=True,
          expected_alignment_status=data.AlignmentStatus.MATCH_LESSER,
      ),
      dict(
          testcase_name="extraction_not_found",
          text="No medications mentioned in this text.",
          extraction_class="Medication",
          extraction_text="lisinopril",
          expected_issues=1,
          expected_has_failed=True,
          expected_has_non_exact=False,
          expected_alignment_status=None,
      ),
  )
  def test_alignment_detection(
      self,
      text,
      extraction_class,
      extraction_text,
      expected_issues,
      expected_has_failed,
      expected_has_non_exact,
      expected_alignment_status,
  ):
    """Test that different alignment types are correctly detected."""
    example = data.ExampleData(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class=extraction_class,
                extraction_text=extraction_text,
                attributes={},
            )
        ],
    )

    report = prompt_validation.validate_prompt_alignment([example])

    self.assertLen(report.issues, expected_issues)
    self.assertEqual(report.has_failed, expected_has_failed)
    self.assertEqual(report.has_non_exact, expected_has_non_exact)

    if expected_issues > 0:
      issue = report.issues[0]
      self.assertEqual(issue.alignment_status, expected_alignment_status)
      self.assertEqual(issue.extraction_class, extraction_class)
      if expected_has_failed:
        self.assertIsNone(issue.alignment_status)
      elif expected_has_non_exact:
        self.assertIsNotNone(issue.alignment_status)

  @parameterized.named_parameters(
      dict(
          testcase_name="one_fails",
          text="Patient takes lisinopril and has diabetes mellitus.",
          extractions=[
              ("Medication", "lisinopril"),  # PASSES - found exactly
              ("Diagnosis", "diabetes"),  # PASSES - found exactly
              ("Medication", "metformin"),  # FAILS - not in text
          ],
          expected_issues=1,
          expected_has_failed=True,
          expected_has_non_exact=False,
          expected_failed_text="metformin",
      ),
      dict(
          testcase_name="all_pass",
          text="Patient takes lisinopril and aspirin for diabetes management.",
          extractions=[
              ("Medication", "lisinopril"),
              ("Medication", "aspirin"),
              ("Diagnosis", "diabetes"),
          ],
          expected_issues=0,
          expected_has_failed=False,
          expected_has_non_exact=False,
          expected_failed_text=None,
      ),
  )
  def test_multiple_extractions_per_example(
      self,
      text,
      extractions,
      expected_issues,
      expected_has_failed,
      expected_has_non_exact,
      expected_failed_text,
  ):
    """Test validation with multiple extractions in a single example."""
    example = data.ExampleData(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class=extraction_class,
                extraction_text=extraction_text,
                attributes={},
            )
            for extraction_class, extraction_text in extractions
        ],
    )

    report = prompt_validation.validate_prompt_alignment([example])

    self.assertLen(report.issues, expected_issues)
    self.assertEqual(report.has_failed, expected_has_failed)
    self.assertEqual(report.has_non_exact, expected_has_non_exact)

    if expected_failed_text:
      issue = report.issues[0]
      self.assertIsNone(issue.alignment_status)
      self.assertEqual(issue.extraction_text_preview, expected_failed_text)

  @parameterized.named_parameters(
      dict(
          testcase_name="warning_mode_with_failed",
          text="Patient has no known allergies.",
          extraction_text="penicillin",
          validation_level=prompt_validation.PromptValidationLevel.WARNING,
          strict_non_exact=False,
      ),
      dict(
          testcase_name="off_mode_with_failed",
          text="Patient history incomplete.",
          extraction_text="aspirin",
          validation_level=prompt_validation.PromptValidationLevel.OFF,
          strict_non_exact=False,
      ),
  )
  def test_validation_levels_that_dont_raise(
      self, text, extraction_text, validation_level, strict_non_exact
  ):
    """Test that WARNING and OFF modes don't raise exceptions."""
    example = data.ExampleData(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class="Medication",
                extraction_text=extraction_text,
                attributes={},
            )
        ],
    )

    report = prompt_validation.validate_prompt_alignment([example])

    # This should not raise an exception in WARNING or OFF modes
    prompt_validation.handle_alignment_report(
        report, validation_level, strict_non_exact=strict_non_exact
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="error_mode_failed_alignment",
          text="Patient has no known allergies.",
          extraction_class="Medication",
          extraction_text="penicillin",
          strict_non_exact=False,
          error_pattern=r"1 extraction\(s\).*could not be aligned",
      ),
      dict(
          testcase_name="error_mode_strict_fuzzy_match",
          text="Type 2 diabetes.",
          extraction_class="Diagnosis",
          extraction_text="type-2 diabetes",
          strict_non_exact=True,
          error_pattern=r"strict mode.*1 non-exact",
      ),
  )
  def test_error_mode_raises_appropriately(
      self,
      text,
      extraction_class,
      extraction_text,
      strict_non_exact,
      error_pattern,
  ):
    """Test that ERROR mode raises with appropriate messages."""
    example = data.ExampleData(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class=extraction_class,
                extraction_text=extraction_text,
                attributes={},
            )
        ],
    )

    report = prompt_validation.validate_prompt_alignment([example])

    with self.assertRaisesRegex(
        prompt_validation.PromptAlignmentError, error_pattern
    ):
      prompt_validation.handle_alignment_report(
          report,
          prompt_validation.PromptValidationLevel.ERROR,
          strict_non_exact=strict_non_exact,
      )

  def test_empty_examples_produces_empty_report(self):
    report = prompt_validation.validate_prompt_alignment([])

    self.assertEmpty(report.issues)
    self.assertFalse(report.has_failed)
    self.assertFalse(report.has_non_exact)

  def test_multiple_examples_preserve_indices(self):
    examples = [
        data.ExampleData(  # Example 0: FAILS - "metformin" not in text
            text="First patient record.",
            extractions=[
                data.Extraction(
                    extraction_class="Medication",
                    extraction_text="metformin",
                    attributes={},
                )
            ],
        ),
        data.ExampleData(  # Example 1: PASSES - "aspirin" found exactly
            text="Patient takes aspirin daily.",
            extractions=[
                data.Extraction(
                    extraction_class="Medication",
                    extraction_text="aspirin",
                    attributes={},
                )
            ],
        ),
        data.ExampleData(  # Example 2: NON-EXACT - "type-2" fuzzy matches "Type 2"
            text="Type 2 diabetes mellitus.",
            extractions=[
                data.Extraction(
                    extraction_class="Diagnosis",
                    extraction_text="type-2 diabetes",
                    attributes={},
                )
            ],
        ),
    ]

    report = prompt_validation.validate_prompt_alignment(examples)

    # Expect 2 issues: example 0 (failed) and example 2 (non-exact)
    self.assertLen(report.issues, 2)
    self.assertTrue(report.has_failed)
    self.assertTrue(report.has_non_exact)

    issue_by_index = {issue.example_index: issue for issue in report.issues}

    # Example 0: Failed alignment (metformin not found)
    self.assertIn(0, issue_by_index)
    self.assertIsNone(issue_by_index[0].alignment_status)

    # Example 1: No issue (aspirin found exactly)
    self.assertNotIn(1, issue_by_index)

    # Example 2: Non-exact match (type-2 vs Type 2)
    self.assertIn(2, issue_by_index)
    self.assertIsNotNone(issue_by_index[2].alignment_status)

  def test_validation_does_not_mutate_input(self):
    example = data.ExampleData(
        text="Patient takes lisinopril 10mg daily.",
        extractions=[
            data.Extraction(
                extraction_class="Medication",
                extraction_text="lisinopril",
                attributes={},
            )
        ],
    )

    original_extraction = example.extractions[0]

    self.assertIsNone(getattr(original_extraction, "token_interval", None))
    self.assertIsNone(getattr(original_extraction, "char_interval", None))
    self.assertIsNone(getattr(original_extraction, "alignment_status", None))

    _ = prompt_validation.validate_prompt_alignment([example])

    self.assertIsNone(getattr(original_extraction, "token_interval", None))
    self.assertIsNone(getattr(original_extraction, "char_interval", None))
    self.assertIsNone(getattr(original_extraction, "alignment_status", None))

  @parameterized.named_parameters(
      dict(
          testcase_name="fuzzy_disabled_rejects_non_exact",
          text="Patient has type 2 diabetes.",
          extraction_class="Diagnosis",
          extraction_text="Type-2 Diabetes",
          enable_fuzzy=False,
          accept_lesser=False,
          fuzzy_threshold=0.75,
          expected_has_failed=True,
          expected_has_non_exact=False,
      ),
      dict(
          testcase_name="fuzzy_enabled_accepts_close_match",
          text="Patient has type 2 diabetes.",
          extraction_class="Diagnosis",
          extraction_text="Type-2 Diabetes",
          enable_fuzzy=True,
          accept_lesser=False,
          fuzzy_threshold=0.75,
          expected_has_failed=False,
          expected_has_non_exact=True,
      ),
  )
  def test_alignment_policies(
      self,
      text,
      extraction_class,
      extraction_text,
      enable_fuzzy,
      accept_lesser,
      fuzzy_threshold,
      expected_has_failed,
      expected_has_non_exact,
  ):
    """Test different alignment policy configurations."""
    example = data.ExampleData(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class=extraction_class,
                extraction_text=extraction_text,
                attributes={},
            )
        ],
    )

    if not enable_fuzzy:
      default_report = prompt_validation.validate_prompt_alignment([example])
      self.assertFalse(default_report.has_failed)
      self.assertTrue(default_report.has_non_exact)

    policy = prompt_validation.AlignmentPolicy(
        enable_fuzzy_alignment=enable_fuzzy,
        accept_match_lesser=accept_lesser,
        fuzzy_alignment_threshold=fuzzy_threshold,
    )
    report = prompt_validation.validate_prompt_alignment(
        [example], policy=policy
    )

    self.assertEqual(report.has_failed, expected_has_failed)
    self.assertEqual(report.has_non_exact, expected_has_non_exact)


class ExtractIntegrationTest(absltest.TestCase):
  """Minimal integration test for extract() entry point validation."""

  def test_extract_validates_in_error_mode(self):
    """Verify extract() runs validation when configured."""
    examples = [
        data.ExampleData(
            text="Patient takes aspirin.",
            extractions=[
                data.Extraction(
                    extraction_class="Medication",
                    extraction_text="ibuprofen",
                    attributes={},
                )
            ],
        )
    ]

    with self.assertRaisesRegex(
        prompt_validation.PromptAlignmentError,
        r"1 extraction\(s\).*could not be aligned",
    ):
      extraction.extract(
          text_or_documents="Test document",
          prompt_description="Extract medications",
          examples=examples,
          prompt_validation_level=prompt_validation.PromptValidationLevel.ERROR,
          model_id="fake-model",
      )


if __name__ == "__main__":
  absltest.main()
