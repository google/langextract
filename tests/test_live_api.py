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

"""Live API integration tests that require real API keys.

These tests are skipped if API keys are not available in the environment.
They should run in CI after all other tests pass.
"""

from functools import wraps
import os
import re
import textwrap
import time
import unittest

from dotenv import load_dotenv
import pytest

import langextract as lx
from langextract.inference import OpenAILanguageModel

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get(
    "LANGEXTRACT_API_KEY"
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

skip_if_no_gemini = pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason=(
        "Gemini API key not available (set GEMINI_API_KEY or"
        " LANGEXTRACT_API_KEY)"
    ),
)
skip_if_no_openai = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OpenAI API key not available (set OPENAI_API_KEY)",
)

live_api = pytest.mark.live_api

GEMINI_MODEL_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.0,
    "max_output_tokens": 256,
}

OPENAI_MODEL_PARAMS = {
    "temperature": 0.0,
}


INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 8.0


def retry_on_transient_errors(max_retries=3, backoff_factor=2.0):
  """Decorator to retry tests on transient API errors with exponential backoff.

  Args:
    max_retries: Maximum number of retry attempts
    backoff_factor: Multiplier for exponential backoff (e.g., 2.0 = 1s, 2s, 4s)
  """

  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      last_exception = None
      delay = INITIAL_RETRY_DELAY

      for attempt in range(max_retries + 1):
        try:
          return func(*args, **kwargs)
        except (
            lx.exceptions.LangExtractError,
            ConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
        ) as e:
          last_exception = e
          error_str = str(e).lower()
          error_type = type(e).__name__

          transient_errors = [
              "503",
              "service unavailable",
              "temporarily unavailable",
              "rate limit",
              "429",
              "too many requests",
              "connection reset",
              "timeout",
              "deadline exceeded",
          ]

          is_transient = any(
              err in error_str for err in transient_errors
          ) or error_type in ["ServiceUnavailable", "RateLimitError", "Timeout"]

          if is_transient and attempt < max_retries:
            print(
                f"\nTransient error ({error_type}) on attempt"
                f" {attempt + 1}/{max_retries + 1}: {e}"
            )
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * backoff_factor, MAX_RETRY_DELAY)
          else:
            raise

      raise last_exception

    return wrapper

  return decorator


@pytest.fixture(autouse=True)
def add_delay_between_tests():
  """Add a small delay between tests to avoid rate limiting."""
  yield
  time.sleep(0.5)


def get_basic_medication_examples():
  """Get example data for basic medication extraction."""
  return [
      lx.data.ExampleData(
          text="Patient was given 250 mg IV Cefazolin TID for one week.",
          extractions=[
              lx.data.Extraction(
                  extraction_class="dosage", extraction_text="250 mg"
              ),
              lx.data.Extraction(
                  extraction_class="route", extraction_text="IV"
              ),
              lx.data.Extraction(
                  extraction_class="medication", extraction_text="Cefazolin"
              ),
              lx.data.Extraction(
                  extraction_class="frequency",
                  extraction_text="TID",  # TID = three times a day
              ),
              lx.data.Extraction(
                  extraction_class="duration", extraction_text="for one week"
              ),
          ],
      )
  ]


def get_relationship_examples():
  """Get example data for medication relationship extraction."""
  return [
      lx.data.ExampleData(
          text=(
              "Patient takes Aspirin 100mg daily for heart health and"
              " Simvastatin 20mg at bedtime."
          ),
          extractions=[
              # First medication group
              lx.data.Extraction(
                  extraction_class="medication",
                  extraction_text="Aspirin",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class="dosage",
                  extraction_text="100mg",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class="frequency",
                  extraction_text="daily",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class="condition",
                  extraction_text="heart health",
                  attributes={"medication_group": "Aspirin"},
              ),
              # Second medication group
              lx.data.Extraction(
                  extraction_class="medication",
                  extraction_text="Simvastatin",
                  attributes={"medication_group": "Simvastatin"},
              ),
              lx.data.Extraction(
                  extraction_class="dosage",
                  extraction_text="20mg",
                  attributes={"medication_group": "Simvastatin"},
              ),
              lx.data.Extraction(
                  extraction_class="frequency",
                  extraction_text="at bedtime",
                  attributes={"medication_group": "Simvastatin"},
              ),
          ],
      )
  ]


def extract_by_class(result, extraction_class):
  """Helper to extract entities by class.

  Returns a set of extraction texts for the given class.
  """
  return {
      e.extraction_text
      for e in result.extractions
      if e.extraction_class == extraction_class
  }


def assert_extractions_contain(test_case, result, expected_classes):
  """Assert that result contains all expected extraction classes.

  Uses unittest assertions for richer error messages.
  """
  actual_classes = {e.extraction_class for e in result.extractions}
  missing_classes = expected_classes - actual_classes
  test_case.assertFalse(
      missing_classes,
      f"Missing expected classes: {missing_classes}. Found extractions:"
      f" {[f'{e.extraction_class}:{e.extraction_text}' for e in result.extractions]}",
  )


def assert_valid_char_intervals(test_case, result):
  """Assert that all extractions have valid char intervals and alignment status."""
  for extraction in result.extractions:
    test_case.assertIsNotNone(
        extraction.char_interval,
        f"Missing char_interval for extraction: {extraction.extraction_text}",
    )
    test_case.assertIsNotNone(
        extraction.alignment_status,
        "Missing alignment_status for extraction:"
        f" {extraction.extraction_text}",
    )
    if hasattr(result, "text") and result.text:
      text_length = len(result.text)
      test_case.assertGreaterEqual(
          extraction.char_interval.start_pos,
          0,
          f"Invalid start_pos for extraction: {extraction.extraction_text}",
      )
      test_case.assertLessEqual(
          extraction.char_interval.end_pos,
          text_length,
          f"Invalid end_pos for extraction: {extraction.extraction_text}",
      )


class TestLiveAPIGemini(unittest.TestCase):
  """Tests using real Gemini API."""

  @skip_if_no_gemini
  @live_api
  @retry_on_transient_errors(max_retries=2)
  def test_medication_extraction(self):
    """Test medication extraction with entities in order."""
    prompt = textwrap.dedent("""\
        Extract medication information including medication name, dosage, route, frequency,
        and duration in the order they appear in the text.""")

    examples = get_basic_medication_examples()
    input_text = "Patient took 400 mg PO Ibuprofen q4h for two days."

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id=DEFAULT_GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        language_model_params=GEMINI_MODEL_PARAMS,
    )

    assert result is not None
    assert hasattr(result, "extractions")
    assert len(result.extractions) > 0

    expected_classes = {
        "dosage",
        "route",
        "medication",
        "frequency",
        "duration",
    }
    assert_extractions_contain(self, result, expected_classes)
    assert_valid_char_intervals(self, result)

    # Using regex for precise matching to avoid false positives
    medication_texts = extract_by_class(result, "medication")
    self.assertTrue(
        any(
            re.search(r"\bIbuprofen\b", text, re.IGNORECASE)
            for text in medication_texts
        ),
        f"No Ibuprofen found in: {medication_texts}",
    )

    dosage_texts = extract_by_class(result, "dosage")
    self.assertTrue(
        any(
            re.search(r"\b400\s*mg\b", text, re.IGNORECASE)
            for text in dosage_texts
        ),
        f"No 400mg dosage found in: {dosage_texts}",
    )

    route_texts = extract_by_class(result, "route")
    self.assertTrue(
        any(
            re.search(r"\b(PO|oral)\b", text, re.IGNORECASE)
            for text in route_texts
        ),
        f"No PO/oral route found in: {route_texts}",
    )

  @skip_if_no_gemini
  @live_api
  @retry_on_transient_errors(max_retries=2)
  @pytest.mark.xfail(
      reason=(
          "Known tokenizer issue with non-Latin characters - see GitHub"
          " issue #13"
      ),
      strict=True,
  )
  def test_multilingual_medication_extraction(self):
    """Test medication extraction with Japanese text."""
    text = (  # "The patient takes 10 mg of medication daily."
        "患者は毎日10mgの薬を服用します。"
    )

    prompt = "Extract medication information including dosage and frequency."

    examples = [
        lx.data.ExampleData(
            text="The patient takes 20mg of aspirin twice daily.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="medication",
                    extraction_text="aspirin",
                    attributes={"dosage": "20mg", "frequency": "twice daily"},
                ),
            ],
        )
    ]

    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=DEFAULT_GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        language_model_params=GEMINI_MODEL_PARAMS,
    )

    assert result is not None
    assert hasattr(result, "extractions")
    assert len(result.extractions) > 0

    medication_extractions = [
        e for e in result.extractions if e.extraction_class == "medication"
    ]
    assert (
        len(medication_extractions) > 0
    ), "No medication entities found in Japanese text"
    assert_valid_char_intervals(self, result)

  @skip_if_no_gemini
  @live_api
  @retry_on_transient_errors(max_retries=2)
  def test_medication_relationship_extraction(self):
    """Test relationship extraction for medications with Gemini."""
    input_text = """
    The patient was prescribed Lisinopril and Metformin last month.
    He takes the Lisinopril 10mg daily for hypertension, but often misses
    his Metformin 500mg dose which should be taken twice daily for diabetes.
    """

    prompt = textwrap.dedent("""
        Extract medications with their details, using attributes to group related information:

        1. Extract entities in the order they appear in the text
        2. Each entity must have a 'medication_group' attribute linking it to its medication
        3. All details about a medication should share the same medication_group value
    """)

    examples = get_relationship_examples()

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id=DEFAULT_GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        language_model_params=GEMINI_MODEL_PARAMS,
    )

    assert result is not None
    assert len(result.extractions) > 0
    assert_valid_char_intervals(self, result)

    medication_groups = {}
    for extraction in result.extractions:
      assert (
          extraction.attributes is not None
      ), f"Missing attributes for {extraction.extraction_text}"
      assert (
          "medication_group" in extraction.attributes
      ), f"Missing medication_group for {extraction.extraction_text}"

      group_name = extraction.attributes["medication_group"]
      medication_groups.setdefault(group_name, []).append(extraction)

    assert (
        len(medication_groups) >= 2
    ), f"Expected at least 2 medications, found {len(medication_groups)}"

    # Allow flexible matching for dosage field (could be "dosage" or "dose")
    for med_name, extractions in medication_groups.items():
      extraction_classes = {e.extraction_class for e in extractions}
      # At minimum, each group should have the medication itself
      assert (
          "medication" in extraction_classes
      ), f"{med_name} group missing medication entity"
      # Dosage is expected but might be formatted differently
      assert any(
          c in extraction_classes for c in ["dosage", "dose"]
      ), f"{med_name} group missing dosage"


class TestLiveAPIOpenAI(unittest.TestCase):
  """Tests using real OpenAI API."""

  @skip_if_no_openai
  @live_api
  @retry_on_transient_errors(max_retries=2)
  def test_medication_extraction(self):
    """Test medication extraction with OpenAI models."""
    prompt = textwrap.dedent("""\
        Extract medication information including medication name, dosage, route, frequency,
        and duration in the order they appear in the text.""")

    examples = get_basic_medication_examples()
    input_text = "Patient took 400 mg PO Ibuprofen q4h for two days."

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        language_model_type=OpenAILanguageModel,
        model_id=DEFAULT_OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        fence_output=True,
        use_schema_constraints=False,
        language_model_params=OPENAI_MODEL_PARAMS,
    )

    assert result is not None
    assert hasattr(result, "extractions")
    assert len(result.extractions) > 0

    expected_classes = {
        "dosage",
        "route",
        "medication",
        "frequency",
        "duration",
    }
    assert_extractions_contain(self, result, expected_classes)
    assert_valid_char_intervals(self, result)

    # Using regex for precise matching to avoid false positives
    medication_texts = extract_by_class(result, "medication")
    self.assertTrue(
        any(
            re.search(r"\bIbuprofen\b", text, re.IGNORECASE)
            for text in medication_texts
        ),
        f"No Ibuprofen found in: {medication_texts}",
    )

    dosage_texts = extract_by_class(result, "dosage")
    self.assertTrue(
        any(
            re.search(r"\b400\s*mg\b", text, re.IGNORECASE)
            for text in dosage_texts
        ),
        f"No 400mg dosage found in: {dosage_texts}",
    )

    route_texts = extract_by_class(result, "route")
    self.assertTrue(
        any(
            re.search(r"\b(PO|oral)\b", text, re.IGNORECASE)
            for text in route_texts
        ),
        f"No PO/oral route found in: {route_texts}",
    )

  @skip_if_no_openai
  @live_api
  @retry_on_transient_errors(max_retries=2)
  def test_medication_relationship_extraction(self):
    """Test relationship extraction for medications with OpenAI."""
    input_text = """
    The patient was prescribed Lisinopril and Metformin last month.
    He takes the Lisinopril 10mg daily for hypertension, but often misses
    his Metformin 500mg dose which should be taken twice daily for diabetes.
    """

    prompt = textwrap.dedent("""
        Extract medications with their details, using attributes to group related information:

        1. Extract entities in the order they appear in the text
        2. Each entity must have a 'medication_group' attribute linking it to its medication
        3. All details about a medication should share the same medication_group value
    """)

    examples = get_relationship_examples()

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        language_model_type=OpenAILanguageModel,
        model_id=DEFAULT_OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        fence_output=True,
        use_schema_constraints=False,
        language_model_params=OPENAI_MODEL_PARAMS,
    )

    assert result is not None
    assert len(result.extractions) > 0
    assert_valid_char_intervals(self, result)

    medication_groups = {}
    for extraction in result.extractions:
      assert (
          extraction.attributes is not None
      ), f"Missing attributes for {extraction.extraction_text}"
      assert (
          "medication_group" in extraction.attributes
      ), f"Missing medication_group for {extraction.extraction_text}"

      group_name = extraction.attributes["medication_group"]
      medication_groups.setdefault(group_name, []).append(extraction)

    assert (
        len(medication_groups) >= 2
    ), f"Expected at least 2 medications, found {len(medication_groups)}"

    # Allow flexible matching for dosage field (could be "dosage" or "dose")
    for med_name, extractions in medication_groups.items():
      extraction_classes = {e.extraction_class for e in extractions}
      # At minimum, each group should have the medication itself
      assert (
          "medication" in extraction_classes
      ), f"{med_name} group missing medication entity"
      # Dosage is expected but might be formatted differently
      assert any(
          c in extraction_classes for c in ["dosage", "dose"]
      ), f"{med_name} group missing dosage"
