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

import functools
import json
import os
import re
import textwrap
import time
from typing import Any
import unittest
from unittest import mock
import uuid

import dotenv
import google.auth
import google.auth.exceptions
import google.genai.errors
import pytest

from langextract import data
import langextract as lx
from langextract.core import tokenizer as tokenizer_lib
from langextract.providers import gemini_batch as gb

dotenv.load_dotenv(override=True)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get(
    "LANGEXTRACT_API_KEY"
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT") or os.environ.get(
    "GOOGLE_CLOUD_PROJECT"
)
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")


def has_vertex_ai_credentials():
  """Check if Vertex AI credentials are available."""
  if not VERTEX_PROJECT:
    return False
  try:
    credentials, _ = google.auth.default()
    return credentials is not None
  except (ImportError, google.auth.exceptions.DefaultCredentialsError):
    return False


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
skip_if_no_vertex = pytest.mark.skipif(
    not has_vertex_ai_credentials(),
    reason=(
        "Vertex AI credentials not available (set GOOGLE_CLOUD_PROJECT and"
        " configure gcloud auth)"
    ),
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

# Extraction Classes
_CLASS_MEDICATION = "medication"
_CLASS_DOSAGE = "dosage"
_CLASS_ROUTE = "route"
_CLASS_FREQUENCY = "frequency"
_CLASS_DURATION = "duration"
_CLASS_CONDITION = "condition"

INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 8.0


def retry_on_transient_errors(max_retries=3, backoff_factor=2.0):
  """Decorator to retry tests on transient API errors with exponential backoff.

  Args:
    max_retries (int): Maximum number of retry attempts
    backoff_factor (float): Multiplier for exponential backoff (e.g., 2.0 = 1s, 2s, 4s)
  """

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      last_exception = None
      delay = INITIAL_RETRY_DELAY

      for attempt in range(max_retries + 1):
        try:
          return func(*args, **kwargs)
        except (
            lx.exceptions.LangExtractError,
            google.genai.errors.ClientError,
            ConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
        ) as e:
          last_exception = e
          if attempt < max_retries:
            print(
                f"\nRetryable error ({type(e).__name__}) on attempt"
                f" {attempt + 1}/{max_retries + 1}: {e}"
            )
            time.sleep(delay)
            delay = min(delay * backoff_factor, MAX_RETRY_DELAY)
            continue

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
                  extraction_class=_CLASS_DOSAGE, extraction_text="250 mg"
              ),
              lx.data.Extraction(
                  extraction_class=_CLASS_ROUTE, extraction_text="IV"
              ),
              lx.data.Extraction(
                  extraction_class=_CLASS_MEDICATION,
                  extraction_text="Cefazolin",
              ),
              lx.data.Extraction(
                  extraction_class=_CLASS_FREQUENCY,
                  extraction_text="TID",  # TID = three times a day
              ),
              lx.data.Extraction(
                  extraction_class=_CLASS_DURATION,
                  extraction_text="for one week",
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
                  extraction_class=_CLASS_MEDICATION,
                  extraction_text="Aspirin",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class=_CLASS_DOSAGE,
                  extraction_text="100mg",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class=_CLASS_FREQUENCY,
                  extraction_text="daily",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class=_CLASS_CONDITION,
                  extraction_text="heart health",
                  attributes={"medication_group": "Aspirin"},
              ),
              # Second medication group
              lx.data.Extraction(
                  extraction_class=_CLASS_MEDICATION,
                  extraction_text="Simvastatin",
                  attributes={"medication_group": "Simvastatin"},
              ),
              lx.data.Extraction(
                  extraction_class=_CLASS_DOSAGE,
                  extraction_text="20mg",
                  attributes={"medication_group": "Simvastatin"},
              ),
              lx.data.Extraction(
                  extraction_class=_CLASS_FREQUENCY,
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
    if isinstance(result, lx.data.AnnotatedDocument) and result.text:
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

  def _check_cached_result(self, result_json: dict[str, Any]) -> bool:
    """Check if cached result contains expected medication data.

    Args:
      result_json: The raw JSON dict from the cache file.
                   Expected format: {"text": "JSON_STRING_OF_RESULT"}

    Returns:
      True if the result contains valid medication extractions, False otherwise.
    """
    try:
      text_content = result_json.get("text")
      if not isinstance(text_content, str):
        return False

      inner_json = json.loads(text_content)
      if not isinstance(inner_json, dict):
        return False

      extractions_data = inner_json.get(data.EXTRACTIONS_KEY)
      if not isinstance(extractions_data, list):
        return False

      extractions = []
      for item in extractions_data:
        if isinstance(item, dict):
          clean_item = {k: v for k, v in item.items() if not k.startswith("_")}
          extractions.append(data.Extraction(**clean_item))

      doc = data.AnnotatedDocument(
          text=inner_json.get("text"), extractions=extractions
      )

      if not doc.extractions:
        return False

      # Check for specific content
      medication_texts = extract_by_class(doc, _CLASS_MEDICATION)
      dosage_texts = extract_by_class(doc, _CLASS_DOSAGE)

      has_lisinopril = any("Lisinopril" in t for t in medication_texts)
      has_10mg = any("10mg" in t for t in dosage_texts)

      return has_lisinopril and has_10mg

    except (json.JSONDecodeError, TypeError, ValueError):
      return False

  def _verify_gcs_cache_content(self, bucket_name):
    """Verify that GCS cache contains expected structured results."""
    cache = gb.GCSBatchCache(bucket_name, project=VERTEX_PROJECT)
    found_content = False

    # Use iter_items() to check cache content
    items = list(cache.iter_items())
    self.assertTrue(len(items) > 0, "No cache files found in GCS bucket")

    for _, text in items:
      try:
        result_json = json.loads(text)
        if self._check_cached_result(result_json):
          found_content = True
          break
      except (json.JSONDecodeError, TypeError, ValueError):
        continue

    self.assertTrue(
        found_content,
        "Could not find expected structured result in GCS cache files",
    )

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
    self.assertIsInstance(result, lx.data.AnnotatedDocument)
    assert len(result.extractions) > 0

    expected_classes = {
        _CLASS_DOSAGE,
        _CLASS_ROUTE,
        _CLASS_MEDICATION,
        _CLASS_FREQUENCY,
        _CLASS_DURATION,
    }
    assert_extractions_contain(self, result, expected_classes)
    assert_valid_char_intervals(self, result)

    # Using regex for precise matching to avoid false positives
    medication_texts = extract_by_class(result, _CLASS_MEDICATION)
    self.assertTrue(
        any(
            re.search(r"\bIbuprofen\b", text, re.IGNORECASE)
            for text in medication_texts
        ),
        f"No Ibuprofen found in: {medication_texts}",
    )

    dosage_texts = extract_by_class(result, _CLASS_DOSAGE)
    self.assertTrue(
        any(
            re.search(r"\b400\s*mg\b", text, re.IGNORECASE)
            for text in dosage_texts
        ),
        f"No 400mg dosage found in: {dosage_texts}",
    )

    route_texts = extract_by_class(result, _CLASS_ROUTE)
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
                    extraction_class=_CLASS_MEDICATION,
                    extraction_text="aspirin",
                    attributes={
                        _CLASS_DOSAGE: "20mg",
                        _CLASS_FREQUENCY: "twice daily",
                    },
                ),
            ],
        )
    ]

    unicode_tokenizer = tokenizer_lib.UnicodeTokenizer()

    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=DEFAULT_GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        language_model_params=GEMINI_MODEL_PARAMS,
        tokenizer=unicode_tokenizer,
    )

    assert result is not None
    self.assertIsInstance(result, lx.data.AnnotatedDocument)
    assert len(result.extractions) > 0

    medication_extractions = [
        e for e in result.extractions if e.extraction_class == _CLASS_MEDICATION
    ]
    assert (
        len(medication_extractions) > 0
    ), "No medication entities found in Japanese text"
    assert_valid_char_intervals(self, result)

  @skip_if_no_gemini
  @live_api
  @retry_on_transient_errors(max_retries=2)
  def test_explicit_provider_gemini(self):
    """Test using explicit provider with Gemini."""
    config = lx.factory.ModelConfig(
        model_id=DEFAULT_GEMINI_MODEL,
        provider="GeminiLanguageModel",
        provider_kwargs={
            "api_key": GEMINI_API_KEY,
            "temperature": 0.0,
        },
    )

    model = lx.factory.create_model(config)
    self.assertEqual(model.__class__.__name__, "GeminiLanguageModel")
    self.assertEqual(model.model_id, DEFAULT_GEMINI_MODEL)

    config2 = lx.factory.ModelConfig(
        model_id=DEFAULT_GEMINI_MODEL,
        provider="gemini",
        provider_kwargs={
            "api_key": GEMINI_API_KEY,
        },
    )

    model2 = lx.factory.create_model(config2)
    self.assertEqual(model2.__class__.__name__, "GeminiLanguageModel")

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
          _CLASS_MEDICATION in extraction_classes
      ), f"{med_name} group missing medication entity"
      # Dosage is expected but might be formatted differently
      assert any(
          c in extraction_classes for c in [_CLASS_DOSAGE, "dose"]
      ), f"{med_name} group missing dosage"

  @skip_if_no_vertex
  @live_api
  @pytest.mark.vertex_ai
  @mock.patch.object(gb, "infer_batch", wraps=gb.infer_batch, autospec=True)
  def test_batch_extraction_vertex_gcs(self, mock_infer_batch):
    """Test extraction using Vertex AI Batch API with GCS.

    This test runs a real Vertex AI Batch job and will take time to complete.
    It is skipped unless VERTEX_PROJECT is set.

    We wrap `infer_batch` to verify that:
    - Batch API is actually called (not falling back to real-time API)
    - Schema dict is passed (non-None) to the batch function
    """

    prompt = textwrap.dedent("""\
        Extract medication information including medication name, dosage, route, frequency,
        and duration in the order they appear in the text.""")

    examples = get_basic_medication_examples()

    documents = [
        lx.data.Document(
            document_id="vx_doc1",
            text="Patient took 400 mg PO Ibuprofen q4h for two days.",
        ),
        lx.data.Document(
            document_id="vx_doc2",
            text="Patient was given 250 mg IV Cefazolin TID for one week.",
        ),
        lx.data.Document(
            document_id="vx_doc3",
            text="Administered 2 mg IV Morphine once for acute pain.",
        ),
        lx.data.Document(
            document_id="vx_doc4",
            text="Prescribed 500 mg PO Amoxicillin BID for infection.",
        ),
        lx.data.Document(
            document_id="vx_doc5",
            text="Given 10 mg IM Haloperidol PRN for agitation.",
        ),
    ]
    expected_meds = [
        "Ibuprofen",
        "Cefazolin",
        "Morphine",
        "Amoxicillin",
        "Haloperidol",
    ]

    language_model_params = dict(GEMINI_MODEL_PARAMS)
    language_model_params["vertexai"] = True
    language_model_params["project"] = VERTEX_PROJECT
    language_model_params["location"] = VERTEX_LOCATION
    language_model_params["batch"] = {
        "enabled": True,
        "threshold": 2,
        "poll_interval": 1,  # Fast polling for test
        "timeout": 900,  # 15 minutes for actual batch job completion
    }

    batch_result = lx.extract(
        text_or_documents=documents,
        prompt_description=prompt,
        examples=examples,
        model_id=DEFAULT_GEMINI_MODEL,
        language_model_params=language_model_params,
    )

    mock_infer_batch.assert_called_once()
    call_args = mock_infer_batch.call_args
    schema_dict_arg = call_args.kwargs.get("schema_dict")
    self.assertIsNotNone(
        schema_dict_arg,
        "schema_dict should be passed to batch API (not None)",
    )

    self.assertIsInstance(batch_result, list)
    self.assertEqual(
        len(batch_result),
        len(documents),
        f"Expected {len(documents)} results from Vertex batch API",
    )

    for i, (res, med_name) in enumerate(zip(batch_result, expected_meds)):
      self.assertIsInstance(
          res,
          lx.data.AnnotatedDocument,
          f"Result {i} should be an AnnotatedDocument, got {type(res)}",
      )
      self.assertTrue(
          res.extractions,
          f"No extractions for document {i}",
      )
      for extraction in res.extractions:
        self.assertIsInstance(
            extraction,
            lx.data.Extraction,
            "Extraction item should be Extraction object, got"
            f" {type(extraction)}",
        )

      meds = extract_by_class(res, _CLASS_MEDICATION)
      self.assertTrue(
          any(
              re.search(rf"\b{re.escape(med_name)}\b", m, re.IGNORECASE)
              for m in meds
          ),
          f"Expected medication '{med_name}' not found in results: {meds}",
      )

      dosages = extract_by_class(res, _CLASS_DOSAGE)
      self.assertTrue(
          dosages,
          f"No dosage extracted for medication '{med_name}'",
      )

      assert_valid_char_intervals(self, res)

  @skip_if_no_vertex
  @live_api
  @pytest.mark.vertex_ai
  def test_batch_caching_live(self):
    """Test batch caching with real Vertex AI Batch API.

    Verifies that:
    1. First run populates GCS cache
    2. Second run uses cache (returns same results faster)
    """
    prompt = "Extract the medication: Patient takes 10mg Lisinopril."
    examples = get_basic_medication_examples()

    # Use unique IDs to ensure cache isolation between test runs.
    run_id = uuid.uuid4().hex[:8]
    documents = [
        lx.data.Document(
            document_id=f"doc_{i}_{run_id}",
            text=f"Patient takes 10mg Lisinopril {i} {run_id}.",
        )
        for i in range(2)
    ]

    language_model_params = dict(GEMINI_MODEL_PARAMS)
    language_model_params["vertexai"] = True
    language_model_params["project"] = VERTEX_PROJECT
    language_model_params["location"] = VERTEX_LOCATION
    language_model_params["batch"] = {
        "enabled": True,
        "threshold": 2,
        "poll_interval": 1,
        "timeout": 900,
        "enable_caching": True,
    }

    print("\nStarting first batch run (API)...")
    start_time = time.time()
    results1 = list(
        lx.extract(
            text_or_documents=documents,
            prompt_description=prompt,
            examples=examples,
            model_id=DEFAULT_GEMINI_MODEL,
            language_model_params=language_model_params,
        )
    )
    duration1 = time.time() - start_time
    print(f"First run took {duration1:.2f}s")

    print("Starting second batch run (Cache)...")
    start_time = time.time()
    results2 = list(
        lx.extract(
            text_or_documents=documents,
            prompt_description=prompt,
            examples=examples,
            model_id=DEFAULT_GEMINI_MODEL,
            language_model_params=language_model_params,
        )
    )
    duration2 = time.time() - start_time
    print(f"Second run took {duration2:.2f}s")

    self.assertEqual(len(results1), len(results2))
    for r1, r2 in zip(results1, results2):
      self.assertEqual(r1.text, r2.text)
      self.assertEqual(len(r1.extractions), len(r2.extractions))

    self.assertLess(duration2, 10.0, "Second run took too long for cache hit")

    self.assertLess(duration2, 10.0, "Second run took too long for cache hit")

    print("\nVerifying GCS cache content...")
    bucket_name = gb._get_bucket_name(VERTEX_PROJECT, VERTEX_LOCATION)
    print(f"Checking bucket: {bucket_name}")
    self._verify_gcs_cache_content(bucket_name)


class TestCrossChunkContext(unittest.TestCase):
  """Tests for cross-chunk context feature with real API."""

  @skip_if_no_gemini
  @live_api
  @retry_on_transient_errors(max_retries=3)
  def test_context_window_extracts_from_both_chunks(self):
    """Verify context_window_chars enables extraction across chunk boundaries."""
    input_text = (
        "Dr. Sarah Chen is the lead researcher at the institute. "
        "She published groundbreaking work on neural networks last year."
    )
    prompt = textwrap.dedent(
        """\
        Extract all person names, roles, and achievements mentioned in the text.
        Include both explicit names and information associated with pronouns."""
    )
    examples = [
        lx.data.ExampleData(
            text=(
                "Professor James Miller leads the physics department. "
                "He won the Nobel Prize in 2020."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="person",
                    extraction_text="Professor James Miller",
                    attributes={"role": "leads the physics department"},
                ),
                lx.data.Extraction(
                    extraction_class="achievement",
                    extraction_text="won the Nobel Prize in 2020",
                ),
            ],
        )
    ]

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id=DEFAULT_GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        language_model_params=GEMINI_MODEL_PARAMS,
        max_char_buffer=60,
        context_window_chars=50,
    )

    self.assertIsNotNone(result)
    self.assertGreater(len(result.extractions), 0)

    all_extraction_text = " ".join(
        str(e.extraction_text) + " " + str(e.attributes)
        for e in result.extractions
    ).lower()

    has_chunk1_content = any(
        term in all_extraction_text
        for term in ("sarah", "chen", "researcher", "lead")
    )
    has_chunk2_content = any(
        term in all_extraction_text
        for term in ("published", "groundbreaking", "neural", "networks")
    )

    self.assertTrue(
        has_chunk1_content,
        f"Expected chunk 1 content (Sarah Chen). Got: {result.extractions}",
    )
    self.assertTrue(
        has_chunk2_content,
        f"Expected chunk 2 content (publication). Got: {result.extractions}",
    )


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
        model_id=DEFAULT_OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        use_schema_constraints=False,
        language_model_params=OPENAI_MODEL_PARAMS,
    )

    assert result is not None
    self.assertIsInstance(result, lx.data.AnnotatedDocument)
    assert len(result.extractions) > 0

    expected_classes = {
        _CLASS_DOSAGE,
        _CLASS_ROUTE,
        _CLASS_MEDICATION,
        _CLASS_FREQUENCY,
        _CLASS_DURATION,
    }
    assert_extractions_contain(self, result, expected_classes)
    assert_valid_char_intervals(self, result)

    # Using regex for precise matching to avoid false positives
    medication_texts = extract_by_class(result, _CLASS_MEDICATION)
    self.assertTrue(
        any(
            re.search(r"\bIbuprofen\b", text, re.IGNORECASE)
            for text in medication_texts
        ),
        f"No Ibuprofen found in: {medication_texts}",
    )

    dosage_texts = extract_by_class(result, _CLASS_DOSAGE)
    self.assertTrue(
        any(
            re.search(r"\b400\s*mg\b", text, re.IGNORECASE)
            for text in dosage_texts
        ),
        f"No 400mg dosage found in: {dosage_texts}",
    )

    route_texts = extract_by_class(result, _CLASS_ROUTE)
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
  def test_explicit_provider_selection(self):
    """Test using explicit provider parameter for disambiguation."""
    # Test with explicit model_id and provider
    config = lx.factory.ModelConfig(
        model_id=DEFAULT_OPENAI_MODEL,
        provider="OpenAILanguageModel",  # Explicit provider selection
        provider_kwargs={
            "api_key": OPENAI_API_KEY,
            "temperature": 0.0,
        },
    )

    model = lx.factory.create_model(config)

    self.assertIsInstance(model, lx.providers.openai.OpenAILanguageModel)
    self.assertEqual(model.model_id, DEFAULT_OPENAI_MODEL)

    # Also test using provider without model_id (uses default)
    config_default = lx.factory.ModelConfig(
        provider="OpenAILanguageModel",
        provider_kwargs={
            "api_key": OPENAI_API_KEY,
        },
    )

    model_default = lx.factory.create_model(config_default)
    self.assertEqual(model_default.__class__.__name__, "OpenAILanguageModel")
    # Should use the default model_id from the provider
    self.assertEqual(model_default.model_id, "gpt-4o-mini")

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
        model_id=DEFAULT_OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
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
          _CLASS_MEDICATION in extraction_classes
      ), f"{med_name} group missing medication entity"
      # Dosage is expected but might be formatted differently
      assert any(
          c in extraction_classes for c in [_CLASS_DOSAGE, "dose"]
      ), f"{med_name} group missing dosage"
