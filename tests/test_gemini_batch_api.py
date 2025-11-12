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

"""Tests for Gemini Batch API functionality."""

import io
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google import genai
from google.api_core import exceptions

from langextract.providers import gemini
from langextract.providers import gemini_batch as gb
from langextract.providers import schemas


def create_mock_batch_job(
    state=genai.types.JobState.JOB_STATE_SUCCEEDED,
    gcs_uri=f"gs://bucket/output/file{gb._EXT_JSONL}",
):
  """Create a mock BatchJob for testing."""
  job = mock.create_autospec(genai.types.BatchJob, instance=True)
  job.name = "batches/123"
  job.state = state
  job.dest = mock.create_autospec(
      genai.types.BatchJobDestination, instance=True
  )
  job.dest.gcs_uri = gcs_uri
  return job


def _create_batch_response(idx, text_content):
  """Helper to create a batch output line with response."""
  if not isinstance(text_content, str):
    text_content = json.dumps(text_content, separators=(",", ":"))
  return json.dumps({
      "key": f"{gb._KEY_IDX}{idx}",
      "response": {
          "candidates": [{"content": {"parts": [{"text": text_content}]}}]
      },
  })


def _create_batch_error(idx, code, message):
  """Helper to create a batch output line with error."""
  return json.dumps({
      "key": f"{gb._KEY_IDX}{idx}",
      "error": {"code": code, "message": message},
  })


class TestGeminiBatchAPI(absltest.TestCase):
  """Test Gemini Batch API routing and functionality."""

  def setUp(self):
    super().setUp()
    self.mock_storage_cls = self.enter_context(
        mock.patch.object(gb.storage, "Client", autospec=True)
    )
    self.mock_storage_client = self.mock_storage_cls.return_value
    self.mock_bucket = self.mock_storage_client.bucket.return_value
    self.mock_blob = self.mock_bucket.blob.return_value

  @mock.patch.object(genai, "Client", autospec=True)
  def test_batch_routing_vertex(self, mock_client_cls):
    """Test that batch API is used when enabled and threshold is met (Vertex)."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True

    self.mock_storage_client.create_bucket.return_value = self.mock_bucket

    output_blob = mock.create_autospec(gb.storage.Blob, instance=True)
    output_blob.name = "output.jsonl"
    # Mock blob.open context manager
    output_blob.open.return_value.__enter__.return_value = io.StringIO(
        "\n".join([
            _create_batch_response(0, {"ok": 1}),
            _create_batch_response(1, {"ok": 2}),
        ])
    )
    self.mock_bucket.list_blobs.return_value = [output_blob]

    mock_client.batches.create.return_value = create_mock_batch_job()
    mock_client.batches.get.return_value = create_mock_batch_job()

    model = gemini.GeminiLanguageModel(
        model_id="gemini-2.5-flash",
        vertexai=True,
        project="test-project",
        location=gb._DEFAULT_LOCATION,
        batch={"enabled": True, "threshold": 2, "poll_interval": 1},
    )
    prompts = ["p1", "p2"]
    outs = list(model.infer(prompts))

    self.assertLen(outs, 2)
    self.assertEqual(outs[0][0].output, '{"ok":1}')
    self.assertEqual(outs[1][0].output, '{"ok":2}')

    self.mock_blob.upload_from_filename.assert_called()

    mock_client.batches.create.assert_called()

  @mock.patch.object(genai, "Client", autospec=True)
  def test_realtime_when_disabled(self, mock_client_cls):
    """Test that real-time API is used when batch is disabled."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True
    mock_response = mock.create_autospec(
        genai.types.GenerateContentResponse, instance=True
    )
    mock_response.text = '{"ok":1}'
    mock_client.models.generate_content.return_value = mock_response

    model = gemini.GeminiLanguageModel(
        model_id="gemini-2.5-flash",
        vertexai=True,
        project="p",
        location="l",
        batch={"enabled": False},
    )
    outs = list(model.infer(["hello"]))

    self.assertLen(outs, 1)
    self.assertEqual(outs[0][0].output, '{"ok":1}')
    mock_client.models.generate_content.assert_called()
    mock_client.batches.create.assert_not_called()

  @mock.patch.object(genai, "Client", autospec=True)
  def test_realtime_when_below_threshold(self, mock_client_cls):
    """Test that real-time API is used when prompt count is below threshold."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True
    mock_response = mock.create_autospec(
        genai.types.GenerateContentResponse, instance=True
    )
    mock_response.text = '{"ok":1}'
    mock_client.models.generate_content.return_value = mock_response

    model = gemini.GeminiLanguageModel(
        model_id="gemini-2.5-flash",
        vertexai=True,
        project="p",
        location="l",
        batch={"enabled": True, "threshold": 10},
    )
    outs = list(model.infer(["hello"]))

    self.assertLen(outs, 1)
    self.assertEqual(outs[0][0].output, '{"ok":1}')
    mock_client.models.generate_content.assert_called()
    mock_client.batches.create.assert_not_called()

  @mock.patch.object(genai, "Client", autospec=True)
  def test_batch_with_schema(self, mock_client_cls):
    """Test that batch API properly includes schema when configured."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True

    output_blob = mock.create_autospec(gb.storage.Blob, instance=True)
    output_blob.name = f"output{gb._EXT_JSONL}"
    output_blob.open.return_value.__enter__.return_value = io.StringIO(
        _create_batch_response(0, {"name": "test"})
    )
    self.mock_bucket.list_blobs.return_value = [output_blob]

    mock_client.batches.create.return_value = create_mock_batch_job()
    mock_client.batches.get.return_value = create_mock_batch_job()

    mock_schema = mock.create_autospec(
        schemas.gemini.GeminiSchema, instance=True
    )
    mock_schema.schema_dict = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }

    model = gemini.GeminiLanguageModel(
        model_id="gemini-2.5-flash",
        vertexai=True,
        project="p",
        location="l",
        gemini_schema=mock_schema,
        batch={"enabled": True, "threshold": 1},
    )

    # Mock _submit_file to verify the request payload contains the schema.
    with mock.patch.object(gb, "_submit_file", autospec=True) as mock_submit:
      mock_submit.return_value = create_mock_batch_job()

      outs = list(model.infer(["test prompt"]))

      self.assertLen(outs, 1)
      self.assertEqual(outs[0][0].output, '{"name":"test"}')

      # Verify _submit_file was called with correct arguments.
      mock_submit.assert_called_with(
          mock_client,
          "gemini-2.5-flash",
          [{
              "contents": [
                  {"role": "user", "parts": [{"text": "test prompt"}]}
              ],
              "generationConfig": {
                  "responseMimeType": "application/json",
                  "responseSchema": mock_schema.schema_dict,
                  "temperature": 0.0,
              },
          }],
          mock.ANY,  # Display name contains timestamp/random.
      )

    self.assertEqual(model.gemini_schema.schema_dict, mock_schema.schema_dict)

  @mock.patch.object(genai, "Client", autospec=True)
  def test_batch_error_handling(self, mock_client_cls):
    """Test that batch errors are properly handled and raised."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True
    mock_client.batches.create.side_effect = Exception("Batch API error")

    model = gemini.GeminiLanguageModel(
        model_id="gemini-2.5-flash",
        vertexai=True,
        project="p",
        location="l",
        batch={"enabled": True, "threshold": 1},
    )

    with self.assertRaisesRegex(Exception, "Gemini Batch API error"):
      list(model.infer(["test prompt"]))

  @mock.patch.object(genai, "Client", autospec=True)
  def test_file_based_ordering(self, mock_client_cls):
    """Test that file-based results are returned in correct order."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True

    # Define inputs and expected outputs
    prompts = ["prompt 0", "prompt 1", "prompt 2"]
    # Simulate shuffled response in the file
    output_blob = mock.create_autospec(gb.storage.Blob, instance=True)
    output_blob.name = f"output{gb._EXT_JSONL}"
    output_blob.open.return_value.__enter__.return_value = io.StringIO(
        "\n".join([
            _create_batch_response(2, "response 2"),
            _create_batch_response(0, "response 0"),
            _create_batch_response(1, "response 1"),
        ])
    )
    self.mock_bucket.list_blobs.return_value = [output_blob]

    job = create_mock_batch_job()
    mock_client.batches.create.return_value = job
    mock_client.batches.get.return_value = job

    model = gemini.GeminiLanguageModel(
        model_id="gemini-2.5-flash",
        vertexai=True,
        project="p",
        location="l",
        batch={"enabled": True, "threshold": 1},
    )

    results = list(model.infer(prompts))

    # Verify results are in original order despite shuffled response
    self.assertListEqual(
        [r[0].output for r in results],
        ["response 0", "response 1", "response 2"],
    )

  @mock.patch.object(genai, "Client", autospec=True)
  def test_max_prompts_per_job(self, mock_client_cls):
    """Test that requests are split into multiple batch jobs when they exceed max_prompts_per_job.

    This verifies that:
    1. Large requests are chunked correctly based on the limit.
    2. Multiple batch jobs are submitted.
    3. Results are aggregated and returned in the correct order.
    """
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True

    # Define inputs and expected behavior
    prompts = ["p1", "p2", "p3", "p4", "p5"]
    max_prompts_per_job = 2
    # Expected chunks: ["p1", "p2"], ["p3", "p4"], ["p5"]

    # Setup mock storage and blobs for 3 separate jobs
    blob0 = mock.create_autospec(gb.storage.Blob, instance=True)
    blob0.name = f"out0{gb._EXT_JSONL}"
    blob0.open.return_value.__enter__.return_value = io.StringIO(
        "\n".join([
            _create_batch_response(0, "r1"),
            _create_batch_response(1, "r2"),
        ])
    )

    blob1 = mock.create_autospec(gb.storage.Blob, instance=True)
    blob1.name = f"out1{gb._EXT_JSONL}"
    blob1.open.return_value.__enter__.return_value = io.StringIO(
        "\n".join([
            _create_batch_response(0, "r3"),
            _create_batch_response(1, "r4"),
        ])
    )

    blob2 = mock.create_autospec(gb.storage.Blob, instance=True)
    blob2.name = f"out2{gb._EXT_JSONL}"
    blob2.open.return_value.__enter__.return_value = io.StringIO(
        _create_batch_response(0, "r5")
    )

    def list_blobs_side_effect(prefix=None):
      if "part-0" in prefix:
        return [blob0]
      if "part-1" in prefix:
        return [blob1]
      if "part-2" in prefix:
        return [blob2]
      return []

    self.mock_bucket.list_blobs.side_effect = list_blobs_side_effect

    # Setup mock jobs
    job0 = create_mock_batch_job(gcs_uri="gs://b/batch-input/part-0/out")
    job1 = create_mock_batch_job(gcs_uri="gs://b/batch-input/part-1/out")
    job2 = create_mock_batch_job(gcs_uri="gs://b/batch-input/part-2/out")

    mock_client.batches.create.side_effect = [job0, job1, job2]
    mock_client.batches.get.side_effect = [job0, job1, job2]

    model = gemini.GeminiLanguageModel(
        model_id="gemini-2.5-flash",
        vertexai=True,
        project="p",
        location="l",
        batch={
            "enabled": True,
            "threshold": 1,
            "max_prompts_per_job": max_prompts_per_job,
        },
    )

    results = list(model.infer(prompts))

    self.assertEqual(mock_client.batches.create.call_count, 3)
    self.assertListEqual(
        [r[0].output for r in results], ["r1", "r2", "r3", "r4", "r5"]
    )

  @mock.patch.object(genai, "Client", autospec=True)
  def test_batch_item_error(self, mock_client_cls):
    """Test that batch item errors raise exception."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True

    output_blob = mock.create_autospec(gb.storage.Blob, instance=True)
    output_blob.name = f"output{gb._EXT_JSONL}"
    output_blob.open.return_value.__enter__.return_value = io.StringIO(
        _create_batch_error(0, 13, "Internal error")
    )
    self.mock_bucket.list_blobs.return_value = [output_blob]

    job = create_mock_batch_job()
    mock_client.batches.create.return_value = job
    mock_client.batches.get.return_value = job

    model = gemini.GeminiLanguageModel(
        model_id="gemini-2.5-flash",
        vertexai=True,
        project="p",
        location="l",
        batch={"enabled": True, "threshold": 1},
    )

    with self.assertRaisesRegex(Exception, "Batch item error"):
      list(model.infer(["test"]))


class BatchConfigValidationTest(parameterized.TestCase):
  """Test BatchConfig validation logic."""

  @parameterized.named_parameters(
      dict(testcase_name="threshold_lt_1", threshold=0),
      dict(testcase_name="poll_interval_le_0", poll_interval=0),
      dict(testcase_name="timeout_le_0", timeout=0),
      dict(testcase_name="max_prompts_per_job_le_0", max_prompts_per_job=0),
  )
  def test_validation_errors(self, **overrides):
    """Verify validation errors for invalid config values."""
    with self.assertRaises(ValueError):
      gb.BatchConfig(**overrides)


class EmptyAndPaddingTest(absltest.TestCase):
  """Test empty prompt handling and result padding/trimming."""

  @mock.patch.object(genai, "Client", autospec=True)
  def test_empty_prompts_fast_path(self, mock_client_cls):
    """Verify empty prompts return immediately without API calls."""
    outs = gb.infer_batch(
        client=mock_client_cls.return_value,
        model_id="m",
        prompts=[],
        schema_dict=None,
        gen_config={},
        cfg=gb.BatchConfig(enabled=True, poll_interval=1),
    )
    self.assertEqual(outs, [])

  @mock.patch.object(genai, "Client", autospec=True)
  def test_file_pad_to_expected_count(self, mock_client_cls):
    """Verify padding to maintain 1:1 alignment with input prompts."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True

    with mock.patch.object(gb.storage, "Client", autospec=True) as mock_storage:
      mock_bucket = mock_storage.return_value.bucket.return_value
      output_blob = mock.create_autospec(gb.storage.Blob, instance=True)
      output_blob.name = f"output{gb._EXT_JSONL}"
      output_blob.open.return_value.__enter__.return_value = io.StringIO(
          _create_batch_response(0, "only_one")
      )
      mock_bucket.list_blobs.return_value = [output_blob]

      job = create_mock_batch_job()
      mock_client.batches.create.return_value = job
      mock_client.batches.get.return_value = job

      cfg = gb.BatchConfig(enabled=True, threshold=1, poll_interval=1)
      outs = gb.infer_batch(
          client=mock_client,
          model_id="m",
          prompts=["p1", "p2"],
          schema_dict=None,
          gen_config={},
          cfg=cfg,
      )
      self.assertEqual(outs, ["only_one", ""])  # padded


class GCSBatchCachingTest(absltest.TestCase):
  """Test GCS batch caching functionality."""

  def setUp(self):
    super().setUp()
    self.mock_storage_cls = self.enter_context(
        mock.patch.object(gb.storage, "Client", autospec=True)
    )
    self.mock_storage_client = self.mock_storage_cls.return_value
    self.mock_bucket = self.mock_storage_client.bucket.return_value
    self.mock_blob = self.mock_bucket.blob.return_value

  @mock.patch.object(genai, "Client", autospec=True)
  def test_cache_hit_skips_inference(self, mock_client_cls):
    """Test that fully cached prompts skip inference."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True
    mock_client.project = "p"
    mock_client.location = "l"

    self.mock_blob.download_as_text.return_value = '{"text": "cached_response"}'

    cfg = gb.BatchConfig(
        enabled=True,
        threshold=1,
        enable_caching=True,
    )

    outs = gb.infer_batch(
        client=mock_client,
        model_id="m",
        prompts=["p1"],
        schema_dict=None,
        gen_config={},
        cfg=cfg,
    )

    self.assertListEqual(outs, ["cached_response"])

    mock_client.batches.create.assert_not_called()

    self.mock_bucket.blob.assert_called()

  @mock.patch.object(genai, "Client", autospec=True)
  def test_partial_cache_hit(self, mock_client_cls):
    """Test that partial cache hits only submit missing prompts."""
    mock_client = mock_client_cls.return_value
    mock_client.vertexai = True
    mock_client.project = "p"
    mock_client.location = "l"

    # Mock GCS cache: hit for "cached_prompt", miss for "new_prompt"
    # We mock _compute_hash to avoid dealing with complex hashing in test
    with mock.patch.object(gb.GCSBatchCache, "_compute_hash") as mock_hash:
      mock_hash.side_effect = lambda k: f"hash_{k['prompt']}"

      # Pre-configure blobs
      blob_hit = mock.create_autospec(gb.storage.Blob, instance=True)
      blob_hit.download_as_text.return_value = '{"text": "cached_response"}'

      blob_miss = mock.create_autospec(gb.storage.Blob, instance=True)
      blob_miss.download_as_text.side_effect = exceptions.NotFound("Not found")

      def get_blob(name):
        if "hash_cached_prompt" in name:
          return blob_hit
        return blob_miss

      self.mock_bucket.blob.side_effect = get_blob

      # Mock list_blobs to return the batch output file for the new prompt
      output_blob = mock.create_autospec(gb.storage.Blob, instance=True)
      output_blob.name = f"output{gb._EXT_JSONL}"
      output_blob.open.return_value.__enter__.return_value = io.StringIO(
          _create_batch_response(0, "new_response")
      )
      self.mock_bucket.list_blobs.return_value = [output_blob]

      job = create_mock_batch_job()
      mock_client.batches.create.return_value = job
      mock_client.batches.get.return_value = job

      cfg = gb.BatchConfig(
          enabled=True,
          threshold=1,
          enable_caching=True,
      )

      outs = gb.infer_batch(
          client=mock_client,
          model_id="m",
          prompts=["cached_prompt", "new_prompt"],
          schema_dict=None,
          gen_config={},
          cfg=cfg,
      )

      self.assertListEqual(outs, ["cached_response", "new_response"])
      mock_client.batches.create.assert_called_once()

      # Verify "new_response" was uploaded to cache (using the miss blob)
      # The blob used for upload is blob_miss because it was returned for the miss key
      upload_calls = [
          call
          for call in blob_miss.upload_from_string.mock_calls
          if "new_response" in str(call)
      ]
      self.assertTrue(
          upload_calls, "Should have uploaded new_response to cache"
      )

  def test_cache_hashing_stability(self):
    """Test that hash is stable for same inputs."""
    cache = gb.GCSBatchCache("b")
    data1 = {"a": 1, "b": 2}
    data2 = {"b": 2, "a": 1}
    self.assertEqual(cache._compute_hash(data1), cache._compute_hash(data2))


if __name__ == "__main__":
  absltest.main()
