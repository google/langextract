# Vertex AI Batch Processing Guide

The Vertex AI Batch API offers significant cost savings (~50%) for large, non-time-critical workloads. `langextract` seamlessly integrates this with automatic routing, caching, and fault tolerance.

**[Vertex AI Batch Prediction Documentation →](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini)**
**[Quotas & Limits →](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/quotas#batch-prediction-quotas)**

## Real-World Example: Processing Shakespeare

This example demonstrates how to process a large text (the first ~20 pages of *Romeo and Juliet*) using the Batch API. We use a small chunk size (`max_char_buffer=500`) to generate enough chunks to trigger batch processing.

```python
import requests
import textwrap
import langextract as lx
import logging

# Configure logging to see progress (both in console and file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_process.log"),
        logging.StreamHandler()
    ]
)

# 1. Download Text (Shakespeare's Romeo and Juliet)
url = "https://www.gutenberg.org/files/1513/1513-0.txt"
print(f"Downloading {url}...")
text = requests.get(url).text

# Process first ~20 pages (approx. 60k characters).
text_subset = text[:60000]
print(f"Processing first {len(text_subset)} characters...")

# 2. Define Prompt & Examples
prompt = textwrap.dedent("""\
    Extract characters and emotions from the text.
    Use exact text from the input for extraction_text.""")

examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks?",
        extractions=[
            lx.data.Extraction(extraction_class="character", extraction_text="ROMEO"),
            lx.data.Extraction(extraction_class="emotion", extraction_text="But soft!"),
        ]
    )
]

# 3. Configure Batch Settings
batch_config = {
    "enabled": True,
    "threshold": 10,
    "poll_interval": 30,
    "timeout": 3600,
    # Set to True to cache results in GCS. Add timestamp to prompt to force re-run.
    "enable_caching": True,
    # Retention policy for GCS bucket (days). None for permanent.
    "retention_days": 30,
}

# 4. Run Extraction
# langextract will automatically chunk the text and submit a batch job.
results = lx.extract(
    text_or_documents=text_subset,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    max_char_buffer=500,
    batch_length=1000,
    language_model_params={
        "vertexai": True,
        "project": "your-gcp-project", # TODO: Replace with your Project ID.
        "location": "us-central1",
        "batch": batch_config
    }
)

## GCS File Structure

The library automatically creates and manages a GCS bucket for you, named:
`langextract-{project}-{location}-batch`

Inside this bucket, data is organized as follows:

- **Input**: `batch-input/{job_name}.jsonl`
- **Output**: `batch-input/{job_name}/dest/prediction-model-{timestamp}/predictions.jsonl`
- **Cache**: `cache/{hash}.json` (Individual cached results)

## Cost Optimization & Caching

LangExtract's batch processing is designed to minimize costs:

1.  **Cost Efficiency**: Vertex AI Batch predictions are typically ~50% cheaper than online predictions.
2.  **Smart Caching**:
    -   Results are cached in your GCS bucket (`cache/` directory).
    -   **Instant Retrieval**: Re-running identical prompts fetches results directly from storage, bypassing model inference.
    -   **Reduced Inference**: You avoid paying for redundant model calls on previously processed data.
    -   **Lifecycle Management**: Use `retention_days` (e.g., 30) to automatically clean up old data and manage storage usage.

## Analyze Results
print(f"Extracted {len(results.extractions)} entities.")
print("First 5 extractions:")
for extraction in results.extractions[:5]:
    print(f"- {extraction.extraction_class}: {extraction.extraction_text}")
```

## Sample Output

```text
Extracted 767 entities.
First 5 extractions:
- character: ESCALUS
- character: MERCUTIO
- character: PARIS
- character: Page to Paris
- character: MONTAGUE
```

> **Note on `batch_length`**: The `batch_length` parameter controls how many chunks are submitted in a single batch job. For optimal performance with the Batch API, set this to a high value (e.g., `1000`) to process all chunks in a single job rather than multiple sequential jobs.

## Key Features

### 1. Automatic Routing
`langextract` automatically switches between real-time and batch APIs based on your `threshold`.
- **< Threshold**: Uses real-time API for immediate results.
- **>= Threshold**: Uses Batch API for cost savings.

### 2. Fault Tolerance & Caching
Built-in GCS caching (`enable_caching=True`) allows you to resume interrupted jobs without re-processing completed items, saving time and cost.

### 3. Automated Storage
`langextract` handles all GCS operations automatically using a dedicated bucket (`gs://langextract-{project}-{location}-batch`). Note that input/output files are retained for debugging.

## Tracking Job Status

To monitor progress, you can watch the log file from a separate terminal:

```bash
tail -f batch_process.log
```

When running a batch job, `langextract` provides clear log feedback with a direct link to the Google Cloud Console:

```text
INFO - Batch job created successfully: projects/123456789/locations/us-central1/batchPredictionJobs/987654321
INFO - Job State: JobState.JOB_STATE_PENDING
INFO - Job Console URL: https://console.cloud.google.com/vertex-ai/jobs/batch-predictions/987654321?project=123456789
INFO - Batch job is running... (State: JOB_STATE_PENDING)
INFO - Batch job is running... (State: JOB_STATE_RUNNING)
```

- **Completion**: Once the job succeeds, `langextract` automatically downloads, parses, and aligns the results.
