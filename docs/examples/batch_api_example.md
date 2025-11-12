# Vertex AI Batch Processing Guide

The Vertex AI Batch API offers significant cost savings (~50%) for large, non-time-critical workloads. `langextract` seamlessly integrates this with automatic routing, caching, and fault tolerance.

**[Vertex AI Batch Prediction Documentation →](https://cloud.google.com/vertex-ai/generative-ai/docs/batch-prediction)**

## Quick Start

```python
import langextract as lx

# 1. Configure Batch Settings
batch_config = {
    "enabled": True,              # Enable batch processing
    "threshold": 50,              # Min prompts to trigger batch mode (default: 50)
    "poll_interval": 30,          # Status check interval in seconds (default: 30)
    "timeout": 3600,              # Max wait time in seconds (default: 3600)
    "enable_caching": True        # Enable GCS-based caching (default: True)
}

# 2. Run Extraction
results = lx.extract(
    text_or_documents=documents,  # List of 50+ documents
    prompt_description="Extract person names and locations",
    examples=examples,
    model_id="gemini-2.5-flash",
    language_model_params={
        "vertexai": True,
        "project": "your-gcp-project",
        "location": "us-central1",
        "batch": batch_config
    }
)
```

## Key Features

### 1. Automatic Routing
`langextract` automatically switches between real-time and batch APIs based on your `threshold`.
- **< Threshold**: Uses real-time API for immediate results.
- **>= Threshold**: Uses Batch API for cost savings.

### 2. Fault Tolerance & Caching
Built-in GCS caching (`enable_caching=True`) ensures robustness for long-running jobs:
- **Resume Capability**: If a job fails or times out, simply re-run the script. `langextract` detects cached results and only submits missing items.
- **Cost Efficiency**: You never pay to re-process the same prompt twice.
- **Seamless Restoration**: The final output is always a complete, ordered list of `AnnotatedDocument` objects, indistinguishable from a single successful run.

### 3. Automated Storage
- **Bucket**: Automatically creates/uses `gs://langextract-{project}-{location}-batch`.
- **Permissions**: Requires `storage.buckets.create`, `storage.objects.create`, `storage.objects.get`.
- **Cleanup**: Input/output files are retained for debugging; clean up manually if needed.

## Best Practices

- **Batch Size**: Use for workloads with 50+ prompts (default threshold).
- **Timeout**: Set `timeout` generously (e.g., 3600s+) for large batches.
- **Monitoring**: Watch standard logs for job status updates.
