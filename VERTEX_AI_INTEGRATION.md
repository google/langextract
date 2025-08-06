# Gemini Vertex AI Integration for LangExtract

This document describes the new Gemini Vertex AI integration added to LangExtract, which allows you to use Google Cloud Vertex AI instead of API keys for authentication.

## Overview

The new `GeminiVertexLanguageModel` class provides:
- **Vertex AI Authentication**: Use Google Cloud project and location instead of API keys
- **Thinking Budget Control**: Configure reasoning capabilities for supported models
- **Full Feature Compatibility**: All existing features work with Vertex AI
- **Enhanced Security**: Leverage Google Cloud IAM and service accounts

## Quick Start

### Basic Usage

```python
import langextract as lx

# Define your examples
examples = [
    lx.data.ExampleData(
        text="Patient takes Aspirin 100mg daily for heart health.",
        extractions=[
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Aspirin"
            ),
            lx.data.Extraction(
                extraction_class="dosage", 
                extraction_text="100mg"
            ),
        ]
    )
]

# Extract using Vertex AI
result = lx.extract(
    text_or_documents="Patient was prescribed Lisinopril 10mg daily for hypertension.",
    prompt_description="Extract medication information",
    examples=examples,
    project="your-gcp-project-id",           # Your GCP project
    location="global",                       # Vertex AI location
    language_model_type=lx.inference.GeminiVertexLanguageModel
)
```

### Advanced Configuration

```python
# Advanced Vertex AI configuration
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    project="your-gcp-project-id",
    location="us-central1",                  # Specific region
    thinking_budget=1000,                    # Enable reasoning (0 = no thinking)
    language_model_type=lx.inference.GeminiVertexLanguageModel,
    temperature=0.1,
    max_workers=5,
    language_model_params={
        "safety_settings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        ]
    }
)
```

## Key Features

### 1. Vertex AI Authentication
- **Project-based**: Use your Google Cloud project ID instead of API keys
- **Location-aware**: Specify the region for your Vertex AI deployment
- **IAM Integration**: Leverage Google Cloud's identity and access management

### 2. Thinking Budget
- **Reasoning Control**: Set `thinking_budget` to control model reasoning steps
- **Performance Tuning**: Higher values allow more complex reasoning
- **Cost Management**: Set to 0 to disable thinking and reduce costs

### 3. Full Compatibility
- **Schema Constraints**: Full support for structured outputs
- **Parallel Processing**: Multi-worker support for batch processing
- **Safety Settings**: Configure content filtering and safety thresholds

## Parameters

### New Parameters in `lx.extract()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str \| None` | `None` | Google Cloud project ID for Vertex AI |
| `location` | `str` | `"global"` | Vertex AI location/region |
| `thinking_budget` | `int` | `0` | Reasoning budget (0 = no thinking) |

### GeminiVertexLanguageModel Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str \| None` | `None` | **Required** - GCP project ID |
| `location` | `str` | `"global"` | Vertex AI region |
| `model_id` | `str` | `"gemini-2.5-flash"` | Model identifier |
| `thinking_budget` | `int` | `0` | Reasoning steps allowed |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_workers` | `int` | `10` | Parallel processing workers |

## Authentication Setup

### Prerequisites
1. **Google Cloud Project**: Active GCP project with Vertex AI enabled
2. **Authentication**: One of the following:
   - Application Default Credentials (ADC)
   - Service Account Key
   - Google Cloud SDK authentication

### Setup Steps

1. **Install Google Cloud SDK** (if not already installed):
   ```bash
   # Follow instructions at: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate**:
   ```bash
   # Option 1: User authentication
   gcloud auth application-default login
   
   # Option 2: Service account (recommended for production)
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
   ```

3. **Enable Vertex AI API**:
   ```bash
   gcloud services enable aiplatform.googleapis.com --project=your-project-id
   ```

## Migration from API Key

### Before (API Key)
```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    api_key="your-api-key",
    language_model_type=lx.inference.GeminiLanguageModel
)
```

### After (Vertex AI)
```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    project="your-gcp-project-id",
    location="global",
    language_model_type=lx.inference.GeminiVertexLanguageModel
)
```

## Error Handling

### Common Errors and Solutions

1. **Missing Project ID**:
   ```
   ValueError: Project ID not provided for Vertex AI
   ```
   **Solution**: Provide the `project` parameter

2. **Authentication Error**:
   ```
   google.auth.exceptions.DefaultCredentialsError
   ```
   **Solution**: Set up authentication (see Authentication Setup)

3. **Mutually Exclusive Parameters**:
   ```
   ValueError: Both api_key and project parameters are provided
   ```
   **Solution**: Use either `api_key` OR `project`, not both

## Performance Considerations

### Thinking Budget
- **Low values (0-100)**: Fast responses, basic reasoning
- **Medium values (100-1000)**: Balanced performance and reasoning
- **High values (1000+)**: Deep reasoning, slower responses

### Regional Deployment
- **Global**: Default, automatically routed
- **Regional**: Lower latency for specific regions
- **Multi-region**: Consider for high availability

## Examples

See the complete example in `examples/vertex_ai_example.py` for:
- Basic Vertex AI usage
- Advanced configuration
- Error handling
- Comparison with API key approach

## Troubleshooting

### Debugging Authentication
```python
# Test authentication
from google.auth import default
credentials, project = default()
print(f"Authenticated project: {project}")
```

### Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Benefits of Vertex AI

1. **Enterprise Security**: Leverage Google Cloud's security model
2. **Cost Management**: Better cost tracking and budgeting
3. **Scalability**: Enterprise-grade scaling and reliability
4. **Integration**: Seamless integration with other GCP services
5. **Compliance**: Meet enterprise compliance requirements

## Next Steps

1. **Try the Example**: Run `examples/vertex_ai_example.py`
2. **Set Up Authentication**: Configure your GCP credentials
3. **Test Integration**: Use the test script to verify setup
4. **Migrate Gradually**: Move from API keys to Vertex AI incrementally

For more information, see the [Google Cloud Vertex AI documentation](https://cloud.google.com/vertex-ai/docs).
