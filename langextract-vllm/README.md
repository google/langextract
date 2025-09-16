# LangExtract vLLM Provider

A provider plugin for LangExtract that supports vLLM models with high-performance inference using PagedAttention.

## Installation

```bash
pip install langextract-vllm
```

## Supported Model IDs

Model ID using the format:

- **vLLM models**: `vllm:<model_name_or_path>`

Where `<model_name_or_path>` can be:
- HuggingFace model repository (e.g., `meta-llama/Llama-2-7b-chat-hf`)
- Local model path (e.g., `/path/to/model`)

## Usage

### Basic Usage

```python
import langextract as lx

config = lx.factory.ModelConfig(
    model_id="vllm:meta-llama/Llama-2-7b-chat-hf",
    provider="VLLMLanguageModel",  # optional as vllm: will resolve to the model
    provider_kwargs=dict(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        gpu_memory_utilization=0.7,
        max_model_len=1024,
    ),
)

model = lx.factory.create_model(config)

result = lx.extract(
    model=model,
    text_or_documents="Your input text",
    prompt_description="Extract entities",
    examples=[...],
)
```

### Advanced Configuration

```python
import langextract as lx

config = lx.factory.ModelConfig(
    model_id="vllm:mistralai/Mistral-7B-Instruct-v0.1",
    provider="VLLMLanguageModel",
    provider_kwargs=dict(
        temperature=0.8,
        top_p=0.95,
        max_tokens=2048,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        max_workers=4,
        # Additional vLLM engine parameters
        enforce_eager=True,
        disable_custom_all_reduce=True,
    ),
)

model = lx.factory.create_model(config)

result = lx.extract(
    model=model,
    text_or_documents="Your input text",
    prompt_description="Extract named entities and their relationships",
    examples=[...],
)
```

### Multi-GPU Configuration

For large models or high throughput scenarios, you can distribute the model across multiple GPUs:

```python
import langextract as lx

config = lx.factory.ModelConfig(
    model_id="vllm:meta-llama/Llama-2-70b-chat-hf",
    provider="VLLMLanguageModel",
    provider_kwargs=dict(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        # Multi-GPU configuration
        tensor_parallel_size=4,  # Distribute across 4 GPUs
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        # Optimize for multi-GPU setup
        disable_custom_all_reduce=True,
        enforce_eager=True,
        max_workers=8,  # Higher batch size for better GPU utilization
    ),
)

model = lx.factory.create_model(config)

# Process multiple documents in parallel
results = []
documents = ["Document 1", "Document 2", "Document 3", ...]

for doc in documents:
    result = lx.extract(
        model=model,
        text_or_documents=doc,
        prompt_description="Extract key information",
        examples=[...],
    )
    results.append(result)
```

**Multi-GPU Parameters:**
- `tensor_parallel_size`: Number of GPUs to use (must be power of 2: 1, 2, 4, 8)
- `pipeline_parallel_size`: Pipeline parallelism degree (advanced feature)
- `disable_custom_all_reduce=True`: Recommended for multi-GPU stability

### Using Local vLLM Server

Connect to a locally running vLLM server:

```python
import langextract as lx

# First, start your vLLM server:
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-2-7b-chat-hf \
#     --host 0.0.0.0 \
#     --port 8000

config = lx.factory.ModelConfig(
    model_id="vllm:http://localhost:8000/v1",
    provider="VLLMLanguageModel", 
    provider_kwargs=dict(
        temperature=0.7,
        max_tokens=1024,
        # Server connection settings
        timeout=60.0,
    ),
)

model = lx.factory.create_model(config)

result = lx.extract(
    model=model,
    text_or_documents="Your input text",
    prompt_description="Extract information",
    examples=[...],
)
```

**Local Server Benefits:**
- Decouple model serving from client applications
- Share one model instance across multiple clients
- Better resource management and scaling
- Easier deployment in production environments

## Configuration Parameters

### Provider Arguments (`provider_kwargs`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.9 | Top-p sampling parameter |
| `max_tokens` | int | 1024 | Maximum tokens to generate |
| `gpu_memory_utilization` | float | 0.7 | GPU memory utilization ratio (0.0-1.0) |
| `max_model_len` | int | 1024 | Maximum sequence length |
| `max_workers` | int | 1 | Maximum parallel workers |
| `enforce_eager` | bool | True | Disable torch.compile for stability |
| `disable_custom_all_reduce` | bool | True | Disable custom all-reduce operations |
| `tensor_parallel_size` | int | 1 | Number of GPUs for tensor parallelism (1, 2, 4, 8) |
| `pipeline_parallel_size` | int | 1 | Number of GPUs for pipeline parallelism |
| `api_key` | str | None | API key for server connection (use "EMPTY" for vLLM) |
| `timeout` | float | 60.0 | Request timeout in seconds for server connections |

Additional vLLM engine parameters can be passed through `provider_kwargs`. Refer to the [vLLM documentation](https://docs.vllm.ai/) for complete parameter reference.

## Performance Optimization

### Memory Management
- Adjust `gpu_memory_utilization` based on your GPU memory
- Lower `max_model_len` to reduce KV cache memory usage
- Use `max_workers` to control batch processing

### Inference Optimization
- Set `enforce_eager=True` for stability (disables torch.compile)
- Use `disable_custom_all_reduce=True` for multi-GPU setups
- Batch multiple requests for better throughput

### Multi-GPU Optimization
- Use `tensor_parallel_size` for large models that don't fit on single GPU
- Ensure GPU memory is balanced across all devices
- Set `tensor_parallel_size` to power of 2 (1, 2, 4, 8)
- Increase `max_workers` proportionally to GPU count for better utilization

## Requirements

- Python >= 3.10
- CUDA-compatible GPU (for GPU acceleration)
- vLLM >= 0.5.0
- PyTorch >= 2.0.0
- Transformers >= 4.30.0

## Development

```bash
# Install in development mode
uv pip install -e .

# Run tests
uv run test_plugin.py

# Build package
uv build

# Publish to PyPI
uv publish
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `gpu_memory_utilization` or `max_model_len`
2. **Model Loading Errors**: Ensure the model path/repository is correct and accessible
3. **Performance Issues**: Increase `max_workers` for better batching

### Error Messages

- `vLLM library not installed`: Install vLLM with `pip install vllm`
- `InferenceRuntimeError`: Check GPU memory and model compatibility

## License

MIT License - see LICENSE file for details.