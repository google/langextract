# Ollama Examples

This directory contains examples for using LangExtract with Ollama for local LLM inference.

For setup instructions and documentation, see the [main README's Ollama section](../../README.md#using-local-llms-with-ollama).

## Quick Reference

**Local setup:**
```bash
ollama pull gemma2:2b
python quickstart.py
```

**Docker setup:**
```bash
docker-compose up
```

## Files

- `quickstart.py` - Basic extraction example with configurable model
- `docker-compose.yml` - Production-ready Docker setup with health checks
- `Dockerfile` - Container definition for LangExtract

## Configuration Options

### Timeout Settings

For slower models or large prompts, you may need to increase the timeout (default: 120 seconds):

```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="llama3.1:70b",  # Larger model may need more time
    timeout=300,  # 5 minutes
    model_url="http://localhost:11434",
)
```

Or using ModelConfig:

```python
config = lx.factory.ModelConfig(
    model_id="llama3.1:70b",
    provider_kwargs={
        "model_url": "http://localhost:11434",
        "timeout": 300,  # 5 minutes
    }
)
```

## Model License

Ollama models come with their own licenses. For example:
- Gemma models: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- Llama models: [Meta Llama License](https://llama.meta.com/llama-downloads/)

Please review the license for any model you use.
