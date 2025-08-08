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

## Model License

Ollama models come with their own licenses. For example:
- Gemma models: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- Llama models: [Meta Llama License](https://llama.meta.com/llama-downloads/)

Please review the license for any model you use.
