        # LangExtract VLLM Provider

A provider plugin for LangExtract that supports VLLM models.

## Installation

```bash
pip install -e .
```

## Supported Model IDs

- `vllm*`: Models matching pattern ^vllm

## Environment Variables

- `VLLM_API_KEY`: API key for authentication

## Usage

```python
import langextract as lx

result = lx.extract(
    text="Your document here",
    model_id="vllm-model",
    prompt_description="Extract entities",
    examples=[...]
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

## License

Apache License 2.0
