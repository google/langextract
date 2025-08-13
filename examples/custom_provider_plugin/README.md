# Custom Provider Plugin Example

This example demonstrates how to create a custom provider plugin that extends LangExtract with your own model backend.

**Note**: This is an example included in the LangExtract repository for reference. It is not part of the LangExtract package and won't be installed when you `pip install langextract`.

## Structure

```
custom_provider_plugin/
├── pyproject.toml                      # Package configuration and metadata
├── README.md                            # This file
├── langextract_provider_example/        # Package directory
│   ├── __init__.py                     # Package initialization
│   ├── provider.py                     # Custom provider implementation
│   └── schema.py                       # Custom schema implementation (optional)
└── test_example_provider.py            # Test script
```

## Key Components

### Provider Implementation (`provider.py`)

```python
@lx.providers.registry.register(
    r'^gemini',  # Pattern for model IDs this provider handles
)
class CustomGeminiProvider(lx.inference.BaseLanguageModel):
    def __init__(self, model_id: str, **kwargs):
        # Initialize your backend client

    def infer(self, batch_prompts, **kwargs):
        # Call your backend API and return results
```

### Package Configuration (`pyproject.toml`)

```toml
[project.entry-points."langextract.providers"]
custom_gemini = "langextract_provider_example:CustomGeminiProvider"
```

This entry point allows LangExtract to automatically discover your provider.

### Custom Schema Support (`schema.py`)

Providers can optionally implement custom schemas for structured output:

**Flow:** Examples → `from_examples()` → `to_provider_config()` → Provider kwargs → Inference

```python
class CustomProviderSchema(lx.schema.BaseSchema):
    @classmethod
    def from_examples(cls, examples_data, attribute_suffix="_attributes"):
        # Analyze examples to find patterns
        # Build schema based on extraction classes and attributes seen
        return cls(schema_dict)

    def to_provider_config(self):
        # Convert schema to provider kwargs
        return {
            "response_schema": self._schema_dict,
            "enable_structured_output": True
        }

    @property
    def supports_strict_mode(self):
        # True = valid JSON output, no markdown fences needed
        return True
```

Then in your provider:

```python
class CustomProvider(lx.inference.BaseLanguageModel):
    @classmethod
    def get_schema_class(cls):
        return CustomProviderSchema  # Tell LangExtract about your schema

    def __init__(self, **kwargs):
        # Receive schema config in kwargs when use_schema_constraints=True
        self.response_schema = kwargs.get('response_schema')

    def infer(self, batch_prompts, **kwargs):
        # Use schema during API calls
        if self.response_schema:
            config['response_schema'] = self.response_schema
```

## Installation

```bash
# Navigate to this example directory first
cd examples/custom_provider_plugin

# Install in development mode
pip install -e .

# Test the provider (must be run from this directory)
python test_example_provider.py
```

## Usage

Since this example registers the same pattern as the default Gemini provider, you must explicitly specify it:

```python
import langextract as lx

# Create a configured model with explicit provider selection
config = lx.factory.ModelConfig(
    model_id="gemini-2.5-flash",
    provider="CustomGeminiProvider",
    provider_kwargs={"api_key": "your-api-key"}
)
model = lx.factory.create_model(config)

# Note: Passing model directly to extract() is coming soon.
# For now, use the model's infer() method directly or pass parameters individually:
result = lx.extract(
    text_or_documents="Your text here",
    model_id="gemini-2.5-flash",
    api_key="your-api-key",
    prompt_description="Extract key information",
    examples=[...]
)

# Coming soon: Direct model passing
# result = lx.extract(
#     text_or_documents="Your text here",
#     model=model,  # Planned feature
#     prompt_description="Extract key information"
# )
```

## Creating Your Own Provider

1. Copy this example as a starting point
2. Update the provider class name and registration pattern
3. Replace the Gemini implementation with your own backend
4. Update package name in `pyproject.toml`
5. Install and test your plugin

## License

Apache License 2.0
