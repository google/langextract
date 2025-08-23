# Custom Provider Plugin Example

This example demonstrates how to create a custom provider plugin that extends LangExtract with your own model backend.

**Note**: This is an example included in the LangExtract repository for reference. It is not part of the LangExtract package and won't be installed when you `pip install langextract`.

**Automated Creation**: Instead of manually copying this example, use the [provider plugin generator script](../../scripts/create_provider_plugin.py):
```bash
python scripts/create_provider_plugin.py MyProvider --with-schema
```
This will create a complete plugin structure with all boilerplate code ready for customization.

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

## Creating Your Own Provider - Step by Step

### 1. Copy and Rename
```bash
# Copy this example directory
cp -r examples/custom_provider_plugin/ ~/langextract-myprovider/

# Rename the package directory
cd ~/langextract-myprovider/
mv langextract_provider_example langextract_myprovider
```

### 2. Update Package Configuration
Edit `pyproject.toml`:
- Change `name = "langextract-myprovider"`
- Update description and author information
- Change entry point: `myprovider = "langextract_myprovider:MyProvider"`

### 3. Modify Provider Implementation
Edit `provider.py`:
- Change class name from `CustomGeminiProvider` to `MyProvider`
- Update `@register()` patterns to match your model IDs
- Replace Gemini API calls with your backend
- Add any provider-specific parameters

### 4. Add Schema Support (Optional)
Edit `schema.py`:
- Rename to `MyProviderSchema`
- Customize `from_examples()` for your extraction format
- Update `to_provider_config()` for your API requirements
- Set `supports_strict_mode` based on your capabilities

### 5. Install and Test
```bash
# Install in development mode
pip install -e .

# Test your provider
python -c "
import langextract as lx
lx.providers.load_plugins_once()
print('Provider registered:', any('myprovider' in str(e) for e in lx.providers.registry.list_entries()))
"
```

### 6. Write Tests
- Test that your provider loads and handles basic inference
- Verify schema support works (if implemented)
- Test error handling for your specific API

### 7. Publish to PyPI and Share with Community
```bash
# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

**Share with the community:**
- Submit a PR to add your provider to the [Community Providers Registry](../../COMMUNITY_PROVIDERS.md)
- Open an issue on [LangExtract GitHub](https://github.com/google/langextract/issues) to announce your provider and get feedback

## Common Pitfalls to Avoid

1. **Forgetting to trigger plugin loading** - Plugins load lazily, use `load_plugins_once()` in tests
2. **Pattern conflicts** - Avoid patterns that conflict with built-in providers
3. **Missing dependencies** - List all requirements in `pyproject.toml`
4. **Schema mismatches** - Test schema generation with real examples
5. **Not handling None schema** - Provider must clear schema when `apply_schema(None)` is called (see provider.py for implementation)

## License

Apache License 2.0
