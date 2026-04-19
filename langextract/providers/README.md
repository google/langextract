# LangExtract Provider System

This directory contains the provider system for LangExtract, which enables support for different Large Language Model (LLM) backends.

**Quick Start**: Use the [provider plugin generator script](../../scripts/create_provider_plugin.py) to create a new provider in minutes:
```bash
python scripts/create_provider_plugin.py MyProvider --with-schema
```

## Architecture Overview

The provider system uses a **router pattern** with **automatic discovery**:

1. **Router** (`router.py`): Maps model ID patterns to provider classes. A legacy `registry.py` alias still imports — new code should use `router` directly.
2. **Factory** (`../factory.py`): Creates provider instances based on model IDs
3. **Providers**: Implement the `BaseLanguageModel` interface (from `langextract.core.base_model`)

### Provider Resolution Flow

```
User Code                    LangExtract                      Provider
─────────                    ───────────                      ────────
    |                             |                              |
    | lx.extract(                 |                              |
    |   model_id="gemini-2.5-flash")                             |
    |─────────────────────────────>                              |
    |                             |                              |
    |                    factory.create_model()                  |
    |                             |                              |
    |                    router.resolve("gemini-2.5-flash")      |
    |                       Pattern match: ^gemini               |
    |                             ↓                              |
    |                       GeminiLanguageModel                  |
    |                             |                              |
    |                    Instantiate provider                    |
    |                             |─────────────────────────────>|
    |                             |                              |
    |                             |       Provider API calls     |
    |                             |<─────────────────────────────|
    |                             |                              |
    |<────────────────────────────                               |
    | AnnotatedDocument           |                              |
```

### Explicit Provider Selection

When multiple providers might support the same model ID, or when you want to use a specific provider, you can explicitly specify the provider:

```python
import langextract as lx

# Method 1: Using factory directly with provider parameter
config = lx.factory.ModelConfig(
    model_id="gpt-4",
    provider="OpenAILanguageModel",  # Explicit provider
    provider_kwargs={"api_key": "..."}
)
model = lx.factory.create_model(config)

# Method 2: Using provider without model_id (uses provider's default)
config = lx.factory.ModelConfig(
    provider="GeminiLanguageModel",  # Will use default gemini-2.5-flash
    provider_kwargs={"api_key": "..."}
)
model = lx.factory.create_model(config)

# Method 3: Auto-detection (when no conflicts exist)
config = lx.factory.ModelConfig(
    model_id="gemini-2.5-flash"  # Provider auto-detected
)
model = lx.factory.create_model(config)
```

Provider names can be:
- Full class name: `"GeminiLanguageModel"`, `"OpenAILanguageModel"`, `"OllamaLanguageModel"`
- Partial match: `"gemini"`, `"openai"`, `"ollama"` (case-insensitive)

## Provider Types

### 1. Core Providers (Always Available)
Ships with langextract, dependencies included:
- **Gemini** (`gemini.py`): Google's Gemini models
- **Ollama** (`ollama.py`): Local models via Ollama

### 2. Built-in Provider with Optional Dependencies
Ships with langextract, but requires extra installation:
- **OpenAI** (`openai.py`): OpenAI's GPT models
  - Code included in package
  - Requires: `pip install langextract[openai]` to install OpenAI SDK
  - Future: May be moved to external plugin package

### 3. External Plugins (Third-party)
Separate packages that extend LangExtract with new providers:
- **Installed separately**: `pip install langextract-yourprovider`
- **Auto-discovered**: Uses Python entry points for automatic registration
- **Zero configuration**: Import langextract and the provider is available
- **Independent updates**: Update providers without touching core

```python
# Install a third-party provider
pip install langextract-yourprovider

# Use it immediately - no imports needed!
import langextract as lx
result = lx.extract(
    text_or_documents="...",
    prompt_description="Extract key information",
    examples=[...],
    model_id="yourmodel-latest",  # Automatically finds the provider
)
```

#### How Plugin Discovery Works

```
1. pip install langextract-yourprovider
   └── Installs package containing:
       • Provider class with @router.register decorator
       • Python entry point pointing to this class

2. import langextract
   └── Loads providers/__init__.py
       └── Plugin loading is lazy (on-demand)

3. lx.extract(model_id="yourmodel-latest")
   └── Triggers plugin discovery via entry points
       └── @router.register decorator fires
           └── Provider patterns added to router
               └── Router matches pattern and uses your provider
```

**Important Notes:**
- Plugin loading is **lazy** - plugins are discovered when first needed
- To manually trigger plugin loading: `lx.providers.load_plugins_once()`
- Set `LANGEXTRACT_DISABLE_PLUGINS=1` to disable plugin loading
- Registry entries are tuples: `(patterns_list, priority_int)`

## How Provider Selection Works

When you call `lx.extract(model_id="gemini-2.5-flash", ...)`, here's what happens:

1. **Factory receives model_id**: "gemini-2.5-flash"
2. **Router searches patterns**: Each provider registers regex patterns
3. **First match wins**: Returns the matching provider class
4. **Provider instantiated**: With model_id and any kwargs
5. **Inference runs**: Using the selected provider

### Pattern Registration Example

```python
from langextract.core import base_model
from langextract.providers import router

# Gemini provider registration:
@router.register(
    r'^GeminiLanguageModel$',  # Explicit: model_id="GeminiLanguageModel"
    r'^gemini',                # Prefix: model_id="gemini-2.5-flash"
    r'^palm'                   # Legacy: model_id="palm-2"
)
class GeminiLanguageModel(base_model.BaseLanguageModel):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        # Initialize Gemini client
        ...

    def infer(self, batch_prompts, **kwargs):
        # Call Gemini API
        ...
```

## Usage Examples

### Using Default Provider Selection
```python
import langextract as lx

# Automatically selects Gemini provider
result = lx.extract(
    text_or_documents="...",
    prompt_description="Extract key information",
    examples=[...],
    model_id="gemini-2.5-flash",
)
```

### Passing Parameters to Providers

Parameters flow from `lx.extract()` to providers through several mechanisms:

```python
# 1. Common parameters handled by lx.extract itself:
result = lx.extract(
    text_or_documents="Your document",
    model_id="gemini-2.5-flash",
    prompt_description="Extract key facts",
    examples=[...],             # Used for few-shot prompting (required)
    max_workers=4,              # Parallel processing (provider-dependent)
    max_char_buffer=3000,       # Document chunking
)

# 2. Provider-specific parameters passed via **kwargs:
result = lx.extract(
    text_or_documents="Your document",
    model_id="gemini-2.5-flash",
    prompt_description="Extract entities",
    examples=[...],
    # These go directly to the Gemini provider:
    temperature=0.7,          # Sampling temperature
    api_key="your-key",      # Override environment variable
    max_output_tokens=1000,  # Token limit
)
```

### Using the Factory for Advanced Control
```python
# When you need explicit provider selection or advanced configuration
from langextract import factory

# Specify both model and provider (useful when multiple providers support same model)
config = factory.ModelConfig(
    model_id="gemma2:2b",
    provider="OllamaLanguageModel",  # Explicitly use Ollama
    provider_kwargs={
        "model_url": "http://localhost:11434"
    }
)
model = factory.create_model(config)
```

### Direct Provider Usage
```python
import langextract as lx

# Direct import if you prefer (optional)
from langextract.providers.gemini import GeminiLanguageModel

model = GeminiLanguageModel(
    model_id="gemini-2.5-flash",
    api_key="your-key"
)
outputs = model.infer(["prompt1", "prompt2"])
```

## Creating a New Provider

**📁 Complete Example**: See [examples/custom_provider_plugin/](../../examples/custom_provider_plugin/) for a fully-functional plugin template with testing and documentation.

### Quick Start Checklist

Creating a provider plugin? Follow this checklist:

#### ☐ **1. Setup Package Structure**
```
langextract-yourprovider/
├── pyproject.toml              # Package config with entry point
├── README.md                    # Documentation
├── LICENSE                      # License file
└── langextract_yourprovider/   # Package directory
    ├── __init__.py             # Exports provider class
    ├── provider.py             # Provider implementation
    └── schema.py               # (Optional) Custom schema
```

#### ☐ **2. Configure Entry Point** (`pyproject.toml`)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "langextract-yourprovider"
version = "0.1.0"
dependencies = ["langextract>=1.0.0"]

[project.entry-points."langextract.providers"]
yourprovider = "langextract_yourprovider:YourProviderLanguageModel"
```

#### ☐ **3. Implement Provider** (`provider.py`)
- [ ] Import required modules
- [ ] Add `@router.register()` decorator with patterns
- [ ] Inherit from `base_model.BaseLanguageModel`
- [ ] Implement `__init__()` method
- [ ] Implement `infer()` method returning `ScoredOutput` objects
- [ ] Export class from `__init__.py`

#### ☐ **4. (Optional) Add Schema Support** (`schema.py`)
- [ ] Create schema class inheriting from `langextract.core.schema.BaseSchema`
- [ ] Implement `from_examples()` class method
- [ ] Implement `to_provider_config()` method
- [ ] Add `get_schema_class()` to provider
- [ ] Handle schema in provider's `__init__()` and `infer()`

#### ☐ **5. Testing**
- [ ] Install plugin with `pip install -e .`
- [ ] Test that your provider loads and handles basic inference
- [ ] Verify schema support works (if implemented)

#### ☐ **6. Documentation**
- [ ] Document supported model IDs and patterns
- [ ] List required environment variables
- [ ] Provide usage examples
- [ ] Document any provider-specific parameters

#### ☐ **7. Distribution & Community**
- [ ] Test installation with `pip install -e .`
- [ ] Build package with `python -m build`
- [ ] Test in clean environment
- [ ] Publish to PyPI with `twine upload dist/*`
- [ ] Share your provider by opening an issue on [LangExtract GitHub](https://github.com/google/langextract/issues) to get feedback and help others discover it
- [ ] Consider submitting a PR to add your provider to [COMMUNITY_PROVIDERS.md](../../COMMUNITY_PROVIDERS.md)

### Option 1: External Plugin (Recommended)

External plugins are the recommended approach for adding new providers. They're easy to maintain, distribute, and don't require changes to the core package.

#### For Users (Installing an External Plugin)
Simply install the plugin package:
```bash
pip install langextract-yourprovider
# That's it! The provider is now available in langextract
```

#### For Developers (Creating an External Plugin)

1. Create a new package:
```
langextract-myprovider/
├── pyproject.toml
├── README.md
└── langextract_myprovider/
    └── __init__.py
```

2. Configure entry point in `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "langextract-myprovider"
version = "0.1.0"
dependencies = ["langextract>=1.0.0", "your-sdk"]

[project.entry-points."langextract.providers"]
# Pattern 1: Register the class directly
myprovider = "langextract_myprovider:MyProviderLanguageModel"

# Pattern 2: Register a module that self-registers
# myprovider = "langextract_myprovider"
```

3. Implement your provider:
```python
# langextract_myprovider/__init__.py
import os

from langextract.core import base_model, types
from langextract.providers import router

@router.register(r'^mymodel', r'^custom', priority=10)
class MyProviderLanguageModel(base_model.BaseLanguageModel):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('MYPROVIDER_API_KEY')
        # Initialize your client
        self.client = MyProviderClient(api_key=self.api_key)

    def infer(self, batch_prompts, **kwargs):
        # Implement inference
        for prompt in batch_prompts:
            result = self.client.generate(prompt, **kwargs)
            yield [types.ScoredOutput(score=1.0, output=result)]
```

**Pattern Registration Explained:**
- The `@router.register` decorator patterns (e.g., `r'^mymodel'`, `r'^custom'`) define which model IDs your provider supports
- When users call `lx.extract(model_id="mymodel-3b")`, the router matches against these patterns
- Your provider will handle any model_id starting with "mymodel" or "custom"
- Users can explicitly select your provider using its class name:
  ```python
  config = lx.factory.ModelConfig(provider="MyProviderLanguageModel")
  # Or partial match: provider="myprovider" (matches class name)
  ```

4. Publish your package to PyPI:
```bash
pip install build twine
python -m build
twine upload dist/*
```

Now users can install and use your provider with just `pip install langextract-myprovider`!

### Adding Schema Support

Schemas enable structured output with strict JSON constraints. Here's how to add schema support to your provider:

#### 1. Create a Schema Class

```python
# langextract_myprovider/schema.py
from langextract.core import schema

class MyProviderSchema(schema.BaseSchema):
    def __init__(self, schema_dict: dict):
        self._schema_dict = schema_dict

    @property
    def schema_dict(self) -> dict:
        return self._schema_dict

    @classmethod
    def from_examples(cls, examples_data, attribute_suffix="_attributes"):
        """Build schema from example extractions."""
        # Analyze examples to determine structure
        extraction_types = {}
        for example in examples_data:
            for extraction in example.extractions:
                class_name = extraction.extraction_class
                if class_name not in extraction_types:
                    extraction_types[class_name] = set()
                if extraction.attributes:
                    extraction_types[class_name].update(extraction.attributes.keys())

        # Build JSON schema
        schema_dict = {
            "type": "object",
            "properties": {
                "extractions": {
                    "type": "array",
                    "items": {"type": "object"}  # Simplified
                }
            }
        }
        return cls(schema_dict)

    def to_provider_config(self) -> dict:
        """Convert to provider-specific configuration."""
        return {
            "response_schema": self._schema_dict,
            "structured_output": True
        }

    @property
    def requires_raw_output(self) -> bool:
        """Return True if provider emits raw JSON without fence markers."""
        return True
```

#### 2. Update Your Provider

```python
# langextract_myprovider/provider.py
from langextract.core import base_model, types

class MyProviderLanguageModel(base_model.BaseLanguageModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__()
        self.model_id = model_id
        # Schema config will be in kwargs when use_schema_constraints=True
        self.response_schema = kwargs.get('response_schema')
        self.structured_output = kwargs.get('structured_output', False)

    @classmethod
    def get_schema_class(cls):
        """Tell LangExtract about our schema support."""
        from langextract_myprovider.schema import MyProviderSchema
        return MyProviderSchema

    def apply_schema(self, schema_instance):
        """Apply or clear schema configuration."""
        super().apply_schema(schema_instance)
        if schema_instance:
            config = schema_instance.to_provider_config()
            self.response_schema = config.get('response_schema')
            self.structured_output = config.get('structured_output', False)
        else:
            self.response_schema = None
            self.structured_output = False

    def infer(self, batch_prompts, **kwargs):
        for prompt in batch_prompts:
            # Use schema in API call if available
            api_params = {}
            if self.response_schema:
                api_params['response_schema'] = self.response_schema

            result = self.client.generate(prompt, **api_params)
            yield [types.ScoredOutput(score=1.0, output=result)]
```

#### 3. Schema Usage

When users set `use_schema_constraints=True`, LangExtract will:
1. Call your provider's `get_schema_class()`
2. Use `from_examples()` to build a schema from provided examples
3. Call `to_provider_config()` to get provider-specific kwargs
4. Pass these kwargs to your provider's `__init__()`
5. Your provider uses the schema for structured output

### Option 2: Built-in Provider (Requires Core Team Approval)

**⚠️ Note**: Adding a provider to the core package requires:
- Significant community demand and support
- Commitment to long-term maintenance
- Approval from the LangExtract maintainers
- A pull request to the main repository

This approach should only be used for providers that benefit a large portion of the user base.

1. Create your provider file:
```python
# langextract/providers/myprovider.py
from langextract.core import base_model
from langextract.providers import router

@router.register(r'^mymodel', r'^custom')
class MyProviderLanguageModel(base_model.BaseLanguageModel):
    # Implementation same as above
```

2. Import it in `providers/__init__.py`:
```python
# In langextract/providers/__init__.py
from langextract.providers import myprovider  # noqa: F401
```

3. Submit a pull request with:
   - Provider implementation
   - Comprehensive tests
   - Documentation
   - Justification for inclusion in core

## Environment Variables

The factory automatically resolves API keys from environment:

| Provider | Environment Variables (in priority order) |
|----------|------------------------------------------|
| Gemini   | `GEMINI_API_KEY`, `LANGEXTRACT_API_KEY` |
| OpenAI   | `OPENAI_API_KEY`, `LANGEXTRACT_API_KEY` |
| Ollama   | `OLLAMA_BASE_URL` (default: http://localhost:11434) |

## Design Principles

1. **Zero Configuration**: Providers auto-register when imported
2. **Extensible**: Easy to add new providers without modifying core
3. **Lazy Loading**: Optional dependencies only loaded when needed
4. **Explicit Control**: Users can force specific providers when needed
5. **Pattern Priority**: All patterns have equal priority (0) by default

## Common Issues

### Provider Not Found
```python
InferenceConfigError: No provider registered for model_id='unknown-model'
```
**Solution**: Check available patterns with `langextract.providers.router.list_entries()`

### Plugin Not Loading
```python
# Your plugin isn't being discovered
```
**Solutions**:
1. Manually trigger loading: `lx.providers.load_plugins_once()`
2. Check entry points are installed: `pip show -f your-package`
3. Verify no typos in `pyproject.toml` entry point
4. Ensure package is installed: `pip list | grep your-package`

### Missing Dependencies
```python
InferenceConfigError: OpenAI provider requires openai package
```
**Solution**: Install optional dependencies: `pip install langextract[openai]`

### Schema Not Working
```python
# Schema constraints not being applied
```
**Solutions**:
1. Ensure provider implements `get_schema_class()`
2. Check `use_schema_constraints=True` is set
3. Verify schema's `requires_raw_output` returns `True`
4. Test schema creation with `Schema.from_examples(examples)`

### Pattern Conflicts
```python
# Multiple providers match the same model_id
```
**Solution**: Use explicit provider selection:
```python
config = lx.factory.ModelConfig(
    model_id="model-name",
    provider="YourProviderClass",  # Explicit selection
)
```
