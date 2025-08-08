# LangExtract Provider System

This directory contains the provider system for LangExtract, which enables support for different Large Language Model (LLM) backends.

## Architecture Overview

The provider system uses a **registry pattern** with **automatic discovery**:

1. **Registry** (`registry.py`): Maps model ID patterns to provider classes
2. **Factory** (`../factory.py`): Creates provider instances based on model IDs
3. **Providers**: Implement the `BaseLanguageModel` interface

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
    |                    registry.resolve("gemini-2.5-flash")    |
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
- **Azure OpenAI** (`azure_openai.py`): Azure OpenAI Service
  - Code included in package
  - Inherits from OpenAI provider
  - Requires: `pip install langextract[openai]` to install OpenAI SDK

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
    text="...",
    model_id="yourmodel-latest"  # Automatically finds the provider
)
```

#### How Plugin Discovery Works

```
1. pip install langextract-yourprovider
   └── Installs package containing:
       • Provider class with @lx.providers.registry.register decorator
       • Python entry point pointing to this class

2. import langextract
   └── Loads providers/__init__.py
       └── Discovers and imports plugin via entry points
           └── @lx.providers.registry.register decorator fires
               └── Provider patterns added to registry

3. lx.extract(model_id="yourmodel-latest")
   └── Registry matches pattern and uses your provider
```

## How Provider Selection Works

When you call `lx.extract(model_id="gemini-2.5-flash", ...)`, here's what happens:

1. **Factory receives model_id**: "gemini-2.5-flash"
2. **Registry searches patterns**: Each provider registers regex patterns
3. **First match wins**: Returns the matching provider class
4. **Provider instantiated**: With model_id and any kwargs
5. **Inference runs**: Using the selected provider

### Pattern Registration Example

```python
import langextract as lx

# Gemini provider registration:
@lx.providers.registry.register(
    r'^GeminiLanguageModel$',  # Explicit: model_id="GeminiLanguageModel"
    r'^gemini',                # Prefix: model_id="gemini-2.5-flash"
    r'^palm'                   # Legacy: model_id="palm-2"
)
class GeminiLanguageModel(lx.inference.BaseLanguageModel):
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
    text="...",
    model_id="gemini-2.5-flash"
)

# Automatically selects Azure OpenAI provider
result = lx.extract(
    text="...",
    model_id="azure:gpt-4o",
    language_model_params={
        "azure_endpoint": "https://your-resource.openai.azure.com",
        "api_key": "your-api-key",
        "api_version": "2024-07-01-preview"
    }
)
```

### Passing Parameters to Providers

Parameters flow from `lx.extract()` to providers through several mechanisms:

```python
# 1. Common parameters handled by lx.extract itself:
result = lx.extract(
    text="Your document",
    model_id="gemini-2.5-flash",
    prompt_description="Extract key facts",
    examples=[...],           # Used for few-shot prompting
    num_workers=4,            # Parallel processing
    max_chunk_size=3000,      # Document chunking
)

# 2. Provider-specific parameters passed via **kwargs:
result = lx.extract(
    text="Your document",
    model_id="gemini-2.5-flash",
    prompt_description="Extract entities",
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
    model_id="llama3.2:1b",
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
[project]
name = "langextract-myprovider"
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
import langextract as lx

@lx.providers.registry.register(r'^mymodel', r'^custom')
class MyProviderLanguageModel(lx.inference.BaseLanguageModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__()
        self.model_id = model_id
        # Initialize your client

    def infer(self, batch_prompts, **kwargs):
        # Implement inference
        for prompt in batch_prompts:
            result = self._call_api(prompt)
            yield [lx.inference.ScoredOutput(score=1.0, output=result)]
```

**Pattern Registration Explained:**
- The `@register` decorator patterns (e.g., `r'^mymodel'`, `r'^custom'`) define which model IDs your provider supports
- When users call `lx.extract(model_id="mymodel-3b")`, the registry matches against these patterns
- Your provider will handle any model_id starting with "mymodel" or "custom"
- Users can explicitly select your provider using its class name:
  ```python
  config = lx.factory.ModelConfig(provider="MyProviderLanguageModel")
  # Or partial match: provider="myprovider" (matches class name)

4. Publish your package to PyPI:
```bash
pip install build twine
python -m build
twine upload dist/*
```

Now users can install and use your provider with just `pip install langextract-myprovider`!

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
import langextract as lx

@lx.providers.registry.register(r'^mymodel', r'^custom')
class MyProviderLanguageModel(lx.inference.BaseLanguageModel):
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
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
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
ValueError: No provider registered for model_id='unknown-model'
```
**Solution**: Check available patterns with `registry.list_entries()`

### Missing Dependencies
```python
InferenceConfigError: OpenAI provider requires openai package
```
**Solution**: Install optional dependencies: `pip install langextract[openai]`
