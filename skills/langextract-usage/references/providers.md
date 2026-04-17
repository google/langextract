# Provider Reference

LangExtract ships with three built-in providers (Gemini, OpenAI, Ollama) and
supports custom providers via a plugin system.

## Gemini (default, recommended)

```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",  # or "gemini-2.5-pro" for complex reasoning
)
```

Env: `GEMINI_API_KEY` (falls back to `LANGEXTRACT_API_KEY`).

## OpenAI

```python
result = lx.extract(
    text_or_documents=text,
    examples=examples,
    prompt_description=prompt,
    model_id="gpt-4o",
)
```

Env: `OPENAI_API_KEY` (falls back to `LANGEXTRACT_API_KEY`).

The OpenAI provider uses JSON mode (`response_format={"type":"json_object"}`)
and reports `requires_fence_output=False`. Leave `fence_output` unset —
`extract()` and the factory/provider layer auto-determine fence behavior
from the provider's schema. The OpenAI provider parallelizes batched
prompts with a `ThreadPoolExecutor` when `max_workers > 1`.

The OpenAI provider does not expose a schema class, so
`use_schema_constraints` is a no-op here. You can omit it (as shown above)
or leave it at its default.

**Auto-routing scope:** the built-in OpenAI provider only auto-matches
GPT-style `model_id`s (`^gpt-4`, `^gpt4\.`, `^gpt-5`, `^gpt5\.`), and the
environment-default API-key lookup keys off `"gpt"` rather than
`"openai"`. For **OpenAI-compatible endpoints** (LiteLLM, local servers,
custom base URLs) or **non-GPT model IDs**, use `ModelConfig` with an
explicit provider and kwargs:

```python
from langextract.factory import ModelConfig

result = lx.extract(
    text_or_documents=text,
    examples=examples,
    prompt_description=prompt,
    config=ModelConfig(
        model_id="my-openai-compatible-model",
        provider="openai",
        provider_kwargs={"api_key": "sk-...", "base_url": "https://..."},
    ),
)
```

## Ollama (local)

No API key needed. Requires a running Ollama server.

```python
result = lx.extract(
    text_or_documents=text,
    examples=examples,
    prompt_description=prompt,
    model_id="gemma2:2b",
    model_url="http://localhost:11434",
)
```

The Ollama provider exposes `FormatModeSchema` for JSON mode. Leave
`fence_output` and `use_schema_constraints` unset so `extract()` and the
factory/provider layer auto-configure from the provider's schema. The
Ollama provider processes batched prompts sequentially — `max_workers > 1`
will not parallelize it.

## ModelConfig (advanced)

For provider-specific kwargs (custom base_url, explicit api_key, etc.):

```python
from langextract.factory import ModelConfig

result = lx.extract(
    text_or_documents=text,
    examples=examples,
    config=ModelConfig(
        model_id="gpt-4o",
        provider_kwargs={"api_key": "your_key", "base_url": "https://..."},
    ),
)
```

ModelConfig fields: `model_id`, `provider` (for disambiguation when multiple
providers match a model_id pattern), `provider_kwargs`.

## Custom provider plugins

Subclass `BaseLanguageModel` and register a regex pattern for the model_ids
your provider should handle:

```python
from langextract.core.base_model import BaseLanguageModel
from langextract.providers import router

@router.register(r"^my-model")
class MyProvider(BaseLanguageModel):
    ...
```

For external packages, declare an entry point in your `pyproject.toml` so
LangExtract discovers the provider at model-creation time:

```toml
[project.entry-points."langextract.providers"]
my-model = "my_package.provider:MyProvider"
```

Plugin loading is **lazy**: `load_plugins_once()` runs the first time a
model is created via the factory, not at package import. Once loaded, any
registered provider whose pattern matches the requested `model_id` is
selected automatically.

`langextract.inference` and `langextract.providers.registry` still exist
as backward-compatibility aliases, but new code should import from
`langextract.core.base_model` and `langextract.providers.router` as shown
above. A complete working example lives at
`examples/custom_provider_plugin/` in the repo.
