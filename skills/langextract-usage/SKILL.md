---
name: langextract-usage
description: How to use LangExtract to extract structured information from text. Use when writing code that calls lx.extract(), building extraction pipelines, defining examples, or troubleshooting alignment issues.
---

# LangExtract Usage

LangExtract extracts structured information from unstructured text using LLMs.
The main entry point is `lx.extract()`. Every extraction needs at least one
example and input text. `prompt_description` is optional but recommended.

## Install

```bash
pip install langextract

# For OpenAI model support:
pip install langextract[openai]
```

## API keys

```bash
# Gemini (checks GEMINI_API_KEY then LANGEXTRACT_API_KEY)
export GEMINI_API_KEY="your_key"

# OpenAI (checks OPENAI_API_KEY then LANGEXTRACT_API_KEY)
export OPENAI_API_KEY="your_key"

# Ollama: no API key needed, just a running Ollama server
```

Auto-routing + env-default resolution is tuned for GPT-style model IDs
(`gpt-4*`, `gpt-5*`). For OpenAI-compatible endpoints or non-GPT IDs,
pass an explicit `ModelConfig` — see `references/providers.md`.

## Basic extraction

```python
import langextract as lx

examples = [
    lx.data.ExampleData(
        text="Patient takes lisinopril 10mg daily for hypertension.",
        extractions=[
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="lisinopril",
                attributes={"dose": "10mg", "frequency": "daily"},
            ),
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text="hypertension",
                attributes={"status": "active"},
            ),
        ],
    )
]

result = lx.extract(
    text_or_documents="Patient is prescribed metformin 500mg twice daily.",
    prompt_description="Extract medications and conditions with attributes.",
    examples=examples,
    model_id="gemini-2.5-flash",
)

for e in result.extractions:
    print(e.extraction_class, e.extraction_text)
    print(f"  char_interval: {e.char_interval}")
    print(f"  attributes: {e.attributes}")
```

See `examples/basic_extraction.py` for a runnable version and
`examples/relationship_extraction.py` for using `extraction_class` +
`attributes` to encode relationships between entities.

## Writing good examples

Examples drive model behavior. Follow these rules:

1. `extraction_text` must be verbatim from the example text, not paraphrased
2. List extractions in order of appearance in the text
3. Each `extraction_class` should be consistent across examples
4. Include attributes that match what you want extracted at runtime

LangExtract can raise prompt-alignment warnings when an example's
`extraction_text` values don't align cleanly to the example's `text`
(failed alignment, or fuzzy/lesser rather than exact). The validator does
not enforce rules 2–4 above — those are guidance to steer the model's
output, not checked invariants.

To fail fast on alignment issues during development, pass these kwargs
directly to `lx.extract()` (both default to permissive):

```python
from langextract.prompt_validation import PromptValidationLevel

result = lx.extract(
    ...,
    prompt_validation_level=PromptValidationLevel.ERROR,  # default: WARNING
    prompt_validation_strict=True,                          # default: False
)
```

See `references/prompt-validation.md` for level semantics and strict-mode
behavior.

## Key parameters

```python
result = lx.extract(
    text_or_documents=text,          # str, URL, or list of Documents
    prompt_description=prompt,        # what to extract (optional)
    examples=examples,                # few-shot examples (required)
    model_id="gemini-2.5-flash",     # model to use
    extraction_passes=1,              # >1 for higher recall (costs more)
    max_char_buffer=1000,             # chunk size
    batch_length=10,                  # chunks per batch
    max_workers=10,                   # parallel workers (provider-dependent)
    context_window_chars=None,        # cross-chunk context for coreference
)
```

- `text_or_documents` accepts a URL string (fetch_urls=True by default).
- For full parallelism, keep `batch_length >= max_workers`. Parallel
  processing via `max_workers` is provider-dependent; some providers
  parallelize batched prompts, while others (such as the current Ollama
  provider) process them sequentially.
- `context_window_chars` includes characters from the previous chunk as
  context for the current one, which helps with coreference and entity
  continuity across chunk boundaries.
- For multiple documents, pass a list of `lx.data.Document` — see
  `examples/multiple_documents.py`.

## Working with results

```python
# Filter to grounded extractions only (have source positions)
grounded = [e for e in result.extractions if e.char_interval]

# Access source position
for e in grounded:
    start = e.char_interval.start_pos
    end = e.char_interval.end_pos
    matched_text = result.text[start:end]

# Save to JSONL
lx.io.save_annotated_documents(
    [result], output_name="results.jsonl", output_dir="."
)

# Visualize from JSONL (shows first document)
html = lx.visualize("results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html, "data"):
        f.write(html.data)  # Jupyter/Colab
    else:
        f.write(html)

# Or visualize an AnnotatedDocument directly
html = lx.visualize(result)
```

## Provider selection

The default is Gemini. For other providers, see `references/providers.md`,
which covers:

- OpenAI (JSON mode; fence behavior auto-configured)
- Ollama (local models, `model_url`)
- `ModelConfig` for advanced provider_kwargs (custom `base_url`, etc.)
- Custom provider plugins via `router.register()`

## Common issues

**Extractions with `char_interval=None`**: the extraction could not be
located in the source text. Common causes include paraphrased output,
alignment misses, or hallucinated entities. Filter with
`[e for e in result.extractions if e.char_interval]`. To tune alignment,
see `references/resolver-params.md`.

**Prompt alignment warnings**: your examples have `extraction_text` that
doesn't match the example text verbatim. Fix the examples, or see
`references/prompt-validation.md` to fail fast during development.

**Slow on long documents**: `extraction_passes > 1` multiplies processing
time. Start with 1 and increase only if recall is insufficient.

**Model selection**: `gemini-2.5-flash` is recommended for most tasks;
`gemini-2.5-pro` for complex reasoning.

## Further reading

- `references/providers.md` — OpenAI, Ollama, ModelConfig, custom plugins
- `references/resolver-params.md` — fuzzy alignment tuning
- `references/prompt-validation.md` — catching example issues early
- `examples/` — runnable scripts for each scenario above
