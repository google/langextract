# Prompt Validation Reference

Prompt validation catches example problems (non-verbatim text, alignment
mismatches) early, before expensive extraction runs.

## Levels

```python
from langextract.prompt_validation import PromptValidationLevel

result = lx.extract(
    text_or_documents=text,
    examples=examples,
    prompt_description=prompt,
    model_id="gemini-2.5-flash",
    prompt_validation_level=PromptValidationLevel.ERROR,
    prompt_validation_strict=True,
)
```

- **`OFF`** — skip validation entirely.
- **`WARNING`** (default) — log issues, continue running.
- **`ERROR`** — raise on validation failure.

## Strict mode

When `prompt_validation_strict=True`, non-exact matches (fuzzy alignment,
`accept_match_lesser`) also trigger failures in `ERROR` mode. Without strict
mode, only completely unalignable extractions fail.

## How it aligns with resolver_params

Prompt validation uses the same alignment settings from `resolver_params`.
So if you've tuned `fuzzy_alignment_threshold` or
`fuzzy_alignment_algorithm`, validation honors those same values when
deciding whether an example extraction is aligned.

## When to use each level

- **Development**: `ERROR` + `strict=True` forces you to write clean,
  verbatim examples.
- **Staging/CI**: `ERROR` (non-strict) catches truly broken examples
  without failing on minor fuzzy matches.
- **Production**: `WARNING` or `OFF` — you've already validated offline and
  don't want runtime failures over example drift.

## Common fixes

- **"Extraction text not found in example text"**: your `extraction_text`
  doesn't appear verbatim in the example `text`. Copy-paste the exact
  substring rather than paraphrasing.
- **"Extraction text aligned fuzzily"** (strict only): same issue but with
  minor whitespace/punctuation differences. Fix the example to match
  exactly.
