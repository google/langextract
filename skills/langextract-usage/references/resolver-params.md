# resolver_params Reference

`resolver_params` is a dict passed to `lx.extract()` to fine-tune how the
model's output is parsed and aligned to source text. Keys may change across
versions — check the current `extract()` docstring if in doubt.

## Full example

```python
result = lx.extract(
    text_or_documents=text,
    examples=examples,
    prompt_description=prompt,
    model_id="gemini-2.5-flash",
    resolver_params={
        "suppress_parse_errors": True,
        "extraction_index_suffix": "_index",
        "enable_fuzzy_alignment": True,
        "fuzzy_alignment_threshold": 0.75,
        "fuzzy_alignment_algorithm": "lcs",
        "fuzzy_alignment_min_density": 1 / 3,
        "accept_match_lesser": True,
    },
)
```

## Key descriptions

- **`suppress_parse_errors`** (default True in `extract()`): on parse or
  schema errors for a chunk, log a warning and return `[]` for that chunk
  rather than raising. The chunk is effectively dropped; no partial output
  is recovered from the malformed response.
- **`extraction_index_suffix`**: attribute name suffix used to preserve
  extraction ordering (e.g. `"_index"` looks for attributes like
  `medication_index` and sorts extractions by that value).
- **`enable_fuzzy_alignment`** (default True): attempt fuzzy alignment when
  exact substring match fails.
- **`fuzzy_alignment_threshold`** (default 0.75): minimum fraction of
  extraction tokens that must be matched in the source span.
- **`fuzzy_alignment_algorithm`** (default `"lcs"`): algorithm for fuzzy
  alignment. `"lcs"` uses longest-common-subsequence DP (O(n*m²)); `"legacy"`
  uses the older difflib-based approach and is deprecated.
- **`fuzzy_alignment_min_density`** (default 1/3): minimum ratio of matched
  tokens to the span length. Filters out sparse matches where only a few
  tokens align across a very long source span.
- **`accept_match_lesser`**: accept `MATCH_LESSER` alignments — partial
  exact matches where the model's `extraction_text` is longer than the span
  that exactly matched in the source (e.g. the model returned a few extra
  tokens at the edges). This is distinct from fuzzy alignment: the matched
  portion is still an exact substring. Use it for truncation or
  extra-token cases, not as a substitute for lowering
  `fuzzy_alignment_threshold`.

## When to tune these

- **Too many `char_interval=None` extractions**: lower
  `fuzzy_alignment_threshold` (e.g. 0.6). Enable `accept_match_lesser`
  only if the model is returning extraction_text that contains an exact
  source substring plus extra tokens.
- **Fuzzy alignment matching the wrong span**: raise
  `fuzzy_alignment_min_density` (e.g. 0.5) so matches must be denser.
- **Performance-sensitive pipelines on long documents**: stick with `"lcs"`
  and consider lowering `max_char_buffer` rather than disabling fuzzy
  alignment entirely.
