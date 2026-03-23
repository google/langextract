# PR: Add OpenAI Batch API support and batch_size passthrough

## Title

Add OpenAI Batch API support and batch_size passthrough

## Description

This PR adds true provider-native batching support for OpenAI via the OpenAI Batch API, and makes `BaseLanguageModel.infer_batch()` respect `batch_size` by passing it through to provider implementations as a hint.

Fixes/Related to #[issue number]

Choose one: Feature

### Key changes

- `BaseLanguageModel.infer_batch()` now validates `batch_size > 0` and forwards it into `infer(..., batch_size=...)` as a provider hint.
- OpenAI: adds an OpenAI Batch API helper and wires it into the OpenAI provider behind an explicit batch config and threshold.
- Gemini/Ollama: strips `batch_size` from runtime kwargs to avoid it leaking into provider payload/options.

### Files

- Base batching: `langextract/core/base_model.py`
- OpenAI provider + batch wiring: `langextract/providers/openai.py`
- OpenAI Batch helper: `langextract/providers/openai_batch.py`
- Kwarg hygiene: `langextract/providers/gemini.py`, `langextract/providers/ollama.py`
- Tests: `tests/openai_batch_test.py`, `tests/inference_test.py`

### Risks / notes

- OpenAI batch mode is opt-in via config and only triggers above a threshold; non-batch behavior remains unchanged by default.
- Batch jobs are async (polling + timeout) and output ordering is normalized using `custom_id`.

## How Has This Been Tested?

- `pytest -q`
- `./autoformat.sh`
- `pylint --rcfile=.pylintrc langextract/providers/openai_batch.py langextract/providers/openai.py langextract/core/base_model.py langextract/providers/gemini.py langextract/providers/ollama.py`
- `pylint --rcfile=tests/.pylintrc tests/openai_batch_test.py tests/inference_test.py`
- `pre-commit run --files langextract/core/base_model.py langextract/providers/openai.py langextract/providers/openai_batch.py langextract/providers/gemini.py langextract/providers/ollama.py tests/inference_test.py tests/openai_batch_test.py`

## Checklist

- [ ] I have read and acknowledged Google's Open Source Code of conduct.
- [ ] I have read the Contributing page, and I either signed the Google Individual CLA or am covered by my company's Corporate CLA.
- [ ] I have discussed my proposed solution with code owners in the linked issue(s) and we have agreed upon the general approach.
- [ ] I have made any needed documentation changes, or noted in the linked issue(s) that documentation elsewhere needs updating.
- [x] I have added tests, or I have ensured existing tests cover the changes.
- [x] I have followed Google's Python Style Guide and ran `pylint` over the affected code.
