# Langfuse Integration Example

This example shows how to validate optional Langfuse observability in
LangExtract without changing extraction behavior when observability is disabled.
It demonstrates attaching the observer through `lx.factory.create_model(...)`.

## What it validates

- Langfuse remains optional (`ENABLE_LANGFUSE=false` or missing keys works)
- Extractions still run with no observer
- Generation traces are emitted when observer is enabled
- Token usage is logged when the selected model provider reports usage (accessible via the generation type inside the trace)

## Environment variables

Supported Langfuse variables:
Note: for custom langfuse deployments its best to provide LANGFUSE_BASE_URL.

```bash
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=...   # Optional
LANGFUSE_HOST=...       # Optional fallback for base URL
```

## Run

From repo root:

```bash
pip install -e ".[langfuse]"
python examples/langfuse/test_langfuse_integration.py
```

If Langfuse is disabled or keys are not set, extraction should still succeed and
the script will print that observability is disabled.
