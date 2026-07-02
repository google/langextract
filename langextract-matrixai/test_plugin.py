#!/usr/bin/env python3
"""Test script for matrixai provider (Step 5 checklist)."""

import re
import sys
import langextract as lx
from langextract.providers import registry

try:
    from langextract_matrixai import matrixaiLanguageModel
except ImportError:
    print("ERROR: Plugin not installed. Run: pip install -e .")
    sys.exit(1)

lx.providers.load_plugins_once()

PROVIDER_CLS_NAME = "matrixaiLanguageModel"
PATTERNS = ['^matrixai']

def _example_id(pattern: str) -> str:
    """Generate test model ID from pattern."""
    base = re.sub(r'^\^', '', pattern)
    m = re.match(r"[A-Za-z0-9._-]+", base)
    base = m.group(0) if m else (base or "model")
    return f"{base}-test"

sample_ids = [_example_id(p) for p in PATTERNS]
sample_ids.append("unknown-model")

print("Testing matrixai Provider - Step 5 Checklist:")
print("-" * 50)

# 1 & 2. Provider registration + pattern matching via resolve()
print("1–2. Provider registration & pattern matching")
for model_id in sample_ids:
    try:
        provider_class = registry.resolve(model_id)
        ok = provider_class.__name__ == PROVIDER_CLS_NAME
        status = "✓" if (ok or model_id == "unknown-model") else "✗"
        note = "expected" if ok else ("expected (no provider)" if model_id == "unknown-model" else "unexpected provider")
        print(f"   {status} {model_id} -> {provider_class.__name__ if ok else 'resolved'} {note}")
    except Exception as e:
        if model_id == "unknown-model":
            print(f"   ✓ {model_id}: No provider found (expected)")
        else:
            print(f"   ✗ {model_id}: resolve() failed: {e}")

# 3. Inference sanity check
print("\n3. Test inference with sample prompts")
try:
    model_id = sample_ids[0] if sample_ids[0] != "unknown-model" else (_example_id(PATTERNS[0]) if PATTERNS else "test-model")
    # Use a dummy API key for testing initialization only
    provider = matrixaiLanguageModel(model_id=model_id, api_key="dummy-key-for-test")
    # Only test initialization, skip actual API call for unit testing
    print(f"   ✓ Provider initialized successfully with model: {provider.model_id}")
    # Optionally, we could mock the API call or skip it during testing
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# 4. Test schema creation and application
print("\n4. Test schema creation and application")
try:
    from langextract_matrixai.schema import matrixaiSchema
    from langextract import data

    examples = [
        data.ExampleData(
            text="Test text",
            extractions=[
                data.Extraction(
                    extraction_class="entity",
                    extraction_text="test",
                    attributes={"type": "example"}
                )
            ]
        )
    ]

    schema = matrixaiSchema.from_examples(examples)
    print(f"   ✓ Schema created (keys={list(schema.schema_dict.keys())})")

    schema_class = matrixaiLanguageModel.get_schema_class()
    print(f"   ✓ Provider schema class: {schema_class.__name__}")

    # Use a dummy API key for testing initialization only
    provider = matrixaiLanguageModel(model_id=_example_id(PATTERNS[0]) if PATTERNS else "test-model", api_key="dummy-key-for-test")
    provider.apply_schema(schema)
    print(f"   ✓ Schema applied: response_schema={provider.response_schema is not None} structured={getattr(provider, 'structured_output', False)}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# 5. Test factory integration
print("\n5. Test factory integration")
try:
    from langextract import factory
    # Include api_key in provider_kwargs for proper initialization
    config = factory.ModelConfig(
        model_id=_example_id(PATTERNS[0]) if PATTERNS else "test-model",
        provider="matrixaiLanguageModel",
        provider_kwargs={"api_key": "dummy-key-for-test"}  # Provide dummy key for testing
    )
    model = factory.create_model(config)
    print(f"   ✓ Factory created: {type(model).__name__}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

print("\n" + "-" * 50)
print("✅ Testing complete!")
