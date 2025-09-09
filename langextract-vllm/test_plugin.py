#!/usr/bin/env python3
"""Test script for vLLM provider (Step 5 checklist)."""

import re
import sys
import langextract as lx
from langextract.providers import registry

try:
    from langextract_vllm import VLLMLanguageModel
except ImportError:
    print("ERROR: Plugin not installed. Run: pip install -e .")
    sys.exit(1)

lx.providers.load_plugins_once()

PROVIDER_CLS_NAME = "VLLMLanguageModel"
PATTERNS = ["^vllm"]


def _example_id(pattern: str) -> str:
    """Generate test model ID from pattern."""
    base = re.sub(r"^\^", "", pattern)
    m = re.match(r"[A-Za-z0-9._-]+", base)
    base = m.group(0) if m else (base or "model")
    return f"{base}:Qwen/Qwen3-0.6B"


sample_ids = [_example_id(p) for p in PATTERNS]
sample_ids.append("unknown-model")

print("Testing vLLM Provider - Step 5 Checklist:")
print("-" * 50)

# 1 & 2. Provider registration + pattern matching via resolve()
print("1–2. Provider registration & pattern matching")
for model_id in sample_ids:
    try:
        provider_class = registry.resolve(model_id)
        ok = provider_class.__name__ == PROVIDER_CLS_NAME
        status = "✓" if (ok or model_id == "unknown-model") else "✗"
        note = (
            "expected" if ok else ("expected (no provider)" if model_id == "unknown-model" else "unexpected provider")
        )
        print(f"   {status} {model_id} -> {provider_class.__name__ if ok else 'resolved'} {note}")
    except Exception as e:
        if model_id == "unknown-model":
            print(f"   ✓ {model_id}: No provider found (expected)")
        else:
            print(f"   ✗ {model_id}: resolve() failed: {e}")

# 3. Inference sanity check
print("\n3. Test inference with sample prompts")
try:
    model_id = (
        sample_ids[0] if sample_ids[0] != "unknown-model" else (_example_id(PATTERNS[0]) if PATTERNS else "test-model")
    )
    provider = VLLMLanguageModel(model_id=model_id)
    prompts = ["Test prompt 1", "Test prompt 2"]
    results = list(provider.infer(prompts))
    print(f"   ✓ Inference returned {len(results)} results")
    for i, result in enumerate(results):
        try:
            out = result[0].output if result and result[0] else None
            print(f"   ✓ Result {i + 1}: {(out or '')[:60]}...")
        except Exception:
            print(f"   ✗ Result {i + 1}: Unexpected result shape: {result}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# 5. Test factory integration
print("\n5. Test factory integration")
try:
    from langextract import factory
    
    config = factory.ModelConfig(
        model_id=_example_id(PATTERNS[0]) if PATTERNS else "test-model",
        provider="VLLMLanguageModel",
    )
    model = factory.create_model(config)
    print(f"   ✓ Factory created: {type(model).__name__}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

print("\n" + "-" * 50)
print("✅ Testing complete!")