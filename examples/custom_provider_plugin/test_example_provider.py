#!/usr/bin/env python3
# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple test for the custom provider plugin."""

import os

# Import the provider to trigger registration with LangExtract
# Note: This manual import is only needed when running without installation.
# After `pip install -e .`, the entry point system handles this automatically.
from langextract_provider_example import CustomGeminiProvider  # noqa: F401

import langextract as lx


def main():
  """Test the custom provider."""
  api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")

  if not api_key:
    print("Set GEMINI_API_KEY or LANGEXTRACT_API_KEY to test")
    return

  config = lx.factory.ModelConfig(
      model_id="gemini-2.5-flash",
      provider="CustomGeminiProvider",
      provider_kwargs={"api_key": api_key},
  )
  model = lx.factory.create_model(config)

  print(f"✓ Created {model.__class__.__name__}")

  # Test inference
  prompts = ["Say hello"]
  results = list(model.infer(prompts))

  if results and results[0]:
    print(f"✓ Inference worked: {results[0][0].output[:50]}...")
  else:
    print("✗ No response")


if __name__ == "__main__":
  main()
