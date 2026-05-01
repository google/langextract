"""Test LangExtract with MiniMax using OpenAI-compatible API - Simple test."""

import os

import langextract as lx
from langextract.factory import create_model
from langextract.factory import ModelConfig

# Get MiniMax API key from environment
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")

if not MINIMAX_API_KEY:
  print("ERROR: MINIMAX_API_KEY not set!")
  exit(1)

# Simple test - just check if model can be created and called
try:
  print("Testing LangExtract with MiniMax (via OpenAI-compatible API)...")
  print(f"API Key: {MINIMAX_API_KEY[:10]}...")
  print(f"Base URL: https://api.minimax.io/v1")
  print()

  # Use factory to create model with custom provider
  config = ModelConfig(
      model_id="MiniMax-M2.5",
      provider="OpenAILanguageModel",
      provider_kwargs={
          "api_key": MINIMAX_API_KEY,
          "base_url": "https://api.minimax.io/v1",
      },
  )
  model = create_model(config)
  print(f"✓ Created model: {type(model).__name__}")
  print(f"✓ Model config: {model.model_id}")
  print(f"✓ Base URL: {model.base_url}")

  # Test a simple inference call
  print("\n✓ Testing model inference...")
  response = model(["Hello, this is a test."], "Say hello")
  print(f"✓ Model responded successfully!")
  print(f"  Response: {response}")

  print("\n" + "=" * 50)
  print("SUCCESS! LangExtract works with MiniMax via OpenAI-compatible API!")
  print("=" * 50)

except Exception as e:
  import traceback

  print(f"\nERROR: {type(e).__name__}: {e}")
  traceback.print_exc()
