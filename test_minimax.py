"""Test LangExtract with MiniMax using OpenAI-compatible API."""

import os

import langextract as lx

# Get MiniMax API key from environment
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")

if not MINIMAX_API_KEY:
  print("ERROR: MINIMAX_API_KEY not set!")
  exit(1)

# Sample text to extract from
sample_text = """
Patient John Doe, age 45, was admitted to the hospital on March 15, 2026.
He presented with symptoms of fever, cough, and shortness of breath.
Medical history includes hypertension and diabetes type 2.
Current medications: Metformin 500mg twice daily, Lisinopril 10mg once daily.
"""

# Define extraction instructions
instructions = (
    "Extract patient information including name, age, symptoms, medical"
    " history, and medications."
)

# Try using MiniMax via OpenAI-compatible API
try:
  print("Testing LangExtract with MiniMax (via OpenAI-compatible API)...")
  print(f"API Key: {MINIMAX_API_KEY[:10]}...")
  print()

  result = lx.extract(
      text_or_documents=sample_text,
      prompt_description=instructions,
      model_id="MiniMax-M2.5",  # This won't auto-detect, need to specify provider
      provider="OpenAILanguageModel",
      provider_kwargs={
          "api_key": MINIMAX_API_KEY,
          "base_url": "https://api.minimax.io/v1",
      },
  )

  print("SUCCESS! Extraction result:")
  print(result)

except Exception as e:
  print(f"ERROR: {type(e).__name__}: {e}")
  print()
  print("Let's try a different approach...")
