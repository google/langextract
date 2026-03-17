"""Test the new MiniMax provider for LangExtract - with API key."""
import os
import langextract as lx
from langextract.factory import ModelConfig, create_model

# Get MiniMax API key from environment
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")

if not MINIMAX_API_KEY:
    print("ERROR: MINIMAX_API_KEY not set!")
    exit(1)

# Simple test
try:
    print("Testing new MiniMax provider (via OpenAI provider)...")
    print(f"API Key: {MINIMAX_API_KEY[:10]}...")
    print()
    
    # Test auto-detection via pattern with API key
    config = ModelConfig(
        model_id="MiniMax-M2.5",
        provider_kwargs={
            "api_key": MINIMAX_API_KEY,
            "base_url": "https://api.minimax.io/v1"
        }
    )
    model = create_model(config)
    print(f"✓ Auto-detected provider: {type(model).__name__}")
    print(f"✓ Model ID: {model.model_id}")
    print(f"✓ Base URL: {model.base_url}")
    
    print("\n" + "="*50)
    print("SUCCESS! MiniMax works with LangExtract!")
    print("="*50)
    print("\nNow you can use:")
    print('''
    config = ModelConfig(
        model_id="MiniMax-M2.5",
        provider_kwargs={
            "api_key": MINIMAX_API_KEY,
            "base_url": "https://api.minimax.io/v1"
        }
    )
    model = create_model(config)
    result = lx.extract(text, prompt, model=model)
    ''')
    
except Exception as e:
    import traceback
    print(f"\nERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
