"""Mock test to verify the fix works without API calls."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langextract.providers.openai import OpenAILanguageModel

def test_reasoning_effort_fix_with_mock():
    """Comprehensive mock test to verify the fix."""
    
    # Test the exact scenario from issue #237
    with patch('openai.OpenAI') as mock_openai_class:
        # Setup mock client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"product_name": "iPhone 15 Pro", "price": "$999"}'
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create model with the problematic configuration
        model = OpenAILanguageModel(
            model_id="gpt-5-mini",
            api_key="test-key",
            temperature=0.3,
            reasoning_effort="minimal",  # This was causing the original error
            verbosity="low"
        )
        
        # Make inference call
        prompts = ["Extract product info: iPhone 15 Pro costs $999"]
        results = list(model.infer(prompts))
        
        # Verify the API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        
        # Key verification: reasoning_effort should be preserved, not transformed to 'reasoning'
        print("✅ VERIFICATION RESULTS:")
        print(f"   - reasoning_effort in API call: {'reasoning_effort' in call_kwargs}")
        print(f"   - reasoning_effort value: {call_kwargs.get('reasoning_effort')}")
        print(f"   - incorrect 'reasoning' present: {'reasoning' in call_kwargs}")
        print(f"   - verbosity in API call: {'verbosity' in call_kwargs}")
        print(f"   - model called: {call_kwargs.get('model')}")
        
        # Assertions that prove the fix works
        assert "reasoning_effort" in call_kwargs, "reasoning_effort should be passed through"
        assert call_kwargs["reasoning_effort"] == "minimal", "reasoning_effort value should be preserved"
        assert "reasoning" not in call_kwargs, "Should NOT transform to nested 'reasoning' dict"
        assert "verbosity" in call_kwargs, "verbosity should be supported for GPT-5"
        assert call_kwargs["model"] == "gpt-5-mini", "Correct model should be used"
        
        print("\n🎉 FIX CONFIRMED: Original issue #237 is resolved!")
        print("   - reasoning_effort parameter is now handled correctly")
        print("   - No more 'unexpected keyword argument reasoning' error")
        print("   - GPT-5 models properly supported")
        
        return True

def test_backward_compatibility():
    """Ensure non-GPT-5 models still work correctly."""
    
    with patch('openai.OpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test with GPT-4 (should remove reasoning_effort)
        model = OpenAILanguageModel(
            model_id="gpt-4o-mini",
            api_key="test-key",
            reasoning_effort="minimal"  # Should be removed for non-GPT-5
        )
        
        list(model.infer(["Test prompt"]))
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        
        # For non-GPT-5 models, reasoning_effort should be removed
        assert "reasoning_effort" not in call_kwargs, "Non-GPT-5 models should not get reasoning_effort"
        
        print("✅ BACKWARD COMPATIBILITY: Non-GPT-5 models work correctly")
        return True

if __name__ == "__main__":
    print("Testing GPT-5 reasoning_effort fix with mocks...")
    print("="*60)
    
    success1 = test_reasoning_effort_fix_with_mock()
    success2 = test_backward_compatibility()
    
    if success1 and success2:
        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE FIX VERIFICATION COMPLETE!")
        print("✅ Original issue #237 is fully resolved")
        print("✅ GPT-5 models now handle reasoning_effort correctly") 
        print("✅ Backward compatibility maintained")
        print("✅ Ready for production use")
    else:
        print("❌ Some tests failed")
