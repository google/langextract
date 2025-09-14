"""Tests for GPT-5 reasoning_effort parameter fix."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langextract.providers.openai import OpenAILanguageModel


class TestGPT5ReasoningEffort:
    
    def test_gpt5_reasoning_effort_preserved(self):
        """Test that reasoning_effort is preserved for GPT-5 models."""
        model = OpenAILanguageModel(
            model_id="gpt-5-mini",
            api_key="test-key"
        )
        
        config = {"reasoning_effort": "minimal", "temperature": 0.5}
        normalized = model._normalize_reasoning_params(config)
        
        # Should preserve reasoning_effort for GPT-5 models
        assert "reasoning_effort" in normalized
        assert normalized["reasoning_effort"] == "minimal"
        assert "reasoning" not in normalized
    
    def test_gpt4_reasoning_effort_removed(self):
        """Test that reasoning_effort is removed for non-GPT-5 models."""
        model = OpenAILanguageModel(
            model_id="gpt-4o-mini", 
            api_key="test-key"
        )
        
        config = {"reasoning_effort": "minimal", "temperature": 0.5}
        normalized = model._normalize_reasoning_params(config)
        
        # Should remove reasoning_effort for non-GPT-5 models
        assert "reasoning_effort" not in normalized
        assert normalized["temperature"] == 0.5
    
    def test_gpt5_variants_supported(self):
        """Test all GPT-5 variants support reasoning_effort."""
        variants = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "GPT-5-MINI"]
        
        for variant in variants:
            model = OpenAILanguageModel(
                model_id=variant,
                api_key="test-key" 
            )
            
            config = {"reasoning_effort": "low"}
            normalized = model._normalize_reasoning_params(config)
            
            assert "reasoning_effort" in normalized
            assert normalized["reasoning_effort"] == "low"

    @patch('openai.OpenAI')
    def test_api_call_with_reasoning_effort(self, mock_openai_class):
        """Test that reasoning_effort is passed to OpenAI API correctly."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create model and make inference
        model = OpenAILanguageModel(
            model_id="gpt-5-mini",
            api_key="test-key"
        )
        
        # Process with reasoning_effort
        result = list(model.infer(
            ["Test prompt"], 
            reasoning_effort="minimal",
            verbosity="low"
        ))
        
        # Verify API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        
        assert "reasoning_effort" in call_kwargs
        assert call_kwargs["reasoning_effort"] == "minimal"
        assert "verbosity" in call_kwargs  
        assert call_kwargs["verbosity"] == "low"
        assert "reasoning" not in call_kwargs  # Should not have nested reasoning
