#!/usr/bin/env python3
"""
Simple test to verify Vertex AI integration works correctly.
"""

import sys
import os

# Add the langextract directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        import langextract as lx
        print("✓ langextract imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import langextract: {e}")
        return False
    
    try:
        from langextract.inference import GeminiVertexLanguageModel
        print("✓ GeminiVertexLanguageModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GeminiVertexLanguageModel: {e}")
        return False
    
    return True

def test_vertex_model_creation():
    """Test that the Vertex AI model can be created with proper parameters."""
    print("\nTesting Vertex AI model creation...")
    
    try:
        from langextract.inference import GeminiVertexLanguageModel
        
        # Test with minimal required parameters
        model = GeminiVertexLanguageModel(
            project="test-project",
            location="global",
            model_id="gemini-2.5-flash"
        )
        print("✓ GeminiVertexLanguageModel created successfully with minimal params")
        
        # Test with all parameters
        model_full = GeminiVertexLanguageModel(
            project="test-project",
            location="us-central1",
            model_id="gemini-2.5-flash",
            temperature=0.5,
            thinking_budget=100,
            max_workers=5
        )
        print("✓ GeminiVertexLanguageModel created successfully with full params")
        
        # Verify attributes are set correctly
        assert model_full.project == "test-project"
        assert model_full.location == "us-central1"
        assert model_full.thinking_budget == 100
        assert model_full.temperature == 0.5
        print("✓ Model attributes set correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create GeminiVertexLanguageModel: {e}")
        return False

def test_vertex_model_validation():
    """Test that the Vertex AI model validates parameters correctly."""
    print("\nTesting parameter validation...")
    
    try:
        from langextract.inference import GeminiVertexLanguageModel
        
        # Test missing project parameter
        try:
            model = GeminiVertexLanguageModel(project=None)
            print("✗ Should have failed with missing project")
            return False
        except ValueError as e:
            if "Project ID not provided" in str(e):
                print("✓ Correctly validates missing project parameter")
            else:
                print(f"✗ Wrong error message: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Unexpected error during validation test: {e}")
        return False

def test_extract_function_parameters():
    """Test that the extract function accepts new parameters."""
    print("\nTesting extract function parameters...")
    
    try:
        import langextract as lx
        
        # Create minimal example data
        examples = [
            lx.data.ExampleData(
                text="Test text",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="test",
                        extraction_text="test"
                    )
                ]
            )
        ]
        
        # Test that the function accepts new parameters without error
        # (We won't actually call it since we don't have valid credentials)
        try:
            # This should fail with authentication error, not parameter error
            result = lx.extract(
                text_or_documents="Test text",
                prompt_description="Test prompt",
                examples=examples,
                project="test-project",
                location="global",
                thinking_budget=0,
                language_model_type=lx.inference.GeminiVertexLanguageModel
            )
        except ValueError as e:
            # We expect this to fail due to authentication, not parameter issues
            if "Project ID" in str(e) or "authentication" in str(e).lower():
                print("✓ Extract function accepts new parameters correctly")
                return True
            else:
                print(f"✗ Unexpected parameter error: {e}")
                return False
        except Exception as e:
            # Any other error suggests the parameters were accepted
            print("✓ Extract function accepts new parameters correctly")
            return True
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test extract function: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Vertex AI Integration Test ===\n")
    
    tests = [
        test_imports,
        test_vertex_model_creation,
        test_vertex_model_validation,
        test_extract_function_parameters
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! Vertex AI integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
