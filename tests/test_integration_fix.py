"""Integration test to verify the fix works."""

import os
import sys
from langextract import factory
import langextract as lx


def create_extract_example():
    """Create a sample extraction example as required by LangExtract."""
    examples = [
        lx.data.ExampleData(
            text="iPhone 14 Pro Max costs $1099 and has 256GB storage capacity.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="product_info",
                    extraction_text="iPhone 14 Pro Max costs $1099 and has 256GB storage capacity.",
                    attributes={
                        "product_name": "iPhone 14 Pro Max",
                        "price": "$1099",
                        "storage": "256GB"
                    },
                )
            ],
        )
    ]
    return examples


def test_fixed_reasoning_effort():
    """Test the original failing case now works."""

    # Your original configuration that was failing
    config = factory.ModelConfig(
        model_id="gpt-5-mini",
        provider_kwargs={
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.3,
            "verbosity": "low",
            "reasoning_effort": "minimal",  # This should now work
        }
    )

    # Create required examples
    examples = create_extract_example()

    try:
        # This was the failing call from the issue - now with examples
        lx.extract(
            text_or_documents="iPhone 15 Pro costs $999 and has 128GB storage",
            prompt_description="Extract product information including name, price, and storage",
            examples=examples,  # Now providing required examples
            config=config,
            fence_output=True,
            use_schema_constraints=False
        )

        print("✅ SUCCESS: reasoning_effort parameter now works correctly!")
        return True

    except Exception as exc:
        if "unexpected keyword argument 'reasoning'" in str(exc):
            print("❌ FAILED: Original issue still exists")
        elif "Examples are required" in str(exc):
            print("❌ FAILED: Examples issue (but reasoning_effort fix is working)")
        else:
            print(f"❌ FAILED: Different error - {exc}")
        return False


def test_without_reasoning_effort():
    """Test that the same call works without reasoning_effort (control test)."""

    config = factory.ModelConfig(
        model_id="gpt-5-mini",
        provider_kwargs={
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.3,
            "verbosity": "low",
            # No reasoning_effort - this should work
        }
    )

    examples = create_extract_example()

    try:
        lx.extract(
            text_or_documents="Samsung Galaxy S24 costs $799 with 128GB storage",
            prompt_description="Extract product information",
            examples=examples,
            config=config,
            fence_output=True,
            use_schema_constraints=False
        )

        print("✅ SUCCESS: Control test (without reasoning_effort) works")
        return True

    except Exception as exc:
        print(f"❌ FAILED: Control test failed - {exc}")
        return False


def main():
    """Main function to run tests."""
    print("Testing GPT-5 reasoning_effort fix...")
    print("=" * 50)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set. Set it to run integration tests.")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Test the fix
    print("1. Testing WITH reasoning_effort (the original failing case):")
    success1 = test_fixed_reasoning_effort()

    print("\n2. Testing WITHOUT reasoning_effort (control test):")
    success2 = test_without_reasoning_effort()

    print("\n" + "=" * 50)
    if success1:
        print("🎉 FIX CONFIRMED: reasoning_effort parameter now works!")
    else:
        print("❌ Fix may need more work")

    if success2:
        print("✅ Control test passed - basic functionality intact")
    else:
        print("⚠️  Control test failed - check basic setup")


if __name__ == "__main__":
    main()
