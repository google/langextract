#!/usr/bin/env python3
"""
Example demonstrating Gemini Vertex AI integration with langextract.

This example shows how to use the new GeminiVertexLanguageModel for 
extraction tasks using Google Cloud Vertex AI instead of API keys.
"""

import langextract as lx

def main():
    # Example text for extraction
    text = """
    Patient was prescribed Lisinopril 10mg daily for hypertension and 
    Metformin 500mg twice daily for diabetes management. The patient 
    should take Lisinopril in the morning and Metformin with meals.
    """

    # Define examples for medication extraction
    examples = [
        lx.data.ExampleData(
            text="Patient takes Aspirin 100mg daily for heart health.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="medication",
                    extraction_text="Aspirin",
                    attributes={"medication_group": "Aspirin"}
                ),
                lx.data.Extraction(
                    extraction_class="dosage",
                    extraction_text="100mg",
                    attributes={"medication_group": "Aspirin"}
                ),
                lx.data.Extraction(
                    extraction_class="frequency",
                    extraction_text="daily",
                    attributes={"medication_group": "Aspirin"}
                ),
                lx.data.Extraction(
                    extraction_class="condition",
                    extraction_text="heart health",
                    attributes={"medication_group": "Aspirin"}
                ),
            ],
        )
    ]

    prompt = """
    Extract medication information including:
    - medication name
    - dosage 
    - frequency
    - condition being treated
    
    Group related information using the medication_group attribute.
    """

    print("=== Vertex AI Example ===")
    print("Using GeminiVertexLanguageModel with project and location")
    
    try:
        # Example using Vertex AI (replace with your project details)
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id="gemini-2.5-flash",
            project="your-project-id",  # Replace with your GCP project ID
            location="global",  # or your preferred region like "us-central1"
            thinking_budget=0,  # Set to higher values for more reasoning
            language_model_type=lx.inference.GeminiVertexLanguageModel,
            temperature=0.1,
            use_schema_constraints=True,
        )
        
        print(f"Extracted {len(result.extractions)} entities:")
        for extraction in result.extractions:
            print(f"  - {extraction.extraction_class}: {extraction.extraction_text}")
            if extraction.attributes:
                print(f"    Attributes: {extraction.attributes}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo use this example:")
        print("1. Replace 'your-project-id' with your actual GCP project ID")
        print("2. Ensure you have Vertex AI enabled and proper authentication")
        print("3. Make sure you have the required permissions for Vertex AI")

    print("\n=== API Key Example (for comparison) ===")
    print("Using standard GeminiLanguageModel with API key")
    
    try:
        # Example using API key (traditional approach)
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id="gemini-2.5-flash",
            api_key="your-api-key",  # Replace with your API key or set LANGEXTRACT_API_KEY
            language_model_type=lx.inference.GeminiLanguageModel,
            temperature=0.1,
            use_schema_constraints=True,
        )
        
        print(f"Extracted {len(result.extractions)} entities:")
        for extraction in result.extractions:
            print(f"  - {extraction.extraction_class}: {extraction.extraction_text}")
            if extraction.attributes:
                print(f"    Attributes: {extraction.attributes}")
                
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo use this example:")
        print("1. Replace 'your-api-key' with your actual Gemini API key")
        print("2. Or set the LANGEXTRACT_API_KEY environment variable")

    print("\n=== Advanced Vertex AI Features ===")
    print("Demonstrating thinking budget and safety settings")
    
    # Example with advanced Vertex AI features
    try:
        # Advanced configuration with thinking budget and safety settings
        advanced_params = {
            "safety_settings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            ]
        }
        
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id="gemini-2.5-flash",
            project="your-project-id",  # Replace with your GCP project ID
            location="global",
            thinking_budget=1000,  # Allow more reasoning steps
            language_model_type=lx.inference.GeminiVertexLanguageModel,
            temperature=0.1,
            language_model_params=advanced_params,
            use_schema_constraints=True,
        )
        
        print(f"Advanced extraction found {len(result.extractions)} entities")
        
    except ValueError as e:
        print(f"Advanced configuration error: {e}")

if __name__ == "__main__":
    main()
