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

"""Example demonstrating the use of DashScope provider with LangExtract."""

import os
from typing import List

from pydantic import BaseModel, Field

from langextract.extraction import extract_schema
from langextract.providers import dashscope


# Define a schema for extraction
class Medication(BaseModel):
    """A medication prescribed to a patient."""

    name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage of the medication")
    frequency: str = Field(..., description="How often the medication is taken")
    duration: str = Field(..., description="How long the medication is taken")


class PatientMedications(BaseModel):
    """Medications prescribed to a patient."""

    patient_name: str = Field(..., description="Name of the patient")
    medications: List[Medication] = Field(
        ..., description="List of medications prescribed to the patient"
    )


# Sample text to extract information from
CLINICAL_NOTE = """
Patient Name: John Doe
Date: 2024-05-15

Assessment and Plan:
Mr. Doe is a 45-year-old male with a history of hypertension and type 2 diabetes.

Current Medications:
1. Lisinopril 10mg tablets: Take 1 tablet by mouth daily for high blood pressure.
   Continue for 6 months.
2. Metformin 500mg tablets: Take 1 tablet by mouth twice daily with meals for diabetes.
   Continue indefinitely.
3. Aspirin 81mg tablets: Take 1 tablet by mouth daily for cardiovascular prophylaxis.
   Continue indefinitely.

Follow-up: Return in 3 months for blood pressure and blood sugar monitoring.
"""


def main():
    """Main function to demonstrate DashScope extraction."""
    # Get API key from environment variable
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("Please set DASHSCOPE_API_KEY environment variable.")
        print("Example: export DASHSCOPE_API_KEY='your-api-key'")
        return

    # Method 1: Using extract_schema with model parameter
    print("=== Using extract_schema with model parameter ===")
    try:
        results = extract_schema(
            CLINICAL_NOTE,
            schema_type=PatientMedications,
            model="qwen3-72b-instruct",
        )

        for result in results:
            print(f"\nPatient: {result.patient_name}")
            print("Medications:")
            for med in result.medications:
                print(f"- {med.name}: {med.dosage}, {med.frequency}, {med.duration}")
    except Exception as e:
        print(f"Error with extract_schema: {e}")

    # Method 2: Directly using DashScopeLanguageModel
    print("\n=== Using DashScopeLanguageModel directly ===")
    try:
        # Create model instance
        model = dashscope.DashScopeLanguageModel(
            model_id="qwen3-72b-instruct",
            api_key=api_key,
        )

        # Use the model for extraction
        results = extract_schema(
            CLINICAL_NOTE,
            schema_type=PatientMedications,
            model=model,
        )

        for result in results:
            print(f"\nPatient: {result.patient_name}")
            print("Medications:")
            for med in result.medications:
                print(f"- {med.name}: {med.dosage}, {med.frequency}, {med.duration}")
    except Exception as e:
        print(f"Error with DashScopeLanguageModel: {e}")


if __name__ == "__main__":
    main()
