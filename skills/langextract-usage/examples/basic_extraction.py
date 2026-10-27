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

"""Extract medications and conditions from a clinical note.

Run: GEMINI_API_KEY=... python basic_extraction.py
"""

import langextract as lx

examples = [
    lx.data.ExampleData(
        text="Patient takes lisinopril 10mg daily for hypertension.",
        extractions=[
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="lisinopril",
                attributes={"dose": "10mg", "frequency": "daily"},
            ),
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text="hypertension",
                attributes={"status": "active"},
            ),
        ],
    )
]

result = lx.extract(
    text_or_documents="Patient is prescribed metformin 500mg twice daily.",
    prompt_description="Extract medications and conditions with attributes.",
    examples=examples,
    model_id="gemini-2.5-flash",
)

for e in result.extractions:
  print(e.extraction_class, e.extraction_text)
  print(f"  char_interval: {e.char_interval}")
  print(f"  attributes: {e.attributes}")
