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

"""Extract from multiple documents in a single call.

Pass a list of Document objects to process them together. Each result
carries the original `document_id` for downstream joining.
"""

import langextract as lx

examples = [
    lx.data.ExampleData(
        text="Patient reports chest pain on exertion.",
        extractions=[
            lx.data.Extraction(
                extraction_class="symptom",
                extraction_text="chest pain on exertion",
                attributes={"trigger": "exertion"},
            ),
        ],
    )
]

docs = [
    lx.data.Document(
        document_id="note-001",
        text="Patient complains of shortness of breath when climbing stairs.",
    ),
    lx.data.Document(
        document_id="note-002",
        text="Reports occasional dizziness in the morning.",
    ),
]

results = lx.extract(
    text_or_documents=docs,
    prompt_description="Extract symptoms with their triggers.",
    examples=examples,
    model_id="gemini-2.5-flash",
)

for doc_result in results:
  print(f"{doc_result.document_id}: {len(doc_result.extractions)} extractions")
  for e in doc_result.extractions:
    print(f"  - {e.extraction_text} {e.attributes}")
