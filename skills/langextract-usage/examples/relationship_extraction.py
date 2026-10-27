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

"""Extract characters, emotions, and metaphorical relationships from text.

Demonstrates distinct `extraction_class` values (`character`, `emotion`,
`relationship`) plus `attributes` to encode structured details.
"""

import langextract as lx

examples = [
    lx.data.ExampleData(
        text=(
            "ROMEO. But soft! What light through yonder window breaks? "
            "It is the east, and Juliet is the sun."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"role": "speaker"},
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "wonder", "character": "Romeo"},
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={
                    "type": "metaphor",
                    "source": "Romeo",
                    "target": "Juliet",
                },
            ),
        ],
    )
]

result = lx.extract(
    text_or_documents=(
        "JULIET. O Romeo, Romeo! wherefore art thou Romeo? "
        "Deny thy father and refuse thy name."
    ),
    prompt_description=(
        "Extract characters, emotions, and any metaphorical relationships "
        "between entities."
    ),
    examples=examples,
    model_id="gemini-2.5-flash",
)

for e in result.extractions:
  print(f"[{e.extraction_class}] {e.extraction_text} -> {e.attributes}")
