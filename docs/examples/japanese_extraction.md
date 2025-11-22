# Japanese Information Extraction

This example demonstrates how to use LangExtract to extract structured information from Japanese text.

> **Note:** For non-spaced languages like Japanese, use `UnicodeTokenizer` to ensure correct character-based segmentation and alignment.

## Full Pipeline Example

```python
import langextract as lx
from langextract.core import tokenizer

# Japanese text with entities (Person, Location, Organization)
# "Mr. Tanaka from Tokyo works at Google."
input_text = "東京出身の田中さんはGoogleで働いています。"

# Define extraction prompt
prompt_description = "Extract named entities including Person, Location, and Organization."

# Define example data (few-shot examples help the model understand the task)
examples = [
    lx.data.ExampleData(
        text="大阪の山田さんはソニーに入社しました。",  # Mr. Yamada from Osaka joined Sony.
        extractions=[
            lx.data.Extraction(extraction_class="Location", extraction_text="大阪"),
            lx.data.Extraction(extraction_class="Person", extraction_text="山田"),
            lx.data.Extraction(extraction_class="Organization", extraction_text="ソニー"),
        ]
    )
]

# 1. Initialize the UnicodeTokenizer
# Essential for Japanese to ensure correct grapheme segmentation.
unicode_tokenizer = tokenizer.UnicodeTokenizer()

# 2. Run Extraction with the Custom Tokenizer
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt_description,
    examples=examples,
    model_id="gemini-2.5-flash",
    tokenizer=unicode_tokenizer,   # <--- Pass the tokenizer here
    api_key="your-api-key-here"    # Optional if env var is set
)

# 3. Display Results
print(f"Input: {input_text}\n")
print("Extracted Entities:")
for entity in result.extractions:
    position_info = ""
    if entity.char_interval:
        start, end = entity.char_interval.start_pos, entity.char_interval.end_pos
        position_info = f" (pos: {start}-{end})"
    
    print(f"• {entity.extraction_class}: {entity.extraction_text}{position_info}")

# Expected Output:
# Input: 東京出身の田中さんはGoogleで働いています。
#
# Extracted Entities:
# • Location: 東京 (pos: 0-2)
# • Person: 田中 (pos: 5-7)
# • Organization: Google (pos: 10-16)
```
