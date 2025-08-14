# *Romeo and Juliet* Full Text Extraction

LangExtract can process entire documents directly from URLs, handling large texts with high accuracy through parallel processing and enhanced sensitivity features. This example demonstrates extraction from the complete text of *Romeo and Juliet* from Project Gutenberg.

## Example code

The following code uses a comprehensive prompt and examples optimized for large, complex literary texts. For large complex inputs, using more detailed examples is suggested to increase extraction robustness.

> **Warning:** Running this example processes a large document (~44 000 tokens) and will incur costs. For large-scale use, a Tier 2 Gemini quota is suggested to avoid rate-limit issues ([details](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2)). Please review the [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) before proceeding.

```python
import langextract as lx
import textwrap
from collections import Counter, defaultdict

# Define comprehensive prompt and examples for complex literary text
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships from the given text.

    Provide meaningful attributes for every entity to add context and depth.

    Important: Use exact text from the input for extraction_text. Do not paraphrase.
    Extract entities in order of appearance with no overlapping text spans.

    Note: In play scripts, speaker names appear in ALL-CAPS followed by a period.""")

examples = [
    lx.data.ExampleData(
        text=textwrap.dedent("""\
            ROMEO. But soft! What light through yonder window breaks?
            It is the east, and Juliet is the sun.
            JULIET. O Romeo, Romeo! Wherefore art thou Romeo?"""),
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe", "character": "Romeo"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor", "character_1": "Romeo", "character_2": "Juliet"}
            ),
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="JULIET",
                attributes={"emotional_state": "yearning"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="Wherefore art thou Romeo?",
                attributes={"feeling": "longing question", "character": "Juliet"}
            ),
        ]
    )
]

# Process Romeo & Juliet directly from Project Gutenberg
print("Downloading and processing Romeo and Juliet from Project Gutenberg...")

result = lx.extract(
    text_or_documents="https://www.gutenberg.org/files/1513/1513-0.txt",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    extraction_passes=3,      # Multiple passes for improved recall
    max_workers=20,           # Parallel processing for speed
    max_char_buffer=1000      # Smaller contexts for better accuracy
)

print(f"Extracted {len(result.extractions)} entities from {len(result.text):,} characters")

# Save and visualize the results
lx.io.save_annotated_documents([result], output_name="romeo_juliet_extractions.jsonl", output_dir=".")

# Generate the interactive visualization
html_content = lx.visualize("romeo_juliet_extractions.jsonl")
with open("romeo_juliet_visualization.html", "w") as f:
    f.write(html_content.data)

print("Interactive visualization saved to romeo_juliet_visualization.html")
```

This creates an interactive HTML visualization for exploring the extracted entities:

![Romeo and Juliet Full Visualization](../_static/romeo_juliet_full.gif)

```python

# Analyze character mentions
characters = {}
for e in result.extractions:
    if e.extraction_class == "character":
        char_name = e.extraction_text
        if char_name not in characters:
            characters[char_name] = {"count": 0, "attributes": set()}
        characters[char_name]["count"] += 1
        if e.attributes:
            for attr_key, attr_val in e.attributes.items():
                characters[char_name]["attributes"].add(f"{attr_key}: {attr_val}")

# Print character summary
print(f"\nCHARACTER SUMMARY ({len(characters)} unique characters)")
print("=" * 60)

sorted_chars = sorted(characters.items(), key=lambda x: x[1]["count"], reverse=True)
for char_name, char_data in sorted_chars[:10]:  # Top 10 characters
    attrs_preview = list(char_data["attributes"])[:3]
    attrs_str = f" ({', '.join(attrs_preview)})" if attrs_preview else ""
    print(f"{char_name}: {char_data['count']} mentions{attrs_str}")

# Entity type breakdown
entity_counts = Counter(e.extraction_class for e in result.extractions)
print(f"\nENTITY TYPE BREAKDOWN")
print("=" * 60)
for entity_type, count in entity_counts.most_common():
    percentage = (count / len(result.extractions)) * 100
    print(f"{entity_type}: {count} ({percentage:.1f}%)")
```

## Sample output

```
Downloading and processing Romeo and Juliet from Project Gutenberg...
Downloaded 147,843 characters (25,976 words) from 1513-0.txt
Extracted 4,088 entities from 147,843 characters
Interactive visualization saved to romeo_juliet_visualization.html

CHARACTER SUMMARY (153 unique characters)
============================================================
ROMEO: 287 mentions (emotional_state: excitement, emotional_state: eager to please)
JULIET: 204 mentions (emotional_state: fond, emotional_state: resilient)
NURSE: 168 mentions (emotional_state: reporting, emotional_state: teasing and evasive)
MERCUTIO: 107 mentions (emotional_state: approving, emotional_state: responsive)
BENVOLIO: 82 mentions (emotional_state: cautious, emotional_state: teasing)

ENTITY TYPE BREAKDOWN
============================================================
character: 1,685 (41.2%)
emotion: 1,524 (37.3%)
relationship: 879 (21.5%)
```

## Key benefits for long documents

### Sequential extraction passes

Multiple extraction passes improve recall by performing independent extractions and merging non-overlapping results. Each pass uses identical parameters and processing—they are independent runs of the same extraction task. The number of passes is controlled by the `extraction_passes` parameter (e.g., `extraction_passes=3`).

**How it works**: Each pass processes the full text independently using the same prompt and examples. Results are then merged using a "first-pass wins" strategy for overlapping entities, while adding unique non-overlapping entities from later passes. This approach captures entities that might be missed in any single run due to the stochastic nature of language model generation.

### Portable and Interoperable Data with JSONL
LangExtract uses JSONL, a human-readable format ideal for language model data. Each line is a self-contained JSON object, making outputs easy to parse, share, and integrate with other tools. You can save results with `lx.io.save_annotated_documents` and reload them for later analysis, ensuring your data is both portable and persistent.

### Optimal long context management
While single-inference approaches can be powerful, their accuracy may be affected by distant context. LangExtract uses smart chunking strategies that respect text delimiters (such as paragraph breaks) to keep context intact and well-formed for the model. Users can configure context sizes (`max_char_buffer`) combined with parallel processing (`max_workers`) to maintain extraction quality across large documents. Multiple sequential extraction passes further enhance sensitivity by capturing entities that might be missed in any single run due to the stochastic nature of language model generation.

### Enhanced accuracy through chunking
The chunked processing approach can improve extraction quality over a single inference pass on a large document because each chunk uses a smaller, more manageable context size. This helps the model focus on the most relevant information and prevents interference from distant context. While the overall latency and time required remain similar due to parallelization, the extraction quality can be substantially higher with better entity coverage and more accurate attribute assignment across the entire document.¹

### Interactive visualization at scale
Seamlessly explore hundreds or thousands of entities through interactive HTML visualizations generated directly from JSONL output files. The generated visualizations handle large result sets efficiently, providing navigation and detailed entity inspection capabilities for comprehensive analysis of complex documents.

### Schema-guided knowledge extraction
LangExtract combines precise text positioning with world knowledge enrichment, enabling extraction of information not explicitly stated in the text (like character identities and traits). Under the hood, the library implements [Controlled Generation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output) with supported models to ensure extracted data adheres to your specified schema while maintaining robust extractions across large inputs.

---

¹ Models like Gemini 1.5 Pro show strong performance on many benchmarks, but [needle-in-a-haystack tests](https://cloud.google.com/blog/products/ai-machine-learning/the-needle-in-the-haystack-test-and-how-gemini-pro-solves-it) across million-token contexts indicate that performance can vary in multi-fact retrieval scenarios. This demonstrates how LangExtract's smaller context windows approach ensures consistently high quality across entire documents by avoiding the complexity and potential degradation of massive single-context processing.
