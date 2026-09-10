# Custom Output Schema Example

LangExtract usually derives provider schema constraints from examples. For
advanced cases, pass `output_schema` to constrain the raw model output more
directly. This example restricts a `status` attribute to the enum values
`present` and `absent`.

Examples are optional when `output_schema` is provided. When examples are
included, they still guide the prompt; `output_schema` replaces only the
provider schema constraint. The schema must describe LangExtract's JSON output
envelope with a top-level `extractions` array.

Gemini and OpenAI support `output_schema`. Ollama does not currently support
user-provided output schemas.

The helper emits `additionalProperties: False` so schemas work with OpenAI
strict structured outputs. Gemini receives user-provided schemas through its
native JSON Schema field, so JSON Schema keywords such as
`additionalProperties` are preserved.

## Full Pipeline Example

```python
import langextract as lx

# Text with one affirmed and one negated condition.
input_text = "Patient has hypertension. Patient denies diabetes."

# Define extraction prompt.
prompt_description = """
Extract medical conditions and classify each condition status as present or
absent. Use exact text from the input for extraction_text.
"""

# Define example data. The status values mirror the enum in output_schema.
examples = [
    lx.data.ExampleData(
        text="Patient has asthma. Patient denies fever.",
        extractions=[
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text="asthma",
                attributes={"status": "present"},
            ),
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text="fever",
                attributes={"status": "absent"},
            ),
        ],
    )
]

# Build a LangExtract output envelope with an enum-constrained attribute.
output_schema = lx.schema.extractions_schema(
    lx.schema.extraction_item_schema(
        "condition",
        attributes={
            "status": {
                "type": "string",
                "enum": ["present", "absent"],
            }
        },
    )
)

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt_description,
    examples=examples,
    model_id="gemini-3.5-flash",
    output_schema=output_schema,
    temperature=0.0,
)

print(f"Input: {input_text}\n")
print("Extracted conditions:")
for extraction in result.extractions:
    status = extraction.attributes["status"]
    print(f"• {extraction.extraction_text}: {status}")
```

This will produce output similar to:

```text
Input: Patient has hypertension. Patient denies diabetes.

Extracted conditions:
• hypertension: present
• diabetes: absent
```

## Multiple Extraction Classes

For heterogeneous extraction classes, pass multiple item schemas. The helper
wraps them in `anyOf` under `extractions.items`:

```python
output_schema = lx.schema.extractions_schema(
    lx.schema.extraction_item_schema("condition"),
    lx.schema.extraction_item_schema("medication"),
)
```

## Raw Schema Equivalent

For full control, pass a raw JSON schema dictionary. When targeting OpenAI
strict mode, every object schema must declare `required` fields and
`additionalProperties: False`.

Attribute objects use the `<extraction_class>_attributes` property name.
LangExtract's resolver expects that suffix when parsing raw model output.
Each extraction item should use extraction-class text keys such as
`condition`; generic fields such as `extraction_class`, `extraction_text`,
and `attributes` are not resolver output keys. Extraction class names ending
in `_attributes` are reserved for attribute objects.

The full pipeline example above produces this equivalent envelope:

```python
output_schema = {
    "type": "object",
    "properties": {
        "extractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "condition_attributes": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["present", "absent"],
                            }
                        },
                        "required": ["status"],
                        "additionalProperties": False,
                    },
                },
                "required": ["condition", "condition_attributes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["extractions"],
    "additionalProperties": False,
}
```

Use raw schemas when you need JSON Schema constructs that the helpers do not
cover directly, such as custom `anyOf` variants. OpenAI strict structured
outputs support `anyOf`; use `strict=False` only in lower-level provider code
if you need to experiment with schema features outside OpenAI's strict subset.

## Optional Attributes

The helper marks every supplied attribute as required by default so the schema
is compatible with OpenAI strict structured outputs. To allow an attribute to
be absent in practice, make the value nullable while keeping the key required:

```python
output_schema = lx.schema.extractions_schema(
    lx.schema.extraction_item_schema(
        "condition",
        attributes={
            "status": {
                "anyOf": [
                    {"type": "string", "enum": ["present", "absent"]},
                    {"type": "null"},
                ]
            }
        },
    )
)
```

If you need a schema where an attribute key may be omitted entirely, use a raw
schema for that provider-specific shape.

## Errors and Pitfalls

- Invalid envelopes raise `InferenceConfigError` before provider construction.
- `output_schema` can be passed with either `model_id`/`config` or a
  preconfigured `model` when the provider supports user schemas.
- Examples are optional with `output_schema`. When supplied, keep example
  classes and attribute names aligned with the schema to avoid confusing the
  model.
- `output_schema` requires raw JSON provider output. Leave `format_type` unset
  or set it to `lx.data.FormatType.JSON`, and do not force fences.
- Keep the resolver's default `"_attributes"` suffix. Custom
  `attribute_suffix`/`extraction_attributes_suffix` settings are incompatible
  with the raw schema envelope.
- Do not combine `output_schema` with provider schema kwargs such as
  `response_format`, `response_schema`, or `response_json_schema`.
- When targeting Gemini 2.0 models, add Gemini's `propertyOrdering` keyword
  to object schemas that need an explicit property order. The LangExtract
  helpers stay provider-neutral and do not add that Gemini-specific extension.
- Raw schemas must describe `extractions.items` inline, including each
  extraction text key and `<extraction_class>_attributes` object. LangExtract
  does not resolve `$ref` for those resolver keys before provider construction.
- Use `anyOf`, not `oneOf`, for item unions. Gemini treats `oneOf` like
  `anyOf`, and OpenAI strict structured outputs reject `oneOf`.
- `lx.schema.extraction_item_schema(..., additional_properties=False)` applies
  that setting to both the outer extraction item object and its nested
  `<extraction_class>_attributes` object.
- OpenAI uses strict structured outputs by default with LangExtract's default
  schema name. The lower-level `OpenAISchema.from_schema_dict(...,
  schema_name=..., strict=False)` constructor is an escape hatch for callers
  configuring provider models directly.
- LangExtract validates only the output envelope locally; the provider API
  validates the JSON schema itself. OpenAI strict mode requires every object
  to list all properties in `required` and set
  `additionalProperties: false` — the `lx.schema` helpers emit compliant
  schemas, and the OpenAI API reports the exact path of any violation in
  hand-written schemas. Schema size/depth limits, enum limits, and keyword
  support also vary by provider, model, and endpoint.
- Avoid `stop`/`stop_sequences` with `output_schema`: stop sequences can
  truncate schema-constrained JSON mid-document while the response still
  reports a normal finish reason.
