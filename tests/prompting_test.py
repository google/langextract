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

import textwrap

from absl.testing import absltest
from absl.testing import parameterized

from langextract import prompting
from langextract.core import data
from langextract.core import format_handler as fh


class QAPromptGeneratorTest(parameterized.TestCase):

  def test_generate_prompt(self):
    prompt_template_structured = prompting.PromptTemplateStructured(
        description=(
            "You are an assistant specialized in extracting key extractions"
            " from text.\nIdentify and extract important extractions such as"
            " people, places,\norganizations, dates, and medical conditions"
            " mentioned in the text.\n**Please ensure that the extractions are"
            " extracted in the same order as they\nappear in the source"
            " text.**\nProvide the extracted extractions in a structured YAML"
            " format."
        ),
        examples=[
            data.ExampleData(
                text=(
                    "The patient was diagnosed with hypertension and diabetes."
                ),
                extractions=[
                    data.Extraction(
                        extraction_text="hypertension",
                        extraction_class="medical_condition",
                        attributes={
                            "chronicity": "chronic",
                            "system": "cardiovascular",
                        },
                    ),
                    data.Extraction(
                        extraction_text="diabetes",
                        extraction_class="medical_condition",
                        attributes={
                            "chronicity": "chronic",
                            "system": "endocrine",
                        },
                    ),
                ],
            )
        ],
    )

    format_handler = fh.FormatHandler(
        format_type=data.FormatType.YAML,
        use_wrapper=True,
        wrapper_key="extractions",
        use_fences=True,
    )

    prompt_generator = prompting.QAPromptGenerator(
        template=prompt_template_structured,
        format_handler=format_handler,
        examples_heading="",
        question_prefix="",
        answer_prefix="",
    )

    actual_prompt_text = prompt_generator.render(
        "The patient reports chest pain and shortness of breath."
    )

    expected_prompt_text = textwrap.dedent(f"""\
        You are an assistant specialized in extracting key extractions from text.
        Identify and extract important extractions such as people, places,
        organizations, dates, and medical conditions mentioned in the text.
        **Please ensure that the extractions are extracted in the same order as they
        appear in the source text.**
        Provide the extracted extractions in a structured YAML format.


        The patient was diagnosed with hypertension and diabetes.
        ```yaml
        {data.EXTRACTIONS_KEY}:
        - medical_condition: hypertension
          medical_condition_attributes:
            chronicity: chronic
            system: cardiovascular
        - medical_condition: diabetes
          medical_condition_attributes:
            chronicity: chronic
            system: endocrine
        ```

        The patient reports chest pain and shortness of breath.
        """)
    self.assertEqual(expected_prompt_text, actual_prompt_text)

  @parameterized.named_parameters(
      dict(
          testcase_name="json_basic_format",
          format_type=data.FormatType.JSON,
          example_text="Patient has diabetes and is prescribed insulin.",
          example_extractions=[
              data.Extraction(
                  extraction_text="diabetes",
                  extraction_class="medical_condition",
                  attributes={"chronicity": "chronic"},
              ),
              data.Extraction(
                  extraction_text="insulin",
                  extraction_class="medication",
                  attributes={"prescribed": "prescribed"},
              ),
          ],
          expected_formatted_example=textwrap.dedent(f"""\
              Patient has diabetes and is prescribed insulin.
              ```json
              {{
                "{data.EXTRACTIONS_KEY}": [
                  {{
                    "medical_condition": "diabetes",
                    "medical_condition_attributes": {{
                      "chronicity": "chronic"
                    }}
                  }},
                  {{
                    "medication": "insulin",
                    "medication_attributes": {{
                      "prescribed": "prescribed"
                    }}
                  }}
                ]
              }}
              ```
              """),
      ),
      dict(
          testcase_name="yaml_basic_format",
          format_type=data.FormatType.YAML,
          example_text="Patient has diabetes and is prescribed insulin.",
          example_extractions=[
              data.Extraction(
                  extraction_text="diabetes",
                  extraction_class="medical_condition",
                  attributes={"chronicity": "chronic"},
              ),
              data.Extraction(
                  extraction_text="insulin",
                  extraction_class="medication",
                  attributes={"prescribed": "prescribed"},
              ),
          ],
          expected_formatted_example=textwrap.dedent(f"""\
              Patient has diabetes and is prescribed insulin.
              ```yaml
              {data.EXTRACTIONS_KEY}:
              - medical_condition: diabetes
                medical_condition_attributes:
                  chronicity: chronic
              - medication: insulin
                medication_attributes:
                  prescribed: prescribed
              ```
              """),
      ),
      dict(
          testcase_name="custom_attribute_suffix",
          format_type=data.FormatType.YAML,
          example_text="Patient has a fever.",
          example_extractions=[
              data.Extraction(
                  extraction_text="fever",
                  extraction_class="symptom",
                  attributes={"severity": "mild"},
              ),
          ],
          attribute_suffix="_props",
          expected_formatted_example=textwrap.dedent(f"""\
              Patient has a fever.
              ```yaml
              {data.EXTRACTIONS_KEY}:
              - symptom: fever
                symptom_props:
                  severity: mild
              ```
              """),
      ),
      dict(
          testcase_name="yaml_empty_extractions",
          format_type=data.FormatType.YAML,
          example_text="Text with no extractions.",
          example_extractions=[],
          expected_formatted_example=textwrap.dedent(f"""\
              Text with no extractions.
              ```yaml
              {data.EXTRACTIONS_KEY}: []
              ```
              """),
      ),
      dict(
          testcase_name="json_empty_extractions",
          format_type=data.FormatType.JSON,
          example_text="Text with no extractions.",
          example_extractions=[],
          expected_formatted_example=textwrap.dedent(f"""\
              Text with no extractions.
              ```json
              {{
                "{data.EXTRACTIONS_KEY}": []
              }}
              ```
              """),
      ),
      dict(
          testcase_name="yaml_empty_attributes",
          format_type=data.FormatType.YAML,
          example_text="Patient is resting comfortably.",
          example_extractions=[
              data.Extraction(
                  extraction_text="Patient",
                  extraction_class="person",
                  attributes={},
              ),
          ],
          expected_formatted_example=textwrap.dedent(f"""\
              Patient is resting comfortably.
              ```yaml
              {data.EXTRACTIONS_KEY}:
              - person: Patient
                person_attributes: {{}}
              ```
              """),
      ),
      dict(
          testcase_name="json_empty_attributes",
          format_type=data.FormatType.JSON,
          example_text="Patient is resting comfortably.",
          example_extractions=[
              data.Extraction(
                  extraction_text="Patient",
                  extraction_class="person",
                  attributes={},
              ),
          ],
          expected_formatted_example=textwrap.dedent(f"""\
              Patient is resting comfortably.
              ```json
              {{
                "{data.EXTRACTIONS_KEY}": [
                  {{
                    "person": "Patient",
                    "person_attributes": {{}}
                  }}
                ]
              }}
              ```
              """),
      ),
      dict(
          testcase_name="yaml_same_extraction_class_multiple_times",
          format_type=data.FormatType.YAML,
          example_text=(
              "Patient has multiple medications: aspirin and lisinopril."
          ),
          example_extractions=[
              data.Extraction(
                  extraction_text="aspirin",
                  extraction_class="medication",
                  attributes={"dosage": "81mg"},
              ),
              data.Extraction(
                  extraction_text="lisinopril",
                  extraction_class="medication",
                  attributes={"dosage": "10mg"},
              ),
          ],
          expected_formatted_example=textwrap.dedent(f"""\
              Patient has multiple medications: aspirin and lisinopril.
              ```yaml
              {data.EXTRACTIONS_KEY}:
              - medication: aspirin
                medication_attributes:
                  dosage: 81mg
              - medication: lisinopril
                medication_attributes:
                  dosage: 10mg
              ```
              """),
      ),
      dict(
          testcase_name="json_simplified_no_extractions_key",
          format_type=data.FormatType.JSON,
          example_text="Patient has diabetes and is prescribed insulin.",
          example_extractions=[
              data.Extraction(
                  extraction_text="diabetes",
                  extraction_class="medical_condition",
                  attributes={"chronicity": "chronic"},
              ),
              data.Extraction(
                  extraction_text="insulin",
                  extraction_class="medication",
                  attributes={"prescribed": "prescribed"},
              ),
          ],
          require_extractions_key=False,
          expected_formatted_example=textwrap.dedent("""\
              Patient has diabetes and is prescribed insulin.
              ```json
              [
                {
                  "medical_condition": "diabetes",
                  "medical_condition_attributes": {
                    "chronicity": "chronic"
                  }
                },
                {
                  "medication": "insulin",
                  "medication_attributes": {
                    "prescribed": "prescribed"
                  }
                }
              ]
              ```
              """),
      ),
      dict(
          testcase_name="yaml_simplified_no_extractions_key",
          format_type=data.FormatType.YAML,
          example_text="Patient has a fever.",
          example_extractions=[
              data.Extraction(
                  extraction_text="fever",
                  extraction_class="symptom",
                  attributes={"severity": "mild"},
              ),
          ],
          require_extractions_key=False,
          expected_formatted_example=textwrap.dedent("""\
              Patient has a fever.
              ```yaml
              - symptom: fever
                symptom_attributes:
                  severity: mild
              ```
              """),
      ),
  )
  def test_format_example(
      self,
      format_type,
      example_text,
      example_extractions,
      expected_formatted_example,
      attribute_suffix="_attributes",
      require_extractions_key=True,
  ):
    """Tests formatting of examples in different formats and scenarios."""
    example_data = data.ExampleData(
        text=example_text,
        extractions=example_extractions,
    )

    structured_template = prompting.PromptTemplateStructured(
        description="Extract information from the text.",
        examples=[example_data],
    )

    format_handler = fh.FormatHandler(
        format_type=format_type,
        use_wrapper=require_extractions_key,
        wrapper_key="extractions" if require_extractions_key else None,
        use_fences=True,
        attribute_suffix=attribute_suffix,
    )

    prompt_generator = prompting.QAPromptGenerator(
        template=structured_template,
        format_handler=format_handler,
        question_prefix="",
        answer_prefix="",
    )

    actual_formatted_example = prompt_generator.format_example_as_text(
        example_data
    )
    self.assertEqual(expected_formatted_example, actual_formatted_example)


class PromptBuilderTest(absltest.TestCase):
  """Tests for PromptBuilder base class."""

  def _create_generator(self):
    """Creates a simple QAPromptGenerator for testing."""
    template = prompting.PromptTemplateStructured(
        description="Extract entities.",
        examples=[
            data.ExampleData(
                text="Sample text.",
                extractions=[
                    data.Extraction(
                        extraction_text="Sample",
                        extraction_class="entity",
                    )
                ],
            )
        ],
    )
    format_handler = fh.FormatHandler(
        format_type=data.FormatType.YAML,
        use_wrapper=True,
        wrapper_key="extractions",
        use_fences=True,
    )
    return prompting.QAPromptGenerator(
        template=template,
        format_handler=format_handler,
    )

  def test_build_prompt_renders_chunk_text(self):
    """Verifies build_prompt includes chunk text in the rendered prompt."""
    generator = self._create_generator()
    builder = prompting.PromptBuilder(generator)

    prompt = builder.build_prompt(
        chunk_text="Test input text.",
        document_id="doc1",
    )

    self.assertIn("Test input text.", prompt)
    self.assertIn("Extract entities.", prompt)

  def test_build_prompt_includes_additional_context(self):
    """Verifies build_prompt passes additional_context to renderer."""
    generator = self._create_generator()
    builder = prompting.PromptBuilder(generator)

    prompt = builder.build_prompt(
        chunk_text="Test input.",
        document_id="doc1",
        additional_context="Important context here.",
    )

    self.assertIn("Important context here.", prompt)


class ContextAwarePromptBuilderTest(absltest.TestCase):
  """Tests for ContextAwarePromptBuilder."""

  def _create_generator(self):
    """Creates a simple QAPromptGenerator for testing."""
    template = prompting.PromptTemplateStructured(
        description="Extract entities.",
        examples=[
            data.ExampleData(
                text="Sample text.",
                extractions=[
                    data.Extraction(
                        extraction_text="Sample",
                        extraction_class="entity",
                    )
                ],
            )
        ],
    )
    format_handler = fh.FormatHandler(
        format_type=data.FormatType.YAML,
        use_wrapper=True,
        wrapper_key="extractions",
        use_fences=True,
    )
    return prompting.QAPromptGenerator(
        template=template,
        format_handler=format_handler,
    )

  def test_context_window_chars_property(self):
    """Verifies the context_window_chars property returns configured value."""
    generator = self._create_generator()

    builder_none = prompting.ContextAwarePromptBuilder(generator)
    self.assertIsNone(builder_none.context_window_chars)

    builder_with_value = prompting.ContextAwarePromptBuilder(
        generator, context_window_chars=100
    )
    self.assertEqual(100, builder_with_value.context_window_chars)

  def test_first_chunk_has_no_previous_context(self):
    """Verifies the first chunk does not include previous context."""
    generator = self._create_generator()
    builder = prompting.ContextAwarePromptBuilder(
        generator, context_window_chars=50
    )
    context_prefix = prompting.ContextAwarePromptBuilder._CONTEXT_PREFIX

    prompt = builder.build_prompt(
        chunk_text="First chunk text.",
        document_id="doc1",
    )

    self.assertNotIn(context_prefix, prompt)
    self.assertIn("First chunk text.", prompt)

  def test_second_chunk_includes_previous_context(self):
    """Verifies the second chunk includes text from the first chunk."""
    generator = self._create_generator()
    builder = prompting.ContextAwarePromptBuilder(
        generator, context_window_chars=20
    )
    context_prefix = prompting.ContextAwarePromptBuilder._CONTEXT_PREFIX

    builder.build_prompt(chunk_text="First chunk ending.", document_id="doc1")
    second_prompt = builder.build_prompt(
        chunk_text="Second chunk text.",
        document_id="doc1",
    )

    self.assertIn(context_prefix, second_prompt)
    self.assertIn("chunk ending.", second_prompt)

  def test_context_disabled_when_none(self):
    """Verifies no context is added when context_window_chars is None."""
    generator = self._create_generator()
    builder = prompting.ContextAwarePromptBuilder(
        generator, context_window_chars=None
    )
    context_prefix = prompting.ContextAwarePromptBuilder._CONTEXT_PREFIX

    builder.build_prompt(chunk_text="First chunk.", document_id="doc1")
    second_prompt = builder.build_prompt(
        chunk_text="Second chunk.",
        document_id="doc1",
    )

    self.assertNotIn(context_prefix, second_prompt)

  def test_context_isolated_per_document(self):
    """Verifies context tracking is isolated per document_id."""
    generator = self._create_generator()
    builder = prompting.ContextAwarePromptBuilder(
        generator, context_window_chars=50
    )

    builder.build_prompt(chunk_text="Doc A chunk one.", document_id="docA")
    builder.build_prompt(chunk_text="Doc B chunk one.", document_id="docB")

    prompt_a2 = builder.build_prompt(
        chunk_text="Doc A chunk two.",
        document_id="docA",
    )
    prompt_b2 = builder.build_prompt(
        chunk_text="Doc B chunk two.",
        document_id="docB",
    )

    self.assertIn("Doc A chunk one", prompt_a2)
    self.assertNotIn("Doc B", prompt_a2)
    self.assertIn("Doc B chunk one", prompt_b2)
    self.assertNotIn("Doc A", prompt_b2)

  def test_combines_previous_context_with_additional_context(self):
    """Verifies both previous chunk context and additional_context are included."""
    generator = self._create_generator()
    builder = prompting.ContextAwarePromptBuilder(
        generator, context_window_chars=30
    )
    context_prefix = prompting.ContextAwarePromptBuilder._CONTEXT_PREFIX

    builder.build_prompt(chunk_text="Previous chunk text.", document_id="doc1")
    prompt = builder.build_prompt(
        chunk_text="Current chunk.",
        document_id="doc1",
        additional_context="Extra info here.",
    )

    self.assertIn(context_prefix, prompt)
    self.assertIn("Previous chunk text.", prompt)
    self.assertIn("Extra info here.", prompt)


if __name__ == "__main__":
  absltest.main()
