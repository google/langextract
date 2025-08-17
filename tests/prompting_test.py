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
from langextract import schema
from langextract.core import data


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

    prompt_generator = prompting.QAPromptGenerator(
        template=prompt_template_structured,
        format_type=data.FormatType.YAML,
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
        {schema.EXTRACTIONS_KEY}:
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
      {
          "testcase_name": "json_basic_format",
          "format_type": data.FormatType.JSON,
          "example_text": "Patient has diabetes and is prescribed insulin.",
          "example_extractions": [
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
          "expected_formatted_example": textwrap.dedent(f"""\
              Patient has diabetes and is prescribed insulin.
              ```json
              {{
                "{schema.EXTRACTIONS_KEY}": [
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
      },
      {
          "testcase_name": "yaml_basic_format",
          "format_type": data.FormatType.YAML,
          "example_text": "Patient has diabetes and is prescribed insulin.",
          "example_extractions": [
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
          "expected_formatted_example": textwrap.dedent(f"""\
              Patient has diabetes and is prescribed insulin.
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - medical_condition: diabetes
                medical_condition_attributes:
                  chronicity: chronic
              - medication: insulin
                medication_attributes:
                  prescribed: prescribed
              ```
              """),
      },
      {
          "testcase_name": "custom_attribute_suffix",
          "format_type": data.FormatType.YAML,
          "example_text": "Patient has a fever.",
          "example_extractions": [
              data.Extraction(
                  extraction_text="fever",
                  extraction_class="symptom",
                  attributes={"severity": "mild"},
              ),
          ],
          "attribute_suffix": "_props",
          "expected_formatted_example": textwrap.dedent(f"""\
              Patient has a fever.
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - symptom: fever
                symptom_props:
                  severity: mild
              ```
              """),
      },
      {
          "testcase_name": "yaml_empty_extractions",
          "format_type": data.FormatType.YAML,
          "example_text": "Text with no extractions.",
          "example_extractions": [],
          "expected_formatted_example": textwrap.dedent(f"""\
              Text with no extractions.
              ```yaml
              {schema.EXTRACTIONS_KEY}: []
              ```
              """),
      },
      {
          "testcase_name": "json_empty_extractions",
          "format_type": data.FormatType.JSON,
          "example_text": "Text with no extractions.",
          "example_extractions": [],
          "expected_formatted_example": textwrap.dedent(f"""\
              Text with no extractions.
              ```json
              {{
                "{schema.EXTRACTIONS_KEY}": []
              }}
              ```
              """),
      },
      {
          "testcase_name": "yaml_empty_attributes",
          "format_type": data.FormatType.YAML,
          "example_text": "Patient is resting comfortably.",
          "example_extractions": [
              data.Extraction(
                  extraction_text="Patient",
                  extraction_class="person",
                  attributes={},
              ),
          ],
          "expected_formatted_example": textwrap.dedent(f"""\
              Patient is resting comfortably.
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - person: Patient
                person_attributes: {{}}
              ```
              """),
      },
      {
          "testcase_name": "json_empty_attributes",
          "format_type": data.FormatType.JSON,
          "example_text": "Patient is resting comfortably.",
          "example_extractions": [
              data.Extraction(
                  extraction_text="Patient",
                  extraction_class="person",
                  attributes={},
              ),
          ],
          "expected_formatted_example": textwrap.dedent(f"""\
              Patient is resting comfortably.
              ```json
              {{
                "{schema.EXTRACTIONS_KEY}": [
                  {{
                    "person": "Patient",
                    "person_attributes": {{}}
                  }}
                ]
              }}
              ```
              """),
      },
      {
          "testcase_name": "yaml_same_extraction_class_multiple_times",
          "format_type": data.FormatType.YAML,
          "example_text": (
              "Patient has multiple medications: aspirin and lisinopril."
          ),
          "example_extractions": [
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
          "expected_formatted_example": textwrap.dedent(f"""\
              Patient has multiple medications: aspirin and lisinopril.
              ```yaml
              {schema.EXTRACTIONS_KEY}:
              - medication: aspirin
                medication_attributes:
                  dosage: 81mg
              - medication: lisinopril
                medication_attributes:
                  dosage: 10mg
              ```
              """),
      },
  )
  def test_format_example(
      self,
      format_type,
      example_text,
      example_extractions,
      expected_formatted_example,
      attribute_suffix="_attributes",
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

    prompt_generator = prompting.QAPromptGenerator(
        template=structured_template,
        format_type=format_type,
        attribute_suffix=attribute_suffix,
        question_prefix="",
        answer_prefix="",
    )

    actual_formatted_example = prompt_generator.format_example_as_text(
        example_data
    )
    self.assertEqual(expected_formatted_example, actual_formatted_example)


if __name__ == "__main__":
  absltest.main()
