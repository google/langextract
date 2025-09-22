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

"""Tests for centralized format handler."""

import textwrap

from absl.testing import absltest
from absl.testing import parameterized

from langextract import prompting
from langextract import resolver
from langextract.core import data
from langextract.core import format_handler


class FormatHandlerTest(parameterized.TestCase):
    """Tests for FormatHandler."""

    @parameterized.named_parameters(
        dict(
            testcase_name="json_with_wrapper_and_fences",
            format_type=data.FormatType.JSON,
            use_wrapper=True,
            wrapper_key="extractions",
            use_fences=True,
            extraction_class="person",
            extraction_text="Alice",
            attributes={"role": "engineer"},
            expected_fence="```json",
            expected_wrapper='"extractions":',
            expected_extraction='"person": "Alice"',
            model_output=textwrap.dedent(
                """
              Here is the result:
              ```json
              {
                "extractions": [
                  {"person": "Bob", "person_attributes": {"role": "manager"}}
                ]
              }
              ```
          """
            ).strip(),
            parsed_class="person",
            parsed_text="Bob",
        ),
        dict(
            testcase_name="json_no_wrapper_no_fences",
            format_type=data.FormatType.JSON,
            use_wrapper=False,
            wrapper_key=None,
            use_fences=False,
            extraction_class="item",
            extraction_text="book",
            attributes=None,
            expected_fence=None,
            expected_wrapper=None,
            expected_extraction='"item": "book"',
            model_output='[{"item": "pen", "item_attributes": {}}]',
            parsed_class="item",
            parsed_text="pen",
        ),
        dict(
            testcase_name="yaml_with_wrapper_and_fences",
            format_type=data.FormatType.YAML,
            use_wrapper=True,
            wrapper_key="extractions",
            use_fences=True,
            extraction_class="city",
            extraction_text="Paris",
            attributes=None,
            expected_fence="```yaml",
            expected_wrapper="extractions:",
            expected_extraction="city: Paris",
            model_output=textwrap.dedent(
                """
              ```yaml
              extractions:
                - city: London
                  city_attributes: {}
              ```
          """
            ).strip(),
            parsed_class="city",
            parsed_text="London",
        ),
    )
    def test_format_and_parse(  # pylint: disable=too-many-arguments
        self,
        format_type,
        use_wrapper,
        wrapper_key,
        use_fences,
        extraction_class,
        extraction_text,
        attributes,
        expected_fence,
        expected_wrapper,
        expected_extraction,
        model_output,
        parsed_class,
        parsed_text,
    ):
        """Test formatting and parsing with various configurations."""
        handler = format_handler.FormatHandler(
            format_type=format_type,
            use_wrapper=use_wrapper,
            wrapper_key=wrapper_key,
            use_fences=use_fences,
        )

        extractions = [
            data.Extraction(
                extraction_class=extraction_class,
                extraction_text=extraction_text,
                attributes=attributes,
            )
        ]

        formatted = handler.format_extraction_example(extractions)

        if expected_fence:
            self.assertIn(expected_fence, formatted)
        else:
            self.assertNotIn("```", formatted)

        if expected_wrapper:
            self.assertIn(expected_wrapper, formatted)
        else:
            if wrapper_key:
                self.assertNotIn(wrapper_key, formatted)

        self.assertIn(expected_extraction, formatted)

        parsed = handler.parse_output(model_output)
        self.assertLen(parsed, 1)
        self.assertEqual(parsed[0][parsed_class], parsed_text)

    def test_end_to_end_integration_with_prompt_and_resolver(self):
        """Test that FormatHandler unifies prompt generation and parsing."""
        handler = format_handler.FormatHandler(
            format_type=data.FormatType.JSON,
            use_wrapper=True,
            wrapper_key="extractions",
            use_fences=True,
        )

        template = prompting.PromptTemplateStructured(
            description="Extract entities from text.",
            examples=[
                data.ExampleData(
                    text="Alice is an engineer",
                    extractions=[
                        data.Extraction(
                            extraction_class="person",
                            extraction_text="Alice",
                            attributes={"role": "engineer"},
                        )
                    ],
                )
            ],
        )

        prompt_gen = prompting.QAPromptGenerator(
            template=template,
            format_handler=handler,
        )

        prompt = prompt_gen.render("Bob is a manager")
        self.assertIn("```json", prompt, "Prompt should contain JSON fence")
        self.assertIn('"extractions":', prompt, "Prompt should contain wrapper key")

        test_resolver = resolver.Resolver(
            format_handler=handler,
            extraction_index_suffix=None,
        )

        model_output = textwrap.dedent(
            """
        ```json
        {
          "extractions": [
            {
              "person": "Bob",
              "person_attributes": {"role": "manager"}
            }
          ]
        }
        ```
    """
        ).strip()

        extractions = test_resolver.resolve(model_output)
        self.assertLen(extractions, 1, "Should extract exactly one entity")
        self.assertEqual(
            extractions[0].extraction_class,
            "person",
            "Extraction class should be 'person'",
        )
        self.assertEqual(
            extractions[0].extraction_text, "Bob", "Extraction text should be 'Bob'"
        )

    @parameterized.named_parameters(
        dict(
            testcase_name="yaml_no_wrapper_no_fences",
            format_type=data.FormatType.YAML,
            use_wrapper=False,
            use_fences=False,
        ),
        dict(
            testcase_name="json_with_wrapper_and_fences",
            format_type=data.FormatType.JSON,
            use_wrapper=True,
            wrapper_key="extractions",
            use_fences=True,
        ),
        dict(
            testcase_name="yaml_with_wrapper_no_fences",
            format_type=data.FormatType.YAML,
            use_wrapper=True,
            wrapper_key="extractions",
            use_fences=False,
        ),
    )
    def test_format_parse_roundtrip(
        self, format_type, use_wrapper, use_fences, wrapper_key=None
    ):
        """Test that what we format can be parsed back identically."""
        handler = format_handler.FormatHandler(
            format_type=format_type,
            use_wrapper=use_wrapper,
            wrapper_key=wrapper_key,
            use_fences=use_fences,
        )

        extractions = [
            data.Extraction(
                extraction_class="test",
                extraction_text="value",
                attributes={"key": "data"},
            )
        ]
        formatted = handler.format_extraction_example(extractions)

        parsed = handler.parse_output(formatted)
        self.assertEqual(parsed[0]["test"], "value")
        self.assertEqual(parsed[0]["test_attributes"]["key"], "data")


if __name__ == "__main__":
    absltest.main()
