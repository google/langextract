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

"""Tests for inference module.

Note: This file contains test helper classes that intentionally have
few public methods and define attributes outside __init__. These
pylint warnings are expected for test fixtures.
"""
# pylint: disable=attribute-defined-outside-init

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract import exceptions
from langextract.core import base_model
from langextract.core import data
from langextract.core import types
from langextract.providers import gemini
from langextract.providers import ollama
from langextract.providers import openai


class TestBaseLanguageModel(absltest.TestCase):
    def test_merge_kwargs_with_none(self):
        """Test merge_kwargs handles None runtime_kwargs."""

        class TestModel(
            base_model.BaseLanguageModel
        ):  # pylint: disable=too-few-public-methods
            def infer(self, batch_prompts, **kwargs):
                return iter([])

        model = TestModel()
        model._extra_kwargs = {"a": 1, "b": 2}

        result = model.merge_kwargs(None)
        self.assertEqual(
            {"a": 1, "b": 2},
            result,
            "merge_kwargs(None) should return stored kwargs unchanged",
        )

        result = model.merge_kwargs({})
        self.assertEqual(
            {"a": 1, "b": 2},
            result,
            "merge_kwargs({}) should return stored kwargs unchanged",
        )

        result = model.merge_kwargs({"b": 3, "c": 4})
        self.assertEqual(
            {"a": 1, "b": 3, "c": 4},
            result,
            "Runtime kwargs should override stored kwargs and add new keys",
        )

    def test_merge_kwargs_without_extra_kwargs(self):
        """Test merge_kwargs when _extra_kwargs doesn't exist."""

        class TestModel(
            base_model.BaseLanguageModel
        ):  # pylint: disable=too-few-public-methods
            def infer(self, batch_prompts, **kwargs):
                return iter([])

        model = TestModel()
        # Intentionally not setting _extra_kwargs to test fallback behavior

        result = model.merge_kwargs({"a": 1})
        self.assertEqual(
            {"a": 1},
            result,
            "merge_kwargs should work even without _extra_kwargs attribute",
        )


class TestOllamaLanguageModel(absltest.TestCase):
    @mock.patch("langextract.providers.ollama.OllamaLanguageModel._ollama_query")
    def test_ollama_infer(self, mock_ollama_query):
        # Real gemma2 response structure from Ollama API for validation
        gemma_response = {
            "model": "gemma2:latest",
            "created_at": "2025-01-23T22:37:08.579440841Z",
            "response": "{'bus' : '**autóbusz**'} \n\n\n  \n",
            "done": True,
            "done_reason": "stop",
            "context": [
                106,
                1645,
                108,
                1841,
                603,
                1986,
                575,
                59672,
                235336,
                107,
                108,
                106,
                2516,
                108,
                9766,
                6710,
                235281,
                865,
                664,
                688,
                7958,
                235360,
                6710,
                235306,
                688,
                12990,
                235248,
                110,
                139,
                108,
            ],
            "total_duration": 24038204381,
            "load_duration": 21551375738,
            "prompt_eval_count": 15,
            "prompt_eval_duration": 633000000,
            "eval_count": 17,
            "eval_duration": 1848000000,
        }
        mock_ollama_query.return_value = gemma_response
        model = ollama.OllamaLanguageModel(
            model_id="gemma2:latest",
            model_url="http://localhost:11434",
            structured_output_format="json",
        )
        batch_prompts = ["What is bus in Hungarian?"]
        results = list(model.infer(batch_prompts))

        mock_ollama_query.assert_called_once_with(
            prompt="What is bus in Hungarian?",
            model="gemma2:latest",
            structured_output_format="json",
            model_url="http://localhost:11434",
        )
        expected_results = [
            [
                types.ScoredOutput(
                    score=1.0, output="{'bus' : '**autóbusz**'} \n\n\n  \n"
                )
            ]
        ]
        self.assertEqual(results, expected_results)

    @mock.patch("requests.post")
    def test_ollama_extra_kwargs_passed_to_api(self, mock_post):
        """Verify extra kwargs like timeout and keep_alive are passed to the API."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"test": "value"}',
            "done": True,
        }
        mock_post.return_value = mock_response

        model = ollama.OllamaLanguageModel(
            model_id="test-model",
            timeout=300,
            keep_alive=600,
            num_threads=8,
        )

        prompts = ["Test prompt"]
        list(model.infer(prompts))

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        json_payload = call_args.kwargs["json"]

        self.assertEqual(json_payload["options"]["keep_alive"], 600)
        self.assertEqual(json_payload["options"]["num_thread"], 8)
        # timeout is passed to requests.post, not in the JSON payload
        self.assertEqual(call_args.kwargs["timeout"], 300)

    @mock.patch("requests.post")
    def test_ollama_stop_and_top_p_passthrough(self, mock_post):
        """Verify stop and top_p parameters are passed to Ollama API."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"test": "value"}',
            "done": True,
        }
        mock_post.return_value = mock_response

        model = ollama.OllamaLanguageModel(
            model_id="test-model",
            top_p=0.9,
            stop=["\\n\\n", "END"],
        )

        prompts = ["Test prompt"]
        list(model.infer(prompts))

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        json_payload = call_args.kwargs["json"]

        # Ollama expects 'stop' at top level, not in options
        self.assertEqual(json_payload["stop"], ["\\n\\n", "END"])
        self.assertEqual(json_payload["options"]["top_p"], 0.9)

    @mock.patch("requests.post")
    def test_ollama_defaults_when_unspecified(self, mock_post):
        """Verify Ollama uses correct defaults when parameters are not specified."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"test": "value"}',
            "done": True,
        }
        mock_post.return_value = mock_response

        model = ollama.OllamaLanguageModel(model_id="test-model")

        prompts = ["Test prompt"]
        list(model.infer(prompts))

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        json_payload = call_args.kwargs["json"]

        self.assertEqual(json_payload["options"]["temperature"], 0.1)
        self.assertEqual(json_payload["options"]["keep_alive"], 300)
        self.assertEqual(json_payload["options"]["num_ctx"], 2048)
        self.assertEqual(call_args.kwargs["timeout"], 120)

    @mock.patch("requests.post")
    def test_ollama_runtime_kwargs_override_stored(self, mock_post):
        """Verify runtime kwargs override stored kwargs."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"test": "value"}',
            "done": True,
        }
        mock_post.return_value = mock_response

        model = ollama.OllamaLanguageModel(
            model_id="test-model",
            temperature=0.5,
            keep_alive=300,
        )

        prompts = ["Test prompt"]
        list(model.infer(prompts, temperature=0.8, keep_alive=600))

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        json_payload = call_args.kwargs["json"]

        self.assertEqual(json_payload["options"]["temperature"], 0.8)
        self.assertEqual(json_payload["options"]["keep_alive"], 600)

    @mock.patch("requests.post")
    def test_ollama_temperature_zero(self, mock_post):
        """Test that temperature=0.0 is properly passed to Ollama."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"test": "value"}',
            "done": True,
        }
        mock_post.return_value = mock_response

        model = ollama.OllamaLanguageModel(
            model_id="test-model",
            temperature=0.0,
        )

        list(model.infer(["test prompt"]))

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        json_payload = call_args.kwargs["json"]

        self.assertEqual(json_payload["options"]["temperature"], 0.0)

    def test_ollama_default_timeout(self):
        """Test that default timeout is used when not specified."""
        model = ollama.OllamaLanguageModel(
            model_id="test-model",
            model_url="http://localhost:11434",
        )

        mock_response = mock.Mock(spec=["status_code", "json"])
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test output"}

        with mock.patch.object(
            model._requests, "post", return_value=mock_response
        ) as mock_post:
            model._ollama_query(prompt="test prompt")

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(
                120,
                call_kwargs["timeout"],
                "Should use default timeout of 120 seconds",
            )

    def test_ollama_timeout_through_infer(self):
        """Test that timeout flows correctly through the infer() method."""
        model = ollama.OllamaLanguageModel(
            model_id="test-model",
            model_url="http://localhost:11434",
            timeout=60,
        )

        mock_response = mock.Mock(spec=["status_code", "json"])
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test output"}

        with mock.patch.object(
            model._requests, "post", return_value=mock_response
        ) as mock_post:
            list(model.infer(["test prompt"]))

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(
                60,
                call_kwargs["timeout"],
                "Timeout from constructor should flow through infer()",
            )


class TestGeminiLanguageModel(absltest.TestCase):
    @mock.patch("google.genai.Client")
    def test_gemini_allowlist_filtering(self, mock_client_class):
        """Test that only allow-listed keys are passed through."""
        mock_client = mock.Mock()
        mock_client_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.text = '{"result": "test"}'
        mock_client.models.generate_content.return_value = mock_response

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            api_key="test-key",
            # Allow-listed parameters
            tools=["tool1", "tool2"],
            stop_sequences=["\n\n"],
            system_instruction="Be helpful",
            # Unknown parameters to test filtering
            unknown_param="should_be_ignored",
            another_unknown="also_ignored",
        )

        expected_extra_kwargs = {
            "tools": ["tool1", "tool2"],
            "stop_sequences": ["\n\n"],
            "system_instruction": "Be helpful",
        }
        self.assertEqual(
            expected_extra_kwargs,
            model._extra_kwargs,
            "Only allow-listed kwargs should be stored in _extra_kwargs",
        )

        prompts = ["Test prompt"]
        list(model.infer(prompts))

        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args
        config = call_args.kwargs["config"]

        for key in ["tools", "stop_sequences", "system_instruction"]:
            self.assertIn(key, config, f"Expected {key} to be in API config")
            self.assertEqual(
                expected_extra_kwargs[key],
                config[key],
                f"Config value for {key} should match what was provided",
            )

    @mock.patch("google.genai.Client")
    def test_gemini_runtime_kwargs_filtered(self, mock_client_class):
        """Test that runtime kwargs are also filtered by allow-list."""
        mock_client = mock.Mock()
        mock_client_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.text = '{"result": "test"}'
        mock_client.models.generate_content.return_value = mock_response

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            api_key="test-key",
        )

        prompts = ["Test prompt"]
        list(
            model.infer(
                prompts,
                candidate_count=2,
                safety_settings={"HARM_CATEGORY_DANGEROUS": "BLOCK_NONE"},
                unknown_runtime_param="ignored",
            )
        )

        call_args = mock_client.models.generate_content.call_args
        config = call_args.kwargs["config"]

        self.assertEqual(
            2,
            config.get("candidate_count"),
            "candidate_count should be passed through to API",
        )
        self.assertEqual(
            {"HARM_CATEGORY_DANGEROUS": "BLOCK_NONE"},
            config.get("safety_settings"),
            "safety_settings should be passed through to API",
        )
        self.assertNotIn(
            "unknown_runtime_param", config, "Unknown kwargs should be filtered out"
        )

    def test_gemini_requires_auth_config(self):
        """Test that Gemini requires either API key or Vertex AI config."""
        with self.assertRaises(exceptions.InferenceConfigError) as cm:
            gemini.GeminiLanguageModel()

        self.assertIn("Gemini models require either", str(cm.exception))
        self.assertIn("API key", str(cm.exception))
        self.assertIn("Vertex AI", str(cm.exception))

    def test_gemini_vertexai_requires_project_and_location(self):
        """Test that Vertex AI mode requires both project and location."""
        with self.assertRaises(exceptions.InferenceConfigError) as cm:
            gemini.GeminiLanguageModel(vertexai=True)

        self.assertIn("requires both project and location", str(cm.exception))

    @mock.patch("google.genai.Client")
    def test_gemini_vertexai_initialization(self, mock_client_class):
        """Test successful initialization with Vertex AI config."""
        mock_client = mock.Mock()
        mock_client_class.return_value = mock_client

        model = gemini.GeminiLanguageModel(
            vertexai=True, project="test-project", location="us-central1"
        )

        self.assertIsNone(model.api_key)
        self.assertTrue(model.vertexai)
        self.assertEqual(model.project, "test-project")
        self.assertEqual(model.location, "us-central1")
        mock_client_class.assert_called_once_with(
            api_key=None,
            vertexai=True,
            credentials=None,
            project="test-project",
            location="us-central1",
            http_options=None,
        )

    @mock.patch("absl.logging.warning")
    @mock.patch("google.genai.Client")
    def test_gemini_warns_when_both_auth_provided(
        self, mock_client_class, mock_warning
    ):
        """Test that warning is logged when both API key and Vertex AI are provided."""
        mock_client = mock.Mock()
        mock_client_class.return_value = mock_client

        gemini.GeminiLanguageModel(
            api_key="test-key",
            vertexai=True,
            project="test-project",
            location="us-central1",
        )

        mock_warning.assert_called_once()
        warning_msg = mock_warning.call_args[0][0]
        self.assertIn("Both API key and Vertex AI", warning_msg)
        self.assertIn("API key will take precedence", warning_msg)

    @mock.patch("google.genai.Client")
    def test_gemini_vertexai_with_http_options(self, mock_client_class):
        """Test that http_options are passed to genai.Client for VPC endpoints."""
        mock_client = mock.Mock()
        mock_client_class.return_value = mock_client

        http_options = {"base_url": "https://custom-vpc.p.googleapis.com"}
        model = gemini.GeminiLanguageModel(
            vertexai=True,
            project="test-project",
            location="us-central1",
            http_options=http_options,
        )

        self.assertEqual(model.http_options, http_options)
        mock_client_class.assert_called_once_with(
            api_key=None,
            vertexai=True,
            credentials=None,
            project="test-project",
            location="us-central1",
            http_options=http_options,
        )


class TestOpenAILanguageModelInference(parameterized.TestCase):
    @parameterized.named_parameters(
        ("without", "test-api-key", None, "gpt-4o-mini", 0.5),
        ("with", "test-api-key", "http://127.0.0.1:9001/v1", "gpt-4o-mini", 0.5),
    )
    @mock.patch("openai.OpenAI")
    def test_openai_infer_with_parameters(
        self, api_key, base_url, model_id, temperature, mock_openai_class
    ):
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.choices = [
            mock.Mock(message=mock.Mock(content='{"name": "John", "age": 30}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        model = openai.OpenAILanguageModel(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

        batch_prompts = ["Extract name and age from: John is 30 years old"]
        results = list(model.infer(batch_prompts))

        # JSON format adds a system message; only explicitly set params are passed
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["model"], "gpt-4o-mini")
        self.assertEqual(call_args.kwargs["temperature"], temperature)
        self.assertEqual(call_args.kwargs["n"], 1)
        self.assertEqual(len(call_args.kwargs["messages"]), 2)
        self.assertEqual(call_args.kwargs["messages"][0]["role"], "system")
        self.assertEqual(call_args.kwargs["messages"][1]["role"], "user")

        expected_results = [
            [types.ScoredOutput(score=1.0, output='{"name": "John", "age": 30}')]
        ]
        self.assertEqual(results, expected_results)


class TestOpenAILanguageModel(absltest.TestCase):
    def test_openai_parse_output_json(self):
        model = openai.OpenAILanguageModel(
            api_key="test-key", format_type=data.FormatType.JSON
        )

        output = '{"key": "value", "number": 42}'
        parsed = model.parse_output(output)
        self.assertEqual(parsed, {"key": "value", "number": 42})

        with self.assertRaises(ValueError) as context:
            model.parse_output("invalid json")
        self.assertIn("Failed to parse output as JSON", str(context.exception))

    def test_openai_parse_output_yaml(self):
        model = openai.OpenAILanguageModel(
            api_key="test-key", format_type=data.FormatType.YAML
        )

        output = "key: value\nnumber: 42"
        parsed = model.parse_output(output)
        self.assertEqual(parsed, {"key": "value", "number": 42})

        with self.assertRaises(ValueError) as context:
            model.parse_output("invalid: yaml: bad")
        self.assertIn("Failed to parse output as YAML", str(context.exception))

    def test_openai_no_api_key_raises_error(self):
        with self.assertRaises(exceptions.InferenceConfigError) as context:
            openai.OpenAILanguageModel(api_key=None)
        self.assertEqual(str(context.exception), "API key not provided.")

    @mock.patch("openai.OpenAI")
    def test_openai_extra_kwargs_passed(self, mock_openai_class):
        """Test that extra kwargs are passed to OpenAI API."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.choices = [
            mock.Mock(message=mock.Mock(content='{"result": "test"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        model = openai.OpenAILanguageModel(
            api_key="test-key",
            frequency_penalty=0.5,
            presence_penalty=0.7,
            seed=42,
        )

        list(model.infer(["test prompt"]))

        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["frequency_penalty"], 0.5)
        self.assertEqual(call_args.kwargs["presence_penalty"], 0.7)
        self.assertEqual(call_args.kwargs["seed"], 42)

    @mock.patch("openai.OpenAI")
    def test_openai_runtime_kwargs_override(self, mock_openai_class):
        """Test that runtime kwargs override stored kwargs."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.choices = [
            mock.Mock(message=mock.Mock(content='{"result": "test"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        model = openai.OpenAILanguageModel(
            api_key="test-key",
            temperature=0.5,
            seed=123,
        )

        list(model.infer(["test prompt"], temperature=0.8, seed=456))
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["temperature"], 0.8)
        self.assertEqual(call_args.kwargs["seed"], 456)

    @mock.patch("openai.OpenAI")
    def test_openai_json_response_format(self, mock_openai_class):
        """Test that JSON format adds response_format parameter."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.choices = [
            mock.Mock(message=mock.Mock(content='{"result": "test"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        model = openai.OpenAILanguageModel(
            api_key="test-key", format_type=data.FormatType.JSON
        )

        list(model.infer(["test prompt"]))

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["response_format"], {"type": "json_object"})

    @mock.patch("openai.OpenAI")
    def test_openai_temperature_zero(self, mock_openai_class):
        """Verify temperature=0.0 is properly passed to the API."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.choices = [
            mock.Mock(message=mock.Mock(content='{"result": "test"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        model = openai.OpenAILanguageModel(api_key="test-key", temperature=0.0)

        list(model.infer(["test prompt"]))

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["temperature"], 0.0)
        self.assertEqual(call_args.kwargs["model"], "gpt-4o-mini")
        self.assertEqual(call_args.kwargs["n"], 1)

    @mock.patch("openai.OpenAI")
    def test_openai_temperature_none_not_sent(self, mock_openai_class):
        """Test that temperature=None is not sent to the API."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.choices = [
            mock.Mock(message=mock.Mock(content='{"result": "test"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test with temperature=None in model init
        model = openai.OpenAILanguageModel(
            api_key="test-key",
            temperature=None,
        )

        list(model.infer(["test prompt"]))

        call_args = mock_client.chat.completions.create.call_args
        self.assertNotIn("temperature", call_args.kwargs)

    @mock.patch("openai.OpenAI")
    def test_openai_none_values_filtered(self, mock_openai_class):
        """Test that None values are not passed to the API."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.choices = [
            mock.Mock(message=mock.Mock(content='{"result": "test"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        model = openai.OpenAILanguageModel(
            api_key="test-key",
            top_p=0.9,
        )

        list(model.infer(["test prompt"], top_p=None, seed=None))

        call_args = mock_client.chat.completions.create.call_args
        self.assertNotIn("top_p", call_args.kwargs)
        self.assertNotIn("seed", call_args.kwargs)

    @mock.patch("openai.OpenAI")
    def test_openai_no_system_message_when_not_json_yaml(self, mock_openai_class):
        """Test that no system message is sent when format_type is not JSON/YAML."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock(message=mock.Mock(content="test output"))]
        mock_client.chat.completions.create.return_value = mock_response

        model = openai.OpenAILanguageModel(
            api_key="test-key",
            format_type=None,
        )

        list(model.infer(["test prompt"]))

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "test prompt")

    @mock.patch("google.genai.Client")
    def test_gemini_none_values_filtered(self, mock_client_class):
        """Test that None values are not passed to Gemini API."""
        mock_client = mock.Mock()
        mock_client_class.return_value = mock_client

        mock_response = mock.Mock()
        mock_response.text = '{"result": "test"}'
        mock_client.models.generate_content.return_value = mock_response

        model = gemini.GeminiLanguageModel(
            model_id="gemini-2.5-flash",
            api_key="test-key",
        )

        list(model.infer(["test prompt"], candidate_count=None))

        call_args = mock_client.models.generate_content.call_args
        config = call_args.kwargs["config"]

        self.assertNotIn("candidate_count", config)


if __name__ == "__main__":
    absltest.main()
