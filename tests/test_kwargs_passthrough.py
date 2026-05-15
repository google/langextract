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

"""Tests for enhanced kwargs pass-through in providers."""

import unittest
from unittest import mock
import warnings

from absl.testing import parameterized

from langextract import factory
from langextract.core import data
from langextract.core import exceptions
from langextract.providers import ollama
from langextract.providers import openai
from langextract.providers import schemas


def _configure_openai_mock(mock_openai_class, content='{"result": "test"}'):
  mock_client = mock.Mock()
  mock_openai_class.return_value = mock_client
  mock_response = mock.Mock()
  mock_response.choices = [mock.Mock(message=mock.Mock(content=content))]
  mock_client.chat.completions.create.return_value = mock_response
  return mock_client


def _condition_examples(attributes=None):
  extraction_kwargs = {
      'extraction_text': 'diabetes',
      'extraction_class': 'condition',
  }
  if attributes is not None:
    extraction_kwargs['attributes'] = attributes
  return [
      data.ExampleData(
          text='Patient has diabetes.',
          extractions=[data.Extraction(**extraction_kwargs)],
      )
  ]


class TestOpenAIKwargsPassthrough(unittest.TestCase):
  """Test OpenAI provider's enhanced kwargs handling."""

  @mock.patch('openai.OpenAI', autospec=True)
  def test_reasoning_effort_passed_as_top_level(self, mock_openai_class):
    """reasoning_effort is passed as a top-level Chat Completions parameter."""
    mock_client = _configure_openai_mock(mock_openai_class)

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o-mini',
        api_key='test-key',
        reasoning_effort='low',
    )

    list(model.infer(['test prompt']))

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(call_args.kwargs.get('reasoning_effort'), 'low')
    self.assertNotIn('reasoning', call_args.kwargs)

  @mock.patch('openai.OpenAI', autospec=True)
  def test_runtime_reasoning_effort_override(self, mock_openai_class):
    """Runtime reasoning_effort overrides constructor value."""
    mock_client = _configure_openai_mock(mock_openai_class)

    model = openai.OpenAILanguageModel(
        model_id='o4-mini',
        api_key='test-key',
        reasoning_effort='low',
    )

    list(model.infer(['test prompt'], reasoning_effort='high'))

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(call_args.kwargs.get('reasoning_effort'), 'high')

  @mock.patch('openai.OpenAI', autospec=True)
  def test_runtime_kwargs_override_stored(self, mock_openai_class):
    """Runtime parameters should override constructor parameters."""
    mock_client = _configure_openai_mock(mock_openai_class)

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o-mini',
        api_key='test-key',
        temperature=0.7,
        top_p=0.9,
    )

    list(model.infer(['test prompt'], temperature=0.3, seed=42))

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(
        {
            key: call_args.kwargs.get(key)
            for key in ('temperature', 'top_p', 'seed')
        },
        {'temperature': 0.3, 'top_p': 0.9, 'seed': 42},
    )

  @mock.patch('openai.OpenAI', autospec=True)
  def test_falsy_values_preserved(self, mock_openai_class):
    """Falsy values like 0 should be preserved, not filtered as None."""
    mock_client = _configure_openai_mock(mock_openai_class)

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o',
        api_key='test-key',
        temperature=0,
        top_logprobs=0,
    )

    list(model.infer(['test prompt']))

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(
        {
            key: call_args.kwargs.get(key)
            for key in ('temperature', 'top_logprobs')
        },
        {'temperature': 0, 'top_logprobs': 0},
    )

  @mock.patch('openai.OpenAI', autospec=True)
  def test_reasoning_effort_not_nested(self, mock_openai_class):
    """reasoning_effort should not be converted to a nested reasoning dict."""
    mock_client = _configure_openai_mock(mock_openai_class)

    model = openai.OpenAILanguageModel(
        model_id='o4-mini',
        api_key='test-key',
        reasoning_effort='medium',
    )

    list(model.infer(['test prompt']))

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(call_args.kwargs.get('reasoning_effort'), 'medium')
    self.assertNotIn('reasoning', call_args.kwargs)

  @mock.patch('openai.OpenAI', autospec=True)
  def test_custom_response_format(self, mock_openai_class):
    """Custom response_format should override default JSON format."""
    mock_client = _configure_openai_mock(mock_openai_class)

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o',
        api_key='test-key',
        format_type=openai.data.FormatType.JSON,
    )

    list(
        model.infer(
            ['test prompt'],
            response_format={'type': 'text', 'schema': 'custom'},
        )
    )

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(
        call_args.kwargs.get('response_format'),
        {'type': 'text', 'schema': 'custom'},
    )

  @mock.patch('openai.OpenAI', autospec=True)
  def test_schema_response_format_passed_to_chat_completion(
      self, mock_openai_class
  ):
    """OpenAI schema constraints use structured output response_format."""
    mock_client = _configure_openai_mock(
        mock_openai_class, content='{"extractions": []}'
    )

    config = factory.ModelConfig(
        model_id='gpt-4o-mini', provider_kwargs={'api_key': 'test-key'}
    )
    model = factory.create_model(
        config,
        examples=_condition_examples(attributes={'chronicity': 'chronic'}),
        use_schema_constraints=True,
        fence_output=None,
    )
    self.assertIsInstance(model.schema, schemas.openai.OpenAISchema)
    self.assertIs(model.openai_schema, model.schema)

    list(model.infer(['test prompt']))

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(
        call_args.kwargs.get('response_format'),
        {
            'type': 'json_schema',
            'json_schema': {
                'name': 'langextract_extractions',
                'strict': True,
                'schema': mock.ANY,
            },
        },
    )

  @mock.patch('openai.OpenAI', autospec=True)
  def test_runtime_response_format_overrides_schema(self, mock_openai_class):
    """Runtime response_format wins over schema defaults."""
    mock_client = _configure_openai_mock(
        mock_openai_class, content='{"extractions": []}'
    )

    config = factory.ModelConfig(
        model_id='gpt-4o-mini', provider_kwargs={'api_key': 'test-key'}
    )
    model = factory.create_model(
        config,
        examples=_condition_examples(),
        use_schema_constraints=True,
        fence_output=None,
    )

    with self.assertWarnsRegex(UserWarning, 'schema is bypassed for this call'):
      list(
          model.infer(['test prompt'], response_format={'type': 'json_object'})
      )

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(
        call_args.kwargs.get('response_format'), {'type': 'json_object'}
    )

  @mock.patch('openai.OpenAI', autospec=True)
  def test_apply_schema_rejects_non_openai_schema(self, mock_openai_class):
    """apply_schema rejects foreign BaseSchema subclasses explicitly."""
    mock_openai_class.return_value = mock.Mock()

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o-mini', api_key='test-key'
    )
    gemini_schema = schemas.gemini.GeminiSchema.from_examples([])

    with self.assertRaisesRegex(
        exceptions.InferenceConfigError,
        'only accepts OpenAISchema instances',
    ):
      model.apply_schema(gemini_schema)

    self.assertIsNone(model.openai_schema)
    self.assertIsNone(model.schema)

  @mock.patch('openai.OpenAI', autospec=True)
  def test_apply_schema_rejection_preserves_prior_schema(
      self, mock_openai_class
  ):
    """Rejected foreign schemas leave the active OpenAI schema unchanged."""
    mock_openai_class.return_value = mock.Mock()

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o-mini', api_key='test-key'
    )
    openai_schema = schemas.openai.OpenAISchema.from_examples([])
    model.apply_schema(openai_schema)
    gemini_schema = schemas.gemini.GeminiSchema.from_examples([])

    with self.assertRaisesRegex(
        exceptions.InferenceConfigError,
        'only accepts OpenAISchema instances',
    ):
      model.apply_schema(gemini_schema)

    self.assertIs(model.openai_schema, openai_schema)
    self.assertIs(model.schema, openai_schema)

  @mock.patch('openai.OpenAI', autospec=True)
  def test_apply_schema_none_clears_response_format(self, mock_openai_class):
    """Clearing an OpenAI schema falls back to regular JSON mode."""
    mock_client = _configure_openai_mock(
        mock_openai_class, content='{"extractions": []}'
    )

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o-mini', api_key='test-key'
    )
    model.apply_schema(schemas.openai.OpenAISchema.from_examples([]))
    model.apply_schema(None)
    self.assertIsNone(model.openai_schema)
    self.assertIsNone(model.schema)

    list(model.infer(['test prompt']))

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(
        call_args.kwargs.get('response_format'), {'type': 'json_object'}
    )

  @mock.patch('openai.OpenAI', autospec=True)
  def test_factory_schema_clear_removes_response_format(
      self, mock_openai_class
  ):
    """Clearing a factory-created schema falls back to regular JSON mode."""
    mock_client = _configure_openai_mock(
        mock_openai_class, content='{"extractions": []}'
    )

    config = factory.ModelConfig(
        model_id='gpt-4o-mini', provider_kwargs={'api_key': 'test-key'}
    )
    model = factory.create_model(
        config,
        examples=_condition_examples(),
        use_schema_constraints=True,
        fence_output=None,
    )
    model.apply_schema(None)
    self.assertIsNone(model.openai_schema)
    self.assertIsNone(model.schema)

    list(model.infer(['test prompt']))

    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(
        call_args.kwargs.get('response_format'), {'type': 'json_object'}
    )

  @mock.patch('openai.OpenAI', autospec=True)
  def test_apply_schema_none_preserves_explicit_fence_output(
      self, mock_openai_class
  ):
    """Schema clearing does not erase the caller's fence preference."""
    mock_openai_class.return_value = mock.Mock()

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o-mini', api_key='test-key'
    )
    model.apply_schema(schemas.openai.OpenAISchema.from_examples([]))
    model.set_fence_output(True)

    model.apply_schema(None)

    self.assertIs(model.requires_fence_output, True)

  @mock.patch('openai.OpenAI', autospec=True)
  def test_apply_schema_rejects_yaml_format(self, mock_openai_class):
    """OpenAI structured outputs fail fast for YAML format."""
    mock_openai_class.return_value = mock.Mock()

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o-mini',
        api_key='test-key',
        format_type=data.FormatType.YAML,
    )

    with self.assertRaisesRegex(
        exceptions.InferenceConfigError,
        'OpenAI structured output only supports JSON format',
    ):
      model.apply_schema(schemas.openai.OpenAISchema.from_examples([]))
    self.assertIsNone(model.schema)
    self.assertIsNone(model.openai_schema)

  @mock.patch('openai.OpenAI', autospec=True)
  def test_constructor_schema_populates_public_schema(self, mock_openai_class):
    """Constructor schema support matches apply_schema state."""
    mock_openai_class.return_value = mock.Mock()

    openai_schema = schemas.openai.OpenAISchema.from_examples([])
    model = openai.OpenAILanguageModel(
        model_id='gpt-4o-mini',
        api_key='test-key',
        openai_schema=openai_schema,
    )

    self.assertIs(model.openai_schema, openai_schema)
    self.assertIs(model.schema, openai_schema)

  @mock.patch('openai.OpenAI', autospec=True)
  def test_inference_preserves_schema_config_error(self, mock_openai_class):
    """Late schema configuration errors keep their config exception type."""
    mock_openai_class.return_value = mock.Mock()

    model = openai.OpenAILanguageModel(
        model_id='gpt-4o-mini', api_key='test-key'
    )
    model.apply_schema(schemas.openai.OpenAISchema.from_examples([]))
    model.format_type = data.FormatType.YAML

    with self.assertRaisesRegex(
        exceptions.InferenceConfigError,
        'OpenAI structured output only supports JSON format',
    ):
      list(model.infer(['test prompt']))

  @mock.patch('openai.OpenAI', autospec=True)
  def test_reasoning_not_in_chat_completions(self, mock_openai_class):
    """reasoning dict is not forwarded to Chat Completions API."""
    mock_client = _configure_openai_mock(mock_openai_class)

    model = openai.OpenAILanguageModel(
        model_id='o4-mini',
        api_key='test-key',
    )

    list(model.infer(['test prompt'], reasoning={'effort': 'low'}))

    call_args = mock_client.chat.completions.create.call_args
    self.assertNotIn('reasoning', call_args.kwargs)


class TestOllamaAuthSupport(parameterized.TestCase):
  """Test Ollama provider's authentication support for proxied instances."""

  @mock.patch('requests.post')
  def test_api_key_in_authorization_header(self, mock_post):
    """API key should be sent in Authorization header with Bearer scheme."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': '{"test": "value"}'}
    mock_post.return_value = mock_response

    model = ollama.OllamaLanguageModel(
        model_id='gemma2:2b',
        model_url='https://proxy.example.com',
        api_key='sk-test-key-123',
    )

    list(model.infer(['test prompt']))

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    headers = call_args.kwargs.get('headers', {})
    self.assertEqual(headers.get('Authorization'), 'Bearer sk-test-key-123')
    self.assertEqual(headers.get('Content-Type'), 'application/json')

  @mock.patch('requests.post')
  def test_custom_auth_header_name(self, mock_post):
    """Custom auth header name (e.g. X-API-Key) should be supported."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': '{"test": "value"}'}
    mock_post.return_value = mock_response

    model = ollama.OllamaLanguageModel(
        model_id='gemma2:2b',
        model_url='https://api.example.com',
        api_key='abc123',
        auth_header='X-API-Key',
        auth_scheme='',
    )

    list(model.infer(['test prompt']))

    headers = mock_post.call_args.kwargs.get('headers', {})
    self.assertEqual(headers.get('X-API-Key'), 'abc123')
    self.assertNotIn('Authorization', headers)

  @mock.patch('requests.post')
  def test_pass_through_kwargs(self, mock_post):
    """Future Ollama parameters should pass through without code changes."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': '{"test": "value"}'}
    mock_post.return_value = mock_response

    model = ollama.OllamaLanguageModel(
        model_id='mistral:7b',
        temperature=0.5,
        top_k=40,
        repeat_penalty=1.1,
        mirostat=2,
    )

    list(model.infer(['test prompt']))

    call_args = mock_post.call_args
    payload = call_args.kwargs['json']
    options = payload['options']

    self.assertEqual(options.get('temperature'), 0.5)
    self.assertEqual(options.get('top_k'), 40)
    self.assertEqual(options.get('repeat_penalty'), 1.1)
    self.assertEqual(options.get('mirostat'), 2)

  def test_api_key_redacted_in_repr(self):
    """API key should be redacted in string representation for security."""
    model = ollama.OllamaLanguageModel(
        model_id='gemma2:2b',
        api_key='super-secret-key',
    )

    repr_str = repr(model)
    self.assertIn('[REDACTED]', repr_str, 'API key should be redacted')
    self.assertNotIn(
        'super-secret-key', repr_str, 'Actual API key should not appear'
    )

  @mock.patch('requests.post')
  def test_localhost_auth_warning_but_still_works(self, mock_post):
    """Should warn about localhost auth but still send the auth header."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': '{"test": "value"}'}
    mock_post.return_value = mock_response

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      model = ollama.OllamaLanguageModel(
          model_id='gemma2:2b',
          model_url='http://localhost:11434',
          api_key='unnecessary-key',
      )

      self.assertTrue(
          any('localhost' in str(warning.message) for warning in w),
          'Expected warning about localhost auth',
      )

    # Verify auth header is still sent despite warning
    list(model.infer(['test prompt']))
    headers = mock_post.call_args.kwargs.get('headers', {})
    self.assertEqual(headers.get('Authorization'), 'Bearer unnecessary-key')

  @mock.patch('requests.post')
  def test_runtime_kwargs_override(self, mock_post):
    """Runtime parameters should override constructor parameters."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': '{"test": "value"}'}
    mock_post.return_value = mock_response

    model = ollama.OllamaLanguageModel(
        model_id='gemma2:2b',
        temperature=0.7,
        timeout=60,
    )

    list(model.infer(['test prompt'], temperature=0.3, timeout=120))

    call_args = mock_post.call_args
    payload = call_args.kwargs['json']
    options = payload['options']

    self.assertEqual(options.get('temperature'), 0.3)
    self.assertEqual(call_args.kwargs.get('timeout'), 120)

  @parameterized.named_parameters(
      ('https_localhost', 'https://localhost:11434', True),
      ('ipv6_localhost', 'http://[::1]:11434', True),
      ('ipv4_localhost', 'http://127.0.0.1:8080/', True),
      ('remote_proxy', 'https://proxy.example.com', False),
  )
  @mock.patch('requests.post')
  def test_localhost_detection(self, url, should_warn, mock_post):
    """Should detect localhost in various URL formats (IPv6, https, etc)."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': '{"test": "value"}'}
    mock_post.return_value = mock_response

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      _ = ollama.OllamaLanguageModel(
          model_id='gemma2:2b',
          model_url=url,
          api_key='test-key',
      )

      if should_warn:
        self.assertTrue(
            any('localhost' in str(warning.message) for warning in w),
            f'Expected warning for {url}',
        )
      else:
        self.assertFalse(
            any('localhost' in str(warning.message) for warning in w),
            f'Unexpected warning for {url}',
        )

  @mock.patch('requests.post')
  def test_format_none_not_in_payload(self, mock_post):
    """Format key should be omitted from payload when None (not sent as null)."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': 'plain text'}
    mock_post.return_value = mock_response

    model = ollama.OllamaLanguageModel(
        model_id='gemma2:2b',
    )

    model.format_type = None

    _ = model._ollama_query(
        prompt='test prompt',
        model='gemma2:2b',
        structured_output_format=None,
    )

    call_args = mock_post.call_args
    payload = call_args.kwargs['json']

    self.assertNotIn('format', payload, 'format=None should not be in payload')

  @mock.patch('requests.post')
  def test_reserved_kwargs_not_in_options(self, mock_post):
    """Reserved top-level keys (stop, format) should not go into options dict."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': '{"test": "value"}'}
    mock_post.return_value = mock_response

    model = ollama.OllamaLanguageModel(
        model_id='gemma2:2b',
        stop=['END'],
        temperature=0.5,
        custom_param='value',
    )

    list(model.infer(['test prompt']))

    call_args = mock_post.call_args
    payload = call_args.kwargs['json']
    options = payload['options']

    self.assertEqual(payload.get('stop'), ['END'])
    self.assertNotIn(
        'stop', options, 'stop should be at top level, not in options'
    )
    self.assertEqual(options.get('temperature'), 0.5)
    self.assertEqual(options.get('custom_param'), 'value')

  @mock.patch('requests.post')
  def test_api_key_without_localhost_warning(self, mock_post):
    """Should not warn when using auth with remote/proxied Ollama instances."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': '{"test": "value"}'}
    mock_post.return_value = mock_response

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      model = ollama.OllamaLanguageModel(
          model_id='gemma2:2b',
          model_url='https://proxy.example.com',
          api_key='necessary-key',
      )

      self.assertFalse(
          any('localhost' in str(warning.message) for warning in w)
      )

    list(model.infer(['test prompt']))
    headers = mock_post.call_args.kwargs.get('headers', {})
    self.assertEqual(headers.get('Authorization'), 'Bearer necessary-key')


if __name__ == '__main__':
  unittest.main()
