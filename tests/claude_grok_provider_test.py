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

"""Routing and factory tests for current Claude 5 and Grok 4.6 providers."""

from __future__ import annotations

from unittest import mock

from absl.testing import parameterized

from langextract import factory
from langextract import providers as providers_module
from langextract.core import exceptions
from langextract.providers import anthropic as anthropic_provider
from langextract.providers import latest_models
from langextract.providers import router
from langextract.providers import xai as xai_provider


class ClaudeGrokProviderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    router.clear()
    providers_module._reset_for_testing()
    providers_module.load_builtins_once()

  def tearDown(self):
    super().tearDown()
    router.clear()
    providers_module._reset_for_testing()

  @parameterized.named_parameters(
      ('sonnet5', 'claude-sonnet-5'),
      ('opus5', 'claude-opus-5'),
      ('fable5', 'claude-fable-5'),
      ('haiku45', 'claude-haiku-4-5'),
      ('haiku_dated', 'claude-haiku-4-5-20251001'),
  )
  def test_claude_model_ids_route_to_anthropic(self, model_id):
    resolved = router.resolve(model_id)
    self.assertIs(resolved, anthropic_provider.AnthropicLanguageModel)

  @parameterized.named_parameters(
      ('grok46', 'grok-4.6'),
      ('grok45', 'grok-4.5'),
      ('grok43', 'grok-4.3'),
      ('grok_build', 'grok-build'),
  )
  def test_grok_model_ids_route_to_xai(self, model_id):
    resolved = router.resolve(model_id)
    self.assertIs(resolved, xai_provider.XAILanguageModel)

  def test_latest_model_constants(self):
    self.assertEqual(latest_models.CLAUDE_DEFAULT, 'claude-sonnet-5')
    self.assertEqual(latest_models.CLAUDE_FLAGSHIP, 'claude-opus-5')
    self.assertEqual(latest_models.GROK_DEFAULT, 'grok-4.6')

  def test_factory_reads_anthropic_api_key(self):
    with mock.patch.dict(
        'os.environ', {'ANTHROPIC_API_KEY': 'sk-ant-test'}, clear=False
    ):
      kwargs = factory._kwargs_with_environment_defaults('claude-sonnet-5', {})
    self.assertEqual(kwargs['api_key'], 'sk-ant-test')

  def test_factory_reads_xai_api_key_and_base_url(self):
    with mock.patch.dict(
        'os.environ', {'XAI_API_KEY': 'xai-test'}, clear=False
    ):
      kwargs = factory._kwargs_with_environment_defaults('grok-4.6', {})
    self.assertEqual(kwargs['api_key'], 'xai-test')
    self.assertEqual(kwargs['base_url'], 'https://api.x.ai/v1')

  def test_anthropic_requires_api_key(self):
    with mock.patch.dict('os.environ', {}, clear=True):
      with mock.patch.dict('sys.modules', {'anthropic': mock.MagicMock()}):
        with self.assertRaises(exceptions.InferenceConfigError):
          anthropic_provider.AnthropicLanguageModel(model_id='claude-sonnet-5')

  def test_anthropic_infer_flattens_text_blocks(self):
    fake_anthropic = mock.MagicMock()
    block = mock.Mock()
    block.text = '{"items":[]}'
    fake_anthropic.Anthropic.return_value.messages.create.return_value.content = [
        block
    ]
    with mock.patch.dict('sys.modules', {'anthropic': fake_anthropic}):
      model = anthropic_provider.AnthropicLanguageModel(
          model_id='claude-sonnet-5', api_key='sk-ant-test'
      )
      outputs = list(model.infer(['extract revenue']))
    self.assertEqual(outputs[0][0].output, '{"items":[]}')
    call_kwargs = (
        fake_anthropic.Anthropic.return_value.messages.create.call_args.kwargs
    )
    self.assertEqual(call_kwargs['model'], 'claude-sonnet-5')
    self.assertIn('max_tokens', call_kwargs)

  def test_xai_uses_grok_default_and_xai_base_url(self):
    fake_openai = mock.MagicMock()
    with mock.patch.dict('sys.modules', {'openai': fake_openai}):
      model = xai_provider.XAILanguageModel(api_key='xai-test')
    self.assertEqual(model.model_id, 'grok-4.6')
    self.assertEqual(model.base_url, 'https://api.x.ai/v1')
    fake_openai.OpenAI.assert_called_once()
    _, kwargs = fake_openai.OpenAI.call_args
    self.assertEqual(kwargs['base_url'], 'https://api.x.ai/v1')
    self.assertEqual(kwargs['api_key'], 'xai-test')
