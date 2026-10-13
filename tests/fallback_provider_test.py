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

"""Tests for the fallback provider."""

from __future__ import annotations

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from langextract import factory
from langextract import providers
import langextract as lx
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema as schema_lib
from langextract.core import types
from langextract.providers import fallback
from langextract.providers import router

_ALICE_OUTPUT = '{"extractions": [{"person": "Alice"}]}'
_BOB_OUTPUT = '{"extractions": [{"person": "Bob"}]}'
# Every public BaseLanguageModel member must be listed in exactly one of these
# sets, so a new base member fails the guard test until someone decides
# whether the fallback wrapper has to forward it.
_FORWARDED_BASE_MEMBERS = frozenset({
    "apply_output_schema",
    "requires_fence_output",
    "schema",
    "set_fence_output",
})
_NOT_FORWARDED_BASE_MEMBERS = frozenset({
    "apply_schema",
    "get_schema_class",
    "infer",
    "infer_batch",
    "merge_kwargs",
    "parse_output",
})


class _StatusError(Exception):
  """HTTP-style error carrying `status_code`."""

  def __init__(self, status_code):
    super().__init__(f"status {status_code}")
    self.status_code = status_code


class _CodeError(Exception):
  """genai-style error carrying `code`."""

  def __init__(self, code):
    super().__init__(f"code {code}")
    self.code = code


class ReadTimeout(OSError):
  """requests-style timeout: the class name matters, not the base."""


def _runtime_error(original=None, message="inference failed"):
  return exceptions.InferenceRuntimeError(message, original=original)


def _with_retryable(error, retryable):
  error.retryable = retryable
  return error


def _scored(output, count=1):
  return [[types.ScoredOutput(score=1.0, output=output)] for _ in range(count)]


class _PartialFailure:
  """Outcome: yield the first prompt's output, then raise `error`."""

  def __init__(self, error):
    self.error = error


class _ScriptedModel(base_model.BaseLanguageModel):
  """Fake provider that plays back one outcome per `infer` call.

  The last outcome repeats once the list is used up. A `str` outcome answers
  every prompt, an `Exception` is raised on the first `next()`, and a
  `_PartialFailure` answers the first prompt before raising.
  """

  def __init__(self, model_id="scripted-a", outcomes=(_ALICE_OUTPUT,)):
    super().__init__()
    self.model_id = model_id
    self.outcomes = outcomes
    self.infer_calls = 0
    self.infer_kwargs = []
    self.output_schemas = []

  def infer(self, batch_prompts, **kwargs):
    outcome = self.outcomes[min(self.infer_calls, len(self.outcomes) - 1)]
    self.infer_calls += 1
    self.infer_kwargs.append(kwargs)
    if isinstance(outcome, Exception):
      raise outcome
    if isinstance(outcome, _PartialFailure):
      yield [types.ScoredOutput(score=1.0, output=_ALICE_OUTPUT)]
      raise outcome.error
    for _ in batch_prompts:
      yield [types.ScoredOutput(score=1.0, output=outcome)]

  def apply_output_schema(self, output_schema):
    self.output_schemas.append(output_schema)


def _base_member_handling(wrapper_class):
  abstract_members = base_model.BaseLanguageModel.__abstractmethods__
  return {
      name: (
          "forwarded"
          if name in vars(wrapper_class) and name not in abstract_members
          else "not forwarded"
      )
      for name in dir(base_model.BaseLanguageModel)
      if not name.startswith("_")
  }


_RATE_LIMIT_ERROR = _runtime_error(_StatusError(429))


class _FallbackTestCase(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    router.clear()
    self.addCleanup(router.clear)
    self.enter_context(
        mock.patch.object(providers, "load_plugins_once", autospec=True)
    )
    router.register(r"^scripted-")(_ScriptedModel)
    self.sleeps = []
    self.retry_policy = fallback.RetryPolicy(max_attempts=1)

  def _wrap(self, *models, retry_policy=None):
    return fallback.FallbackLanguageModel(
        models,
        retry_policy=retry_policy or self.retry_policy,
        sleep=self.sleeps.append,
    )

  def _rate_limited_primary_configs(self):
    return [
        factory.ModelConfig(
            model_id="scripted-a",
            provider_kwargs={"outcomes": (_RATE_LIMIT_ERROR,)},
        ),
        factory.ModelConfig(
            model_id="scripted-b",
            provider_kwargs={"outcomes": (_BOB_OUTPUT,)},
        ),
    ]


class FallbackOrderTest(_FallbackTestCase):

  def test_infer_returns_primary_outputs_when_primary_succeeds(self):
    first = _ScriptedModel("scripted-a")
    second = _ScriptedModel("scripted-b", (_BOB_OUTPUT,))
    wrapper = self._wrap(first, second)

    outputs = list(wrapper.infer(["p1"]))

    self.assertEqual(outputs, _scored(_ALICE_OUTPUT))
    self.assertEqual(second.infer_calls, 0)
    self.assertEqual(wrapper.served_model_ids, ("scripted-a",))

  @parameterized.named_parameters(
      dict(
          testcase_name="status_429",
          error=_runtime_error(_StatusError(429)),
      ),
      dict(
          testcase_name="status_503",
          error=_runtime_error(_StatusError(503)),
      ),
      dict(
          testcase_name="genai_code_429",
          error=_runtime_error(_CodeError(429)),
      ),
      dict(
          testcase_name="builtin_timeout",
          error=_runtime_error(TimeoutError()),
      ),
      dict(
          testcase_name="connection_error",
          error=_runtime_error(ConnectionError()),
      ),
      dict(
          testcase_name="requests_read_timeout",
          error=_runtime_error(ReadTimeout()),
      ),
      dict(
          testcase_name="explicit_retryable",
          error=_with_retryable(_runtime_error(), True),
      ),
      dict(
          testcase_name="unwrapped_timeout",
          error=TimeoutError(),
      ),
  )
  def test_infer_falls_back_on_transient_error(self, error):
    first = _ScriptedModel("scripted-a", (error,))
    second = _ScriptedModel("scripted-b", (_BOB_OUTPUT,))

    outputs = list(self._wrap(first, second).infer(["p1"]))

    self.assertEqual(outputs, _scored(_BOB_OUTPUT))
    self.assertEqual(first.infer_calls, 1)
    self.assertEqual(second.infer_calls, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="status_400",
          error=_runtime_error(_StatusError(400)),
      ),
      dict(
          testcase_name="status_401",
          error=_runtime_error(_StatusError(401)),
      ),
      dict(
          testcase_name="config_error",
          error=exceptions.InferenceConfigError("bad api key"),
      ),
      dict(
          testcase_name="message_only_429",
          error=_runtime_error(message="429 Too Many Requests"),
      ),
      dict(
          testcase_name="explicit_not_retryable",
          error=_with_retryable(_runtime_error(_StatusError(503)), False),
      ),
      dict(
          testcase_name="value_error",
          error=ValueError("bad prompt"),
      ),
  )
  def test_infer_raises_non_transient_error_without_fallback(self, error):
    first = _ScriptedModel("scripted-a", (error,))
    second = _ScriptedModel("scripted-b", (_BOB_OUTPUT,))

    with self.assertRaises(type(error)) as raised:
      list(self._wrap(first, second).infer(["p1"]))

    self.assertIs(raised.exception, error)
    self.assertEqual(second.infer_calls, 0)
    self.assertEqual(self.sleeps, [])

  def test_infer_restarts_from_first_model_for_each_batch(self):
    first = _ScriptedModel("scripted-a", (_RATE_LIMIT_ERROR, _ALICE_OUTPUT))
    second = _ScriptedModel("scripted-b", (_BOB_OUTPUT,))
    wrapper = self._wrap(first, second)

    list(wrapper.infer(["p1"]))
    list(wrapper.infer(["p1"]))

    self.assertEqual(wrapper.served_model_ids, ("scripted-b", "scripted-a"))

  def test_infer_discards_partial_outputs_of_failed_model(self):
    first = _ScriptedModel("scripted-a", (_PartialFailure(_RATE_LIMIT_ERROR),))
    second = _ScriptedModel("scripted-b", (_BOB_OUTPUT,))

    outputs = list(self._wrap(first, second).infer(["p1", "p2"]))

    self.assertEqual(outputs, _scored(_BOB_OUTPUT, count=2))

  def test_infer_logs_serving_model(self):
    first = _ScriptedModel("scripted-a", (_RATE_LIMIT_ERROR,))
    second = _ScriptedModel("scripted-b", (_BOB_OUTPUT,))

    with self.assertLogs(level="INFO") as logs:
      list(self._wrap(first, second).infer(["p1"]))

    self.assertIn("served by scripted-b", "\n".join(logs.output))

  def test_infer_forwards_kwargs_to_serving_model(self):
    first = _ScriptedModel("scripted-a")

    list(self._wrap(first).infer(["p1"], temperature=0.5))

    self.assertEqual(first.infer_kwargs, [{"temperature": 0.5}])

  def test_model_id_is_first_model_label(self):
    wrapper = self._wrap(
        _ScriptedModel("scripted-a"), _ScriptedModel("scripted-b")
    )

    self.assertEqual(wrapper.model_id, "scripted-a")


class RouterIntegrationTest(_FallbackTestCase):

  def test_create_model_with_fallback_provider_serves_from_fallback(self):
    model = factory.create_model(
        factory.ModelConfig(
            provider="fallback",
            provider_kwargs={
                "models": self._rate_limited_primary_configs(),
                "retry_policy": fallback.RetryPolicy(max_attempts=1),
                "sleep": self.sleeps.append,
            },
        )
    )

    self.assertIsInstance(model, fallback.FallbackLanguageModel)
    self.assertEqual(list(model.infer(["p1"])), _scored(_BOB_OUTPUT))
    self.assertEqual(model.served_model_ids, ("scripted-b",))

  def test_extract_with_fallback_model_returns_fallback_extractions(self):
    wrapper = self._wrap(*self._rate_limited_primary_configs())

    result = lx.extract(
        "Bob went home.",
        model=wrapper,
        prompt_description="Extract people.",
        examples=[
            data.ExampleData(
                text="Carol met Dan.",
                extractions=[data.Extraction("person", "Carol")],
            ),
        ],
        use_schema_constraints=False,
        show_progress=False,
    )

    self.assertEqual(
        [
            (extraction.extraction_class, extraction.extraction_text)
            for extraction in result.extractions
        ],
        [("person", "Bob")],
    )
    self.assertEqual(wrapper.served_model_ids, ("scripted-b",))


class RetryPolicyTest(_FallbackTestCase):

  def test_infer_retries_transient_error_before_fallback(self):
    first = _ScriptedModel(
        "scripted-a", (_RATE_LIMIT_ERROR, _RATE_LIMIT_ERROR, _ALICE_OUTPUT)
    )
    second = _ScriptedModel("scripted-b", (_BOB_OUTPUT,))
    wrapper = self._wrap(
        first, second, retry_policy=fallback.RetryPolicy(max_attempts=3)
    )

    outputs = list(wrapper.infer(["p1"]))

    self.assertEqual(outputs, _scored(_ALICE_OUTPUT))
    self.assertEqual(first.infer_calls, 3)
    self.assertEqual(second.infer_calls, 0)

  def test_infer_moves_to_next_model_after_max_attempts(self):
    first = _ScriptedModel("scripted-a", (_RATE_LIMIT_ERROR,))
    second = _ScriptedModel("scripted-b", (_BOB_OUTPUT,))
    wrapper = self._wrap(
        first, second, retry_policy=fallback.RetryPolicy(max_attempts=3)
    )

    outputs = list(wrapper.infer(["p1"]))

    self.assertEqual(outputs, _scored(_BOB_OUTPUT))
    self.assertEqual(first.infer_calls, 3)
    self.assertEqual(second.infer_calls, 1)

  @parameterized.named_parameters(
      dict(testcase_name="single_attempt", max_attempts=1),
      dict(testcase_name="repeated_attempts", max_attempts=2),
  )
  def test_infer_raises_last_error_when_every_model_fails_transiently(
      self, max_attempts
  ):
    error_a = _runtime_error(_StatusError(429))
    error_b = _runtime_error(_StatusError(503))
    first = _ScriptedModel("scripted-a", (error_a,))
    second = _ScriptedModel("scripted-b", (error_b,))
    wrapper = self._wrap(
        first,
        second,
        retry_policy=fallback.RetryPolicy(max_attempts=max_attempts),
    )

    with self.assertRaises(exceptions.InferenceRuntimeError) as raised:
      list(wrapper.infer(["p1"]))

    self.assertIs(raised.exception, error_b)
    self.assertEqual(first.infer_calls, max_attempts)
    self.assertEqual(second.infer_calls, max_attempts)

  def test_infer_sleeps_capped_exponential_backoff(self):
    first = _ScriptedModel("scripted-a", (_RATE_LIMIT_ERROR,))
    second = _ScriptedModel("scripted-b", (_BOB_OUTPUT,))
    wrapper = self._wrap(
        first,
        second,
        retry_policy=fallback.RetryPolicy(
            max_attempts=4, initial_delay_seconds=1.0, max_delay_seconds=3.0
        ),
    )

    list(wrapper.infer(["p1"]))

    self.assertEqual(self.sleeps, [1.0, 2.0, 3.0])

  def test_infer_does_not_sleep_after_success(self):
    first = _ScriptedModel("scripted-a", (_RATE_LIMIT_ERROR, _ALICE_OUTPUT))
    wrapper = self._wrap(
        first,
        retry_policy=fallback.RetryPolicy(
            max_attempts=3, initial_delay_seconds=0.5
        ),
    )

    list(wrapper.infer(["p1"]))

    self.assertEqual(self.sleeps, [0.5])

  def test_non_transient_error_is_not_retried(self):
    error = _runtime_error(_StatusError(401))
    first = _ScriptedModel("scripted-a", (error,))
    wrapper = self._wrap(
        first, retry_policy=fallback.RetryPolicy(max_attempts=3)
    )

    with self.assertRaises(exceptions.InferenceRuntimeError) as raised:
      list(wrapper.infer(["p1"]))

    self.assertIs(raised.exception, error)
    self.assertEqual(first.infer_calls, 1)
    self.assertEqual(self.sleeps, [])

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_attempts",
          policy_kwargs=dict(max_attempts=0),
          field="max_attempts",
      ),
      dict(
          testcase_name="negative_attempts",
          policy_kwargs=dict(max_attempts=-1),
          field="max_attempts",
      ),
      dict(
          testcase_name="negative_initial_delay",
          policy_kwargs=dict(initial_delay_seconds=-0.5),
          field="initial_delay_seconds",
      ),
      dict(
          testcase_name="max_below_initial",
          policy_kwargs=dict(initial_delay_seconds=2.0, max_delay_seconds=1.0),
          field="max_delay_seconds",
      ),
  )
  def test_retry_policy_rejects_invalid_values(self, policy_kwargs, field):
    with self.assertRaisesRegex(exceptions.InferenceConfigError, field):
      fallback.RetryPolicy(**policy_kwargs)


def _turn_off_fence_of_second(first, second):
  del first
  second.set_fence_output(False)


def _make_second_yaml(first, second):
  del first
  second.format_type = types.FormatType.YAML


def _give_first_a_schema(first, second):
  del second
  first.apply_schema(schema_lib.FormatModeSchema())
  first.set_fence_output(True)


class ConfigurationTest(_FallbackTestCase):

  def test_rejects_empty_model_list(self):
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "at least one model"
    ):
      self._wrap()

  @parameterized.named_parameters(
      dict(
          testcase_name="prebuilt_fallback_model",
          build_models=lambda: [
              fallback.FallbackLanguageModel([_ScriptedModel()])
          ],
      ),
      dict(
          testcase_name="fallback_model_config",
          build_models=lambda: [
              factory.ModelConfig(
                  provider="fallback",
                  provider_kwargs={
                      "models": [factory.ModelConfig(model_id="scripted-a")]
                  },
              )
          ],
      ),
  )
  def test_rejects_nested_fallback_model(self, build_models):
    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, "cannot be nested"
    ):
      self._wrap(*build_models())

  @parameterized.named_parameters(
      dict(
          testcase_name="fence",
          setup=_turn_off_fence_of_second,
          property_name="requires_fence_output",
      ),
      dict(
          testcase_name="format_type",
          setup=_make_second_yaml,
          property_name="format_type",
      ),
      dict(
          testcase_name="schema_class",
          setup=_give_first_a_schema,
          property_name="schema",
      ),
  )
  def test_rejects_models_with_different_output_format(
      self, setup, property_name
  ):
    first = _ScriptedModel("scripted-a")
    second = _ScriptedModel("scripted-b")
    setup(first, second)

    with self.assertRaisesRegex(
        exceptions.InferenceConfigError, rf"scripted-b.*{property_name}"
    ):
      self._wrap(first, second)

  def test_accepts_models_with_same_output_format(self):
    first = _ScriptedModel("scripted-a")
    second = _ScriptedModel("scripted-b")
    first.set_fence_output(False)
    second.set_fence_output(False)

    wrapper = self._wrap(first, second)

    self.assertIs(wrapper.requires_fence_output, False)

  def test_set_fence_output_applies_to_every_model(self):
    first = _ScriptedModel("scripted-a")
    second = _ScriptedModel("scripted-b")
    wrapper = self._wrap(first, second)

    wrapper.set_fence_output(False)

    self.assertEqual(
        [
            first.requires_fence_output,
            second.requires_fence_output,
            wrapper.requires_fence_output,
        ],
        [False, False, False],
    )

  def test_schema_is_first_model_schema(self):
    first = _ScriptedModel("scripted-a")
    second = _ScriptedModel("scripted-b")
    model_schema = schema_lib.FormatModeSchema()
    first.apply_schema(model_schema)
    second.apply_schema(schema_lib.FormatModeSchema())

    wrapper = self._wrap(first, second)

    self.assertIs(wrapper.schema, model_schema)

  def test_apply_output_schema_applies_to_every_model(self):
    first = _ScriptedModel("scripted-a")
    second = _ScriptedModel("scripted-b")
    output_schema = {"type": "object"}

    self._wrap(first, second).apply_output_schema(output_schema)

    self.assertEqual(first.output_schemas, [output_schema])
    self.assertEqual(second.output_schemas, [output_schema])

  def test_apply_schema_none_keeps_inner_schemas(self):
    first = _ScriptedModel("scripted-a")
    second = _ScriptedModel("scripted-b")
    first_schema = schema_lib.FormatModeSchema()
    second_schema = schema_lib.FormatModeSchema()
    first.apply_schema(first_schema)
    second.apply_schema(second_schema)
    wrapper = self._wrap(first, second)

    wrapper.apply_schema(None)

    self.assertIs(first.schema, first_schema)
    self.assertIs(second.schema, second_schema)

  def test_forwards_exactly_the_classified_base_members(self):
    expected = {
        **dict.fromkeys(_FORWARDED_BASE_MEMBERS, "forwarded"),
        **dict.fromkeys(_NOT_FORWARDED_BASE_MEMBERS, "not forwarded"),
    }

    self.assertEqual(
        _base_member_handling(fallback.FallbackLanguageModel),
        expected,
    )


if __name__ == "__main__":
  absltest.main()
