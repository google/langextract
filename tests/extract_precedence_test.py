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

"""Tests for parameter precedence in extract()."""

from unittest import mock

from absl.testing import absltest

from langextract import factory
import langextract as lx
from langextract.core import data
from langextract.providers import openai


class ExtractParameterPrecedenceTest(absltest.TestCase):
  """Tests ensuring correct precedence among extract() parameters."""

  def setUp(self):
    super().setUp()
    self.examples = [
        data.ExampleData(
            text="example",
            extractions=[
                data.Extraction(
                    extraction_class="entity",
                    extraction_text="example",
                )
            ],
        )
    ]
    self.description = "description"

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_model_overrides_all_other_parameters(
      self, mock_create_model, mock_annotator_cls
  ):
    """Test that model parameter overrides all other model-related parameters."""
    provided_model = mock.MagicMock()
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_text.return_value = "ok"

    config = factory.ModelConfig(model_id="config-id")

    result = lx.extract(
        text_or_documents="text",
        prompt_description=self.description,
        examples=self.examples,
        model=provided_model,
        config=config,
        model_id="ignored-model",
        api_key="ignored-key",
        language_model_type=openai.OpenAILanguageModel,
        use_schema_constraints=False,
    )

    mock_create_model.assert_not_called()
    _, kwargs = mock_annotator_cls.call_args
    self.assertIs(kwargs["language_model"], provided_model)
    self.assertEqual(result, "ok")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_config_overrides_model_id_and_language_model_type(
      self, mock_create_model, mock_annotator_cls
  ):
    """Test that config parameter overrides model_id and language_model_type."""
    config = factory.ModelConfig(
        model_id="config-model", provider_kwargs={"api_key": "config-key"}
    )
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_text.return_value = "ok"

    with mock.patch(
        "langextract.extraction.factory.ModelConfig"
    ) as mock_model_config:
      result = lx.extract(
          text_or_documents="text",
          prompt_description=self.description,
          examples=self.examples,
          config=config,
          model_id="other-model",
          api_key="other-key",
          language_model_type=openai.OpenAILanguageModel,
          use_schema_constraints=False,
      )
      mock_model_config.assert_not_called()

    mock_create_model.assert_called_once()
    called_config = mock_create_model.call_args[1]["config"]
    self.assertEqual(called_config.model_id, "config-model")
    self.assertEqual(called_config.provider_kwargs, {"api_key": "config-key"})

    _, kwargs = mock_annotator_cls.call_args
    self.assertIs(kwargs["language_model"], mock_model)
    self.assertEqual(result, "ok")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_model_id_and_base_kwargs_override_language_model_type(
      self, mock_create_model, mock_annotator_cls
  ):
    """Test that model_id and other kwargs are used when no model or config."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model
    mock_annotator_cls.return_value.annotate_text.return_value = "ok"
    mock_config = mock.MagicMock()

    with mock.patch(
        "langextract.extraction.factory.ModelConfig", return_value=mock_config
    ) as mock_model_config:
      with self.assertWarns(FutureWarning):
        result = lx.extract(
            text_or_documents="text",
            prompt_description=self.description,
            examples=self.examples,
            model_id="model-123",
            api_key="api-key",
            temperature=0.9,
            model_url="http://model",
            language_model_type=openai.OpenAILanguageModel,
            use_schema_constraints=False,
        )

    mock_model_config.assert_called_once()
    _, kwargs = mock_model_config.call_args
    self.assertEqual(kwargs["model_id"], "model-123")
    provider_kwargs = kwargs["provider_kwargs"]
    self.assertEqual(provider_kwargs["api_key"], "api-key")
    self.assertEqual(provider_kwargs["temperature"], 0.9)
    self.assertEqual(provider_kwargs["model_url"], "http://model")
    self.assertEqual(provider_kwargs["base_url"], "http://model")
    mock_create_model.assert_called_once()
    self.assertEqual(result, "ok")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_language_model_type_only_emits_warning_and_works(
      self, mock_create_model, mock_annotator_cls
  ):
    """Test that language_model_type emits deprecation warning but still works."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model
    mock_annotator_cls.return_value.annotate_text.return_value = "ok"
    mock_config = mock.MagicMock()

    with mock.patch(
        "langextract.extraction.factory.ModelConfig", return_value=mock_config
    ) as mock_model_config:
      with self.assertWarns(FutureWarning):
        result = lx.extract(
            text_or_documents="text",
            prompt_description=self.description,
            examples=self.examples,
            language_model_type=openai.OpenAILanguageModel,
            use_schema_constraints=False,
        )

    mock_model_config.assert_called_once()
    _, kwargs = mock_model_config.call_args
    self.assertEqual(kwargs["model_id"], "gemini-2.5-flash")
    mock_create_model.assert_called_once()
    self.assertEqual(result, "ok")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_language_model_params_forward_retry_knobs(
      self, mock_create_model, mock_annotator_cls
  ):
    """Test that provider-specific retry knobs flow through language_model_params."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model
    mock_annotator_cls.return_value.annotate_text.return_value = "ok"
    mock_config = mock.MagicMock()

    with mock.patch(
        "langextract.extraction.factory.ModelConfig", return_value=mock_config
    ) as mock_model_config:
      result = lx.extract(
          text_or_documents="text",
          prompt_description=self.description,
          examples=self.examples,
          model_id="gemini-2.5-flash",
          api_key="api-key",
          language_model_params={
              "max_retries": 5,
              "retry_delay": 0.25,
              "max_retry_delay": 4.0,
          },
          use_schema_constraints=False,
      )

    mock_model_config.assert_called_once()
    _, kwargs = mock_model_config.call_args
    provider_kwargs = kwargs["provider_kwargs"]
    self.assertEqual(provider_kwargs["api_key"], "api-key")
    self.assertEqual(provider_kwargs["max_retries"], 5)
    self.assertEqual(provider_kwargs["retry_delay"], 0.25)
    self.assertEqual(provider_kwargs["max_retry_delay"], 4.0)
    mock_create_model.assert_called_once()
    self.assertEqual(result, "ok")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_use_schema_constraints_warns_with_config(
      self, mock_create_model, mock_annotator_cls
  ):
    """Test that use_schema_constraints emits warning when used with config."""
    config = factory.ModelConfig(
        model_id="gemini-2.5-flash", provider_kwargs={"api_key": "test-key"}
    )

    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = True
    mock_create_model.return_value = mock_model
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_text.return_value = "ok"

    with self.assertWarns(UserWarning) as cm:
      result = lx.extract(
          text_or_documents="text",
          prompt_description=self.description,
          examples=self.examples,
          config=config,
          use_schema_constraints=True,
      )

    self.assertIn("schema constraints", str(cm.warning))
    self.assertIn("applied", str(cm.warning))
    mock_create_model.assert_called_once()
    called_config = mock_create_model.call_args[1]["config"]
    self.assertEqual(called_config.model_id, "gemini-2.5-flash")
    self.assertEqual(result, "ok")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_use_schema_constraints_warns_with_model(
      self, mock_create_model, mock_annotator_cls
  ):
    """Test that use_schema_constraints emits warning when used with model."""
    provided_model = mock.MagicMock()
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_text.return_value = "ok"

    with self.assertWarns(UserWarning) as cm:
      result = lx.extract(
          text_or_documents="text",
          prompt_description=self.description,
          examples=self.examples,
          model=provided_model,
          use_schema_constraints=True,
      )

    self.assertIn("use_schema_constraints", str(cm.warning))
    self.assertIn("ignored", str(cm.warning))
    mock_create_model.assert_not_called()
    self.assertEqual(result, "ok")


class ExtractAdditionalContextTest(absltest.TestCase):
  """Tests for additional_context propagation in the document path of extract()."""

  def setUp(self):
    super().setUp()
    self.examples = [
        data.ExampleData(
            text="example",
            extractions=[
                data.Extraction(
                    extraction_class="entity",
                    extraction_text="example",
                )
            ],
        )
    ]
    self.description = "description"

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_additional_context_applied_to_documents_without_own_context(
      self, mock_create_model, mock_annotator_cls
  ):
    """Global additional_context is applied to Documents that have none set."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_documents.return_value = iter([])

    docs = [
        data.Document(text="first document"),
        data.Document(text="second document"),
    ]
    global_context = "Important disambiguation rule: treat X as a brand name."

    lx.extract(
        text_or_documents=docs,
        prompt_description=self.description,
        examples=self.examples,
        model=mock_model,
        additional_context=global_context,
        use_schema_constraints=False,
    )

    _, kwargs = mock_annotator.annotate_documents.call_args
    passed_docs = list(kwargs["documents"])
    self.assertLen(passed_docs, 2)
    for doc in passed_docs:
      self.assertEqual(doc.additional_context, global_context)

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_document_own_context_takes_precedence_over_global(
      self, mock_create_model, mock_annotator_cls
  ):
    """Per-document additional_context is not overwritten by the global value."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_documents.return_value = iter([])

    per_doc_context = "Document-specific context."
    global_context = "Global context."

    docs = [
        data.Document(
            text="doc with own context", additional_context=per_doc_context
        ),
        data.Document(text="doc without context"),
    ]

    lx.extract(
        text_or_documents=docs,
        prompt_description=self.description,
        examples=self.examples,
        model=mock_model,
        additional_context=global_context,
        use_schema_constraints=False,
    )

    _, kwargs = mock_annotator.annotate_documents.call_args
    passed_docs = list(kwargs["documents"])
    self.assertLen(passed_docs, 2)
    self.assertEqual(passed_docs[0].additional_context, per_doc_context)
    self.assertEqual(passed_docs[1].additional_context, global_context)

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_no_additional_context_leaves_documents_unchanged(
      self, mock_create_model, mock_annotator_cls
  ):
    """When additional_context is None, documents are passed through as-is."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_documents.return_value = iter([])

    docs = [
        data.Document(text="doc one"),
        data.Document(text="doc two"),
    ]
    original_docs = list(docs)

    lx.extract(
        text_or_documents=docs,
        prompt_description=self.description,
        examples=self.examples,
        model=mock_model,
        additional_context=None,
        use_schema_constraints=False,
    )

    _, kwargs = mock_annotator.annotate_documents.call_args
    passed_docs = list(kwargs["documents"])
    self.assertLen(passed_docs, 2)
    for passed, original in zip(passed_docs, original_docs):
      self.assertIs(passed, original)
      self.assertIsNone(passed.additional_context)

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_document_ids_preserved_when_applying_global_context(
      self, mock_create_model, mock_annotator_cls
  ):
    """Document IDs are not lost when global additional_context is applied."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_documents.return_value = iter([])

    docs = [
        data.Document(text="doc one", document_id="custom-id-1"),
        data.Document(text="doc two", document_id="custom-id-2"),
    ]

    lx.extract(
        text_or_documents=docs,
        prompt_description=self.description,
        examples=self.examples,
        model=mock_model,
        additional_context="context",
        use_schema_constraints=False,
    )

    _, kwargs = mock_annotator.annotate_documents.call_args
    passed_docs = list(kwargs["documents"])
    self.assertLen(passed_docs, 2)
    self.assertEqual(passed_docs[0].document_id, "custom-id-1")
    self.assertEqual(passed_docs[1].document_id, "custom-id-2")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_generator_input_works_with_additional_context(
      self, mock_create_model, mock_annotator_cls
  ):
    """Generator inputs are fully consumed when additional_context is applied."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_documents.return_value = iter([])

    def doc_generator():
      yield data.Document(text="gen doc one")
      yield data.Document(text="gen doc two")

    lx.extract(
        text_or_documents=doc_generator(),
        prompt_description=self.description,
        examples=self.examples,
        model=mock_model,
        additional_context="global context",
        use_schema_constraints=False,
    )

    _, kwargs = mock_annotator.annotate_documents.call_args
    passed_docs = list(kwargs["documents"])
    self.assertLen(passed_docs, 2)
    for doc in passed_docs:
      self.assertEqual(doc.additional_context, "global context")

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_caller_documents_not_mutated_by_global_context(
      self, mock_create_model, mock_annotator_cls
  ):
    """Original Document objects are not mutated when global context is applied."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_documents.return_value = iter([])

    docs = [
        data.Document(text="doc one"),
        data.Document(text="doc two"),
    ]

    lx.extract(
        text_or_documents=docs,
        prompt_description=self.description,
        examples=self.examples,
        model=mock_model,
        additional_context="injected context",
        use_schema_constraints=False,
    )

    for original in docs:
      self.assertIsNone(original.additional_context)

  @mock.patch("langextract.annotation.Annotator")
  @mock.patch("langextract.extraction.factory.create_model")
  def test_empty_string_additional_context_applied_to_docs(
      self, mock_create_model, mock_annotator_cls
  ):
    """Empty string additional_context is treated like any non-None value."""
    mock_model = mock.MagicMock()
    mock_model.requires_fence_output = False
    mock_create_model.return_value = mock_model
    mock_annotator = mock_annotator_cls.return_value
    mock_annotator.annotate_documents.return_value = iter([])

    docs = [data.Document(text="doc one")]

    lx.extract(
        text_or_documents=docs,
        prompt_description=self.description,
        examples=self.examples,
        model=mock_model,
        additional_context="",
        use_schema_constraints=False,
    )

    _, kwargs = mock_annotator.annotate_documents.call_args
    passed_docs = list(kwargs["documents"])
    self.assertLen(passed_docs, 1)
    self.assertEqual(passed_docs[0].additional_context, "")


if __name__ == "__main__":
  absltest.main()
