import pytest
import responses

from langextract.core import exceptions
from langextract.providers.groq import GroqLanguageModel


def test_groq_model_name_resolution():
  m = GroqLanguageModel(model_id="groq/llama-3.1-8b-instant", api_key="x")
  assert m._resolve_groq_model_name() == "llama-3.1-8b-instant"

  m2 = GroqLanguageModel(model_id="llama-3.1-8b-instant", api_key="x")
  assert m2._resolve_groq_model_name() == "llama-3.1-8b-instant"


def test_groq_missing_api_key_raises_config_error(monkeypatch):
  monkeypatch.delenv("GROQ_API_KEY", raising=False)
  m = GroqLanguageModel(api_key=None)
  with pytest.raises(exceptions.InferenceConfigError):
    m._resolve_api_key()


@responses.activate
def test_groq_infer_parses_response(monkeypatch):
  monkeypatch.setenv("GROQ_API_KEY", "test-key")

  m = GroqLanguageModel(
      model_id="groq/llama-3.1-8b-instant",
      api_key=None,  # force env var path
      base_url="https://api.groq.com/openai/v1",
      temperature=0.0,
  )

  responses.add(
      responses.POST,
      "https://api.groq.com/openai/v1/chat/completions",
      json={"choices": [{"message": {"content": "John, Sarah"}}]},
      status=200,
  )

  out = next(m.infer(["Extract names"]))
  assert out[0].output == "John, Sarah"


@responses.activate
def test_groq_http_error_raises_runtime_error(monkeypatch):
  monkeypatch.setenv("GROQ_API_KEY", "test-key")

  m = GroqLanguageModel(base_url="https://api.groq.com/openai/v1")

  responses.add(
      responses.POST,
      "https://api.groq.com/openai/v1/chat/completions",
      json={"error": {"message": "unauthorized"}},
      status=401,
  )

  with pytest.raises(exceptions.InferenceRuntimeError):
    next(m.infer(["hi"]))
