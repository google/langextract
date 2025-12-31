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

"""Integration tests for Ollama functionality."""
import socket

import pytest
import requests

import langextract as lx


def _ollama_available():
  """Check if Ollama is running on localhost:11434."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    result = sock.connect_ex(("localhost", 11434))
    return result == 0


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_ollama_extraction():
  input_text = "Isaac Asimov was a prolific science fiction writer."
  prompt = "Extract the author's full name and their primary literary genre."

  examples = [
      lx.data.ExampleData(
          text=(
              "J.R.R. Tolkien was an English writer, best known for"
              " high-fantasy."
          ),
          extractions=[
              lx.data.Extraction(
                  extraction_class="author_details",
                  extraction_text="J.R.R. Tolkien was an English writer...",
                  attributes={
                      "name": "J.R.R. Tolkien",
                      "genre": "high-fantasy",
                  },
              )
          ],
      )
  ]

  model_id = "gemma2:2b"

  result = lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt,
      examples=examples,
      model_id=model_id,
      model_url="http://localhost:11434",
      temperature=0.3,
      fence_output=False,
      use_schema_constraints=False,
  )

  assert len(result.extractions) > 0
  extraction = result.extractions[0]
  assert extraction.extraction_class == "author_details"
  if extraction.attributes:
    assert "asimov" in extraction.attributes.get("name", "").lower()


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_ollama_extraction_with_fence_fallback():
  input_text = "Marie Curie was a physicist who won two Nobel prizes."
  prompt = "Extract information about people and their achievements."

  examples = [
      lx.data.ExampleData(
          text="Albert Einstein developed the theory of relativity.",
          extractions=[
              lx.data.Extraction(
                  extraction_class="person",
                  extraction_text="Albert Einstein",
                  attributes={"achievement": "theory of relativity"},
              )
          ],
      )
  ]

  model_id = "gemma2:2b"

  result = lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt,
      examples=examples,
      model_id=model_id,
      model_url="http://localhost:11434",
      temperature=0.3,
      fence_output=True,  # Testing that fallback works
      use_schema_constraints=False,
  )

  assert len(result.extractions) > 0
  extraction = result.extractions[0]
  assert extraction.extraction_class == "person"
  assert (
      "marie" in extraction.extraction_text.lower()
      or "curie" in extraction.extraction_text.lower()
  )


def _model_available(model_name):
  """Check if a specific model is available in Ollama."""
  if not _ollama_available():
    return False
  try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in response.json().get("models", [])]
    return any(model_name in m for m in models)
  except (requests.RequestException, KeyError, TypeError):
    return False


@pytest.mark.skipif(
    not _model_available("deepseek-r1"),
    reason="DeepSeek-R1 not available in Ollama",
)
def test_deepseek_r1_extraction():
  """Test extraction with DeepSeek-R1 reasoning model.

  DeepSeek-R1 outputs <think> tags before JSON when not using format:json.
  This test verifies the model works correctly with langextract.
  """
  input_text = "John Smith is a software engineer at Google."
  prompt = "Extract people and their roles."

  examples = [
      lx.data.ExampleData(
          text="Alice works as a designer at Apple.",
          extractions=[
              lx.data.Extraction(
                  extraction_class="person",
                  extraction_text="Alice",
                  attributes={"role": "designer", "company": "Apple"},
              )
          ],
      )
  ]

  result = lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt,
      examples=examples,
      model_id="deepseek-r1:1.5b",
      model_url="http://localhost:11434",
      temperature=0.3,
  )

  assert len(result.extractions) > 0
  extraction = result.extractions[0]
  assert extraction.extraction_class == "person"
  assert "john" in extraction.extraction_text.lower()
