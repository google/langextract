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
  """Test extraction using Ollama when available."""
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

  try:
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        language_model_type=lx.inference.OllamaLanguageModel,
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

  except ValueError as e:
    if "Can't find Ollama" in str(e):
      pytest.skip(f"Ollama model {model_id} not available")
    raise
