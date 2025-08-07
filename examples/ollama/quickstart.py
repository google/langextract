#!/usr/bin/env python3
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

"""Quick-start example for using Ollama with langextract."""

import argparse
import os

import langextract as lx


def run_extraction(model_id="gemma2:2b", temperature=0.3):
  """Run a simple extraction example using Ollama."""
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
                  # extraction_text includes full context with ellipsis for clarity
                  extraction_text="J.R.R. Tolkien was an English writer...",
                  attributes={
                      "name": "J.R.R. Tolkien",
                      "genre": "high-fantasy",
                  },
              )
          ],
      )
  ]

  result = lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt,
      examples=examples,
      language_model_type=lx.inference.OllamaLanguageModel,
      model_id=model_id,
      model_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
      temperature=temperature,
      fence_output=False,
      use_schema_constraints=False,
  )

  return result


def main():
  """Main function to run the quick-start example."""
  parser = argparse.ArgumentParser(description="Run Ollama extraction example")
  parser.add_argument(
      "--model-id",
      default=os.getenv("MODEL_ID", "gemma2:2b"),
      help="Ollama model ID (default: gemma2:2b or MODEL_ID env var)",
  )
  parser.add_argument(
      "--temperature",
      type=float,
      default=float(os.getenv("TEMPERATURE", "0.3")),
      help="Model temperature (default: 0.3 or TEMPERATURE env var)",
  )
  args = parser.parse_args()

  print(f"🚀 Running Ollama quick-start example with {args.model_id}...")
  print("-" * 50)

  try:
    result = run_extraction(
        model_id=args.model_id, temperature=args.temperature
    )

    for extraction in result.extractions:
      print(f"Class: {extraction.extraction_class}")
      print(f"Text: {extraction.extraction_text}")
      print(f"Attributes: {extraction.attributes}")

    print("\n✅ SUCCESS! Ollama is working with langextract")
    return True

  except ConnectionError as e:
    print(f"\nConnectionError: {e}")
    print("Make sure Ollama is running: 'ollama serve'")
    return False
  except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    return False


if __name__ == "__main__":
  success = main()
  exit(0 if success else 1)
