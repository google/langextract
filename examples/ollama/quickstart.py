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
import sys

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
                  extraction_text=(
                      "J.R.R. Tolkien was an English writer, best known for"
                      " high-fantasy."
                  ),
                  attributes={
                      "name": "J.R.R. Tolkien",
                      "genre": "high-fantasy",
                  },
              )
          ],
      )
  ]

  model_config = lx.factory.ModelConfig(
      model_id=model_id,
      provider_kwargs={
          "model_url": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
          "format_type": lx.data.FormatType.JSON,
          "temperature": temperature,
      },
  )

  result = lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt,
      examples=examples,
      config=model_config,
      use_schema_constraints=True,
  )

  # Option 2: Pass model_id directly (simpler)
  # result = lx.extract(
  #     text_or_documents=input_text,
  #     prompt_description=prompt,
  #     examples=examples,
  #     model_id=model_id,
  #     model_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
  #     format_type=lx.data.FormatType.JSON,
  #     temperature=temperature,
  #     use_schema_constraints=True,
  # )

  return result


def main():
  """Main function to run the quick-start example."""
  parser = argparse.ArgumentParser(
      description="Run Ollama extraction example",
      epilog=(
          "Supported models: gemma2:2b, llama3.2:1b, mistral:7b, qwen2.5:0.5b,"
          " etc."
      ),
  )
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

    if result.extractions:
      print(f"\n📝 Found {len(result.extractions)} extraction(s):\n")
      for extraction in result.extractions:
        print(f"Class: {extraction.extraction_class}")
        print(f"Text: {extraction.extraction_text}")
        print(f"Attributes: {extraction.attributes}")
        print()
    else:
      print("\n⚠️  No extractions found")

    print("✅ SUCCESS! Ollama is working with langextract")
    print(f"   Model: {args.model_id}")
    print("   JSON mode: enabled")
    print("   Schema constraints: enabled")
    return True

  except ConnectionError as e:
    print(f"\n❌ ConnectionError: {e}")
    print("\n💡 Make sure Ollama is running:")
    print("   ollama serve")
    return False
  except ValueError as e:
    if "Can't find Ollama" in str(e):
      print(f"\n❌ Model not found: {args.model_id}")
      print("\n💡 Install the model first:")
      print(f"   ollama pull {args.model_id}")
    else:
      print(f"\n❌ ValueError: {e}")
    return False
  except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    return False


if __name__ == "__main__":
  SUCCESS = main()
  sys.exit(0 if SUCCESS else 1)
