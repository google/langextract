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

"""Comprehensive demo of Ollama integration with FormatHandler.

This example demonstrates:
- Using the pre-configured OLLAMA_FORMAT_HANDLER for consistent configuration
- Running multiple extraction examples with progress bars
- Generating interactive HTML visualizations
- Handling various extraction patterns (NER, relationships, dialogue extraction)

Prerequisites:
1. Install Ollama: https://ollama.com/
2. Pull the model: ollama pull gemma2:2b
3. Start Ollama: ollama serve

Usage:
    python demo_ollama.py [--model MODEL_NAME]

Examples:
    # Use default model (gemma2:2b)
    python demo_ollama.py

    # Use a different model
    python demo_ollama.py --model llama3.2:3b

Output:
    Results are saved to test_output/ directory (gitignored)
    - JSONL files with extraction data
    - Interactive HTML visualizations
"""

import argparse
import os
from pathlib import Path
import sys
import textwrap
import time
import traceback
import urllib.error
import urllib.request

import dotenv

import langextract as lx
from langextract.providers import ollama

dotenv.load_dotenv(override=True)

DEFAULT_MODEL = "gemma2:2b"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OUTPUT_DIR = "test_output"


def check_ollama_available(url: str = DEFAULT_OLLAMA_URL) -> bool:
  """Check if Ollama is available at the specified URL."""
  try:
    with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as response:
      return response.status == 200
  except (urllib.error.URLError, TimeoutError):
    return False


def ensure_output_directory() -> Path:
  """Create output directory if it doesn't exist."""
  output_path = Path(OUTPUT_DIR)
  output_path.mkdir(exist_ok=True)
  return output_path


def print_header(title: str, width: int = 80) -> None:
  """Print a formatted header."""
  print("\n" + "=" * width)
  print(f"  {title}")
  print("=" * width)


def print_section(title: str, width: int = 60) -> None:
  """Print a formatted section."""
  print(f"\n▶ {title}")
  print("-" * width)


def print_results_summary(extractions: list[lx.data.Extraction]) -> None:
  """Print a summary of extraction results."""
  if not extractions:
    print("  No extractions found")
    return

  class_counts = {}
  for ext in extractions:
    class_counts[ext.extraction_class] = (
        class_counts.get(ext.extraction_class, 0) + 1
    )

  print(f"  Total extractions: {len(extractions)}")
  print("  By type:")
  for cls, count in sorted(class_counts.items()):
    print(f"    • {cls}: {count}")


def example_romeo_juliet(
    model_id: str, model_url: str
) -> lx.data.AnnotatedDocument | None:
  """Romeo & Juliet character and emotion extraction example."""
  print_section("Example 1: Romeo & Juliet - Characters and Emotions")

  prompt = textwrap.dedent("""\
      Extract characters, emotions, and relationships in order of appearance.
      Use exact text for extractions. Do not paraphrase or overlap entities.
      Provide meaningful attributes for each entity to add context.""")

  examples = [
      lx.data.ExampleData(
          text=(
              "ROMEO. But soft! What light through yonder window breaks? It is"
              " the east, and Juliet is the sun."
          ),
          extractions=[
              lx.data.Extraction(
                  extraction_class="character",
                  extraction_text="ROMEO",
                  attributes={"emotional_state": "wonder"},
              ),
              lx.data.Extraction(
                  extraction_class="emotion",
                  extraction_text="But soft!",
                  attributes={"feeling": "gentle awe"},
              ),
              lx.data.Extraction(
                  extraction_class="relationship",
                  extraction_text="Juliet is the sun",
                  attributes={"type": "metaphor"},
              ),
          ],
      )
  ]

  input_text = (
      "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"
  )

  print(f"  Input: {input_text}")
  print(f"  Model: {model_id}")
  print("\n  Extracting...")

  result = lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt,
      examples=examples,
      model_id=model_id,
      model_url=model_url,
      resolver_params={"format_handler": ollama.OLLAMA_FORMAT_HANDLER},
      show_progress=True,
  )

  print("\n  Results:")
  print_results_summary(result.extractions)

  return result


def example_medication_ner(
    model_id: str, model_url: str
) -> lx.data.AnnotatedDocument | None:
  """Medical named entity recognition example."""
  print_section("Example 2: Medication Named Entity Recognition")

  input_text = "Patient took 400 mg PO Ibuprofen q4h for two days."

  prompt_description = (
      "Extract medication information including medication name, dosage, route,"
      " frequency, and duration in the order they appear in the text."
  )

  examples = [
      lx.data.ExampleData(
          text="Patient was given 250 mg IV Cefazolin TID for one week.",
          extractions=[
              lx.data.Extraction(
                  extraction_class="dosage", extraction_text="250 mg"
              ),
              lx.data.Extraction(
                  extraction_class="route", extraction_text="IV"
              ),
              lx.data.Extraction(
                  extraction_class="medication", extraction_text="Cefazolin"
              ),
              lx.data.Extraction(
                  extraction_class="frequency", extraction_text="TID"
              ),
              lx.data.Extraction(
                  extraction_class="duration", extraction_text="for one week"
              ),
          ],
      )
  ]

  print(f"  Input: {input_text}")
  print(f"  Model: {model_id}")
  print("\n  Extracting...")

  result = lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt_description,
      examples=examples,
      model_id=model_id,
      model_url=model_url,
      resolver_params={"format_handler": ollama.OLLAMA_FORMAT_HANDLER},
      show_progress=True,
  )

  print("\n  Results:")
  print_results_summary(result.extractions)

  return result


def example_medication_relationships(
    model_id: str, model_url: str
) -> lx.data.AnnotatedDocument | None:
  """Medication relationship extraction with grouped attributes."""
  print_section("Example 3: Medication Relationship Extraction")

  input_text = textwrap.dedent("""
      The patient was prescribed Lisinopril and Metformin last month.
      He takes the Lisinopril 10mg daily for hypertension, but often misses
      his Metformin 500mg dose which should be taken twice daily for diabetes.
  """).strip()

  prompt_description = textwrap.dedent("""
      Extract medications with their details, using attributes to group related information:

      1. Extract entities in the order they appear in the text
      2. Each entity must have a 'medication_group' attribute linking it to its medication
      3. All details about a medication should share the same medication_group value
  """).strip()

  examples = [
      lx.data.ExampleData(
          text=(
              "Patient takes Aspirin 100mg daily for heart health and"
              " Simvastatin 20mg at bedtime."
          ),
          extractions=[
              lx.data.Extraction(
                  extraction_class="medication",
                  extraction_text="Aspirin",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class="dosage",
                  extraction_text="100mg",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class="frequency",
                  extraction_text="daily",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class="condition",
                  extraction_text="heart health",
                  attributes={"medication_group": "Aspirin"},
              ),
              lx.data.Extraction(
                  extraction_class="medication",
                  extraction_text="Simvastatin",
                  attributes={"medication_group": "Simvastatin"},
              ),
              lx.data.Extraction(
                  extraction_class="dosage",
                  extraction_text="20mg",
                  attributes={"medication_group": "Simvastatin"},
              ),
              lx.data.Extraction(
                  extraction_class="frequency",
                  extraction_text="at bedtime",
                  attributes={"medication_group": "Simvastatin"},
              ),
          ],
      )
  ]

  print(f"  Input: {input_text[:80]}...")
  print(f"  Model: {model_id}")
  print("\n  Extracting...")

  result = lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt_description,
      examples=examples,
      model_id=model_id,
      model_url=model_url,
      resolver_params={"format_handler": ollama.OLLAMA_FORMAT_HANDLER},
      show_progress=True,
  )

  print("\n  Results:")
  print_results_summary(result.extractions)

  medication_groups = {}
  for ext in result.extractions:
    if ext.attributes and "medication_group" in ext.attributes:
      group_name = ext.attributes["medication_group"]
      medication_groups.setdefault(group_name, []).append(ext)

  if medication_groups:
    print("\n  Grouped by medication:")
    for med_name in sorted(medication_groups.keys()):
      print(f"    {med_name}: {len(medication_groups[med_name])} attributes")

  return result


def example_shakespeare_dialogue(
    model_id: str, model_url: str
) -> lx.data.AnnotatedDocument | None:
  """Extract character dialogue from Shakespeare play excerpt."""
  print_section("Example 4: Shakespeare Dialogue Extraction")

  long_text = textwrap.dedent("""
      Act I, Scene I. Verona. A public place.

      Enter SAMPSON and GREGORY, armed with swords and bucklers.

      SAMPSON: Gregory, on my word, we'll not carry coals.
      GREGORY: No, for then we should be colliers.
      SAMPSON: I mean, an we be in choler, we'll draw.
      GREGORY: Ay, while you live, draw your neck out of collar.

      Enter ABRAHAM and BALTHASAR.

      ABRAHAM: Do you bite your thumb at us, sir?
      SAMPSON: I do bite my thumb, sir.
      ABRAHAM: Do you bite your thumb at us, sir?
      SAMPSON: No, sir, I do not bite my thumb at you, sir, but I bite my thumb, sir.
      GREGORY: Do you quarrel, sir?
      ABRAHAM: Quarrel, sir? No, sir.

      Enter BENVOLIO.

      BENVOLIO: Part, fools! Put up your swords. You know not what you do.

      Enter TYBALT.

      TYBALT: What, art thou drawn among these heartless hinds?
      Turn thee, Benvolio; look upon thy death.
      BENVOLIO: I do but keep the peace. Put up thy sword,
      Or manage it to part these men with me.
      TYBALT: What, drawn, and talk of peace? I hate the word,
      As I hate hell, all Montagues, and thee.
      Have at thee, coward!
  """).strip()

  prompt = (
      "Extract all character names and their dialogue in order of appearance."
  )

  examples = [
      lx.data.ExampleData(
          text="JULIET: O Romeo, Romeo! Wherefore art thou Romeo?",
          extractions=[
              lx.data.Extraction(
                  extraction_class="character", extraction_text="JULIET"
              ),
              lx.data.Extraction(
                  extraction_class="dialogue",
                  extraction_text="O Romeo, Romeo! Wherefore art thou Romeo?",
                  attributes={"speaker": "JULIET"},
              ),
          ],
      )
  ]

  print(f"  Input: Romeo and Juliet Act I, Scene I ({len(long_text)} chars)")
  print(f"  Model: {model_id}")
  print("  Note: Automatically chunked for longer text processing")
  print("\n  Extracting...")

  result = lx.extract(
      text_or_documents=long_text,
      prompt_description=prompt,
      examples=examples,
      model_id=model_id,
      model_url=model_url,
      resolver_params={"format_handler": ollama.OLLAMA_FORMAT_HANDLER},
      max_char_buffer=500,
      show_progress=True,
  )

  print("\n  Results:")
  print_results_summary(result.extractions)

  characters = set(
      ext.extraction_text
      for ext in result.extractions
      if ext.extraction_class == "character"
  )
  if characters:
    print("\n  Characters found: " + ", ".join(sorted(characters)))

  return result


def save_results(
    results: list[tuple[str, lx.data.AnnotatedDocument | None]],
    output_dir: Path,
) -> None:
  """Save all results to JSONL and generate HTML visualizations."""
  print_header("Saving Results and Generating Visualizations")

  saved_files = []

  for name, result in results:
    if result is None:
      print(f"  ✗ Skipping {name} (no result)")
      continue

    jsonl_file = f"{name}.jsonl"
    jsonl_path = output_dir / jsonl_file

    lx.io.save_annotated_documents(
        [result], output_name=jsonl_file, output_dir=str(output_dir)
    )
    print(f"  ✓ Saved {jsonl_path}")

    html_file = f"{name}.html"
    html_path = output_dir / html_file

    try:
      html_content = lx.visualize(str(jsonl_path))
      with open(html_path, "w") as f:
        if hasattr(html_content, "data"):
          f.write(html_content.data)
        else:
          f.write(html_content)
      print(f"  ✓ Generated {html_path}")
      saved_files.append((jsonl_path, html_path))
    except Exception as e:
      print(f"  ✗ Failed to generate {html_path}: {e}")

  return saved_files


def main():
  """Run all examples and generate outputs."""
  parser = argparse.ArgumentParser(
      description="Ollama + FormatHandler Demo",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument(
      "--model",
      default=DEFAULT_MODEL,
      help=f"Ollama model to use (default: {DEFAULT_MODEL})",
  )
  parser.add_argument(
      "--url",
      default=DEFAULT_OLLAMA_URL,
      help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
  )
  parser.add_argument(
      "--skip-examples",
      nargs="+",
      choices=["1", "2", "3", "4"],
      help="Skip specific examples (e.g., --skip-examples 3 4)",
  )

  args = parser.parse_args()
  skip_examples = set(args.skip_examples or [])

  print_header("Ollama + FormatHandler Demo")
  print("\nConfiguration:")
  print(f"  Model: {args.model}")
  print(f"  Server: {args.url}")
  print(f"  Output: {OUTPUT_DIR}/")
  print(f"  Format Handler: {ollama.OLLAMA_FORMAT_HANDLER}")

  print("\nChecking Ollama server...")
  if not check_ollama_available(args.url):
    print(f"\n⚠️  ERROR: Ollama not available at {args.url}")
    print("\nTroubleshooting:")
    print("  1. Install Ollama: https://ollama.com/")
    print("  2. Start server: ollama serve")
    print(f"  3. Pull model: ollama pull {args.model}")
    print("\nFor Docker setup, see examples/ollama/docker-compose.yml")
    sys.exit(1)

  print("✓ Ollama server is available")

  output_dir = ensure_output_directory()
  print("✓ Output directory ready: " + str(output_dir) + "/")

  print_header("Running Examples")
  results = []

  try:
    if "1" not in skip_examples:
      result = example_romeo_juliet(args.model, args.url)
      results.append(("romeo_juliet", result))
      time.sleep(0.5)

    if "2" not in skip_examples:
      result = example_medication_ner(args.model, args.url)
      results.append(("medication_ner", result))
      time.sleep(0.5)

    if "3" not in skip_examples:
      result = example_medication_relationships(args.model, args.url)
      results.append(("medication_relationships", result))
      time.sleep(0.5)

    if "4" not in skip_examples:
      result = example_shakespeare_dialogue(args.model, args.url)
      results.append(("shakespeare_dialogue", result))

  except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    print("Saving completed results...")
  except Exception as e:
    print(f"\n\n✗ Error during execution: {e}")
    traceback.print_exc()
    print("\nSaving completed results...")

  if results:
    save_results(results, output_dir)

  print_header("Summary")

  successful = sum(1 for _, r in results if r is not None)
  print(f"\n✓ Successfully ran {successful}/{len(results)} examples")

  if results:
    print(f"\nOutput files in {output_dir}/:")
    for name, result in results:
      if result is not None:
        print(f"  • {name}.jsonl - Extraction data")
        print(f"  • {name}.html  - Interactive visualization")

    print("\nTo view results:")
    print("  open " + str(output_dir) + "/romeo_juliet.html")
    print("\nOr serve locally:")
    print("  python -m http.server 8000 --directory " + str(output_dir))
    print("  Then visit http://localhost:8000")


if __name__ == "__main__":
  main()
