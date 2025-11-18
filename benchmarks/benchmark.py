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

"""LangExtract benchmark suite for performance and quality testing.

Measures tokenization speed and extraction quality across multiple languages
and text types. Automatically downloads test texts from Project Gutenberg
and generates comparative visualizations.

Usage:
  # Run diverse text type benchmark (default)
  python benchmarks/benchmark.py

  # Test with specific model
  python benchmarks/benchmark.py --model gemini-2.5-flash
  python benchmarks/benchmark.py --model gemma2:2b  # Local model via Ollama

  # Generate comparison plots from existing results
  python benchmarks/benchmark.py --compare

Requirements:
  - Set GEMINI_API_KEY for cloud models
  - Install Ollama for local model testing
  - Results saved to benchmark_results/
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error

import dotenv

from benchmarks import config
from benchmarks import plotting
from benchmarks import utils
import langextract
from langextract import core
from langextract import data
from langextract import visualize
import langextract.io as lio

# Load API key from environment
dotenv.load_dotenv(override=True)
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", os.environ.get("LANGEXTRACT_API_KEY")
)


class BenchmarkRunner:
  """Orchestrates benchmark execution and result collection."""

  def __init__(self):
    """Initialize runner with timestamp and git metadata."""
    self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.git_info = utils.get_git_info()

  def print_header(self):
    """Print benchmark header."""
    print("=" * config.DISPLAY.separator_width)
    print("LANGEXTRACT BENCHMARK")
    print("=" * config.DISPLAY.separator_width)
    print(
        f"Branch: {self.git_info['branch']} | Commit: {self.git_info['commit']}"
    )
    print("-" * config.DISPLAY.separator_width)

  def benchmark_tokenization(self) -> list[dict[str, Any]]:
    """Measure tokenization throughput at different text sizes.

    Returns:
      List of dicts with words, tokens, timing, and throughput metrics.
    """
    print("\nTokenization Performance")
    print("-" * config.DISPLAY.subseparator_width)

    results = []

    for word_count in config.TOKENIZATION.default_text_sizes:
      text = " ".join(["word"] * word_count)

      _ = core.tokenizer.tokenize(text)

      times = []
      for _ in range(config.TOKENIZATION.benchmark_iterations):
        start = time.perf_counter()
        tokenized = core.tokenizer.tokenize(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

      avg_time = sum(times) / len(times)
      avg_ms = avg_time * 1000
      num_tokens = len(tokenized.tokens)
      tokens_per_sec = num_tokens / avg_time if avg_time > 0 else 0

      word_str = (
          f"{word_count//1000:,}k" if word_count >= 1000 else f"{word_count:,}"
      )

      print(
          f"{word_str:>6} words: {avg_ms:7.2f}ms  "
          f"({tokens_per_sec/1e6:.1f}M tokens/sec)"
      )

      results.append({
          "words": word_count,
          "tokens": num_tokens,
          "avg_ms": avg_ms,
          "tokens_per_sec": tokens_per_sec,
      })

    return results

  def test_single_extraction(
      self,
      model_id: str = config.MODELS.default_model,
      text_type: config.TextTypes = config.TextTypes.ENGLISH,
  ) -> dict[str, Any]:
    """Execute extraction test on a single text sample.

    Args:
      model_id: Model identifier (e.g., 'gemini-2.5-flash', 'gemma2:2b').
      text_type: Language/text type to test.

    Returns:
      Dict with success status, timing, entity counts, and metrics.
    """
    print("\nExtraction Test")
    print("-" * config.DISPLAY.subseparator_width)

    try:
      # Get test text
      test_text = utils.get_text_from_gutenberg(text_type)
      test_text = utils.get_optimal_text_size(test_text, model_id)

      print(f"   Text: {len(test_text):,} characters ({text_type.value})")
      print(f"   Model: {model_id}")

      # Analyze tokenization
      tokenization_analysis = utils.analyze_tokenization(test_text)
      print(
          "   Tokenization:"
          f" {utils.format_tokenization_summary(tokenization_analysis)}"
      )

      # Get extraction config for text type
      extraction_config = utils.get_extraction_example(text_type)

      example = data.ExampleData(
          text="MACBETH speaks to LADY MACBETH about Duncan.",
          extractions=[
              data.Extraction(
                  extraction_text="Macbeth", extraction_class="Character"
              ),
              data.Extraction(
                  extraction_text="Lady Macbeth", extraction_class="Character"
              ),
              data.Extraction(
                  extraction_text="Duncan", extraction_class="Character"
              ),
          ],
      )

      max_retries = 5
      retry_delay = 3.0

      # Retry logic for transient network/API failures
      for attempt in range(max_retries):
        try:
          start_time = time.time()
          result = langextract.extract(
              text_or_documents=test_text,
              model_id=model_id,
              api_key=GEMINI_API_KEY,
              prompt_description=extraction_config["prompt"],
              examples=[example],
              max_workers=config.MODELS.default_max_workers,
              temperature=config.MODELS.default_temperature,
              extraction_passes=config.MODELS.default_extraction_passes,
          )
          elapsed = time.time() - start_time
          break
        except (ConnectionError, TimeoutError):
          if attempt < max_retries - 1:
            print(f"   Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 1.5
            continue
          raise

      print(f"Extraction completed in {elapsed:.1f}s")

      grounded_entities = []
      ungrounded_entities = []

      if result.extractions:
        for extraction in result.extractions:
          is_grounded = (
              extraction.char_interval
              and extraction.char_interval.start_pos is not None
              and extraction.char_interval.end_pos is not None
          )

          entity_text = extraction.extraction_text
          if entity_text:
            if is_grounded:
              grounded_entities.append(entity_text)
            else:
              ungrounded_entities.append(entity_text)

      unique_grounded = list(set(grounded_entities))
      unique_ungrounded = list(set(ungrounded_entities))

      print(f"Found {len(unique_grounded)} grounded entities")
      if unique_ungrounded:
        print(f"   ({len(unique_ungrounded)} ungrounded entities ignored)")

      if unique_grounded:
        sample = unique_grounded[:5]
        sample_str = ", ".join(sample) + (
            "..." if len(unique_grounded) > 5 else ""
        )
        print(f"   Sample: {sample_str}")

      return {
          "success": True,
          "model": model_id,
          "text_type": text_type.value,
          "time_seconds": elapsed,
          "entity_count": len(unique_grounded),
          "ungrounded_count": len(unique_ungrounded),
          "sample_entities": unique_grounded[:10],
          "tokenization": tokenization_analysis,
          config.EXTRACTION_RESULT_KEY: result,
      }

    except (urllib.error.URLError, RuntimeError) as e:
      # Text download failures - expected and should be handled gracefully
      print(f"Failed: {e}")
      return {
          "success": False,
          "model": model_id,
          "text_type": text_type.value,
          "error": str(e),
      }

  def test_diverse_text_types(
      self, models: list[str] | None = None
  ) -> list[dict[str, Any]]:
    """Test extraction with diverse text types."""
    print("\n" + "=" * config.DISPLAY.separator_width)
    print("DIVERSE TEXT TYPE MODE")
    print("=" * config.DISPLAY.separator_width)

    if models is None:
      models = [config.MODELS.default_model]

    results = []
    test_count = 0

    for model_id in models:
      print(f"\nTesting {model_id}")
      print("-" * 30)

      for text_type in config.TextTypes:
        print(f"\n  Testing {text_type.value} text...")
        result = self.test_single_extraction(model_id, text_type)
        results.append(result)

        if result.get("success"):
          test_count += 1
          if test_count % 3 == 0:
            print(
                "   Rate limit delay"
                f" ({config.MODELS.gemini_rate_limit_delay}s)..."
            )
            time.sleep(config.MODELS.gemini_rate_limit_delay)

    print(f"\nCompleted {test_count} successful tests")
    return results

  def save_results(self, results: dict[str, Any]):
    """Save benchmark results to JSON and create plots."""
    results["timestamp"] = self.timestamp
    results["git"] = self.git_info

    json_path = config.PATHS.get_result_path(self.timestamp, "").with_suffix(
        ".json"
    )

    viz_dir = json_path.parent / "visualizations" / self.timestamp
    viz_dir.mkdir(parents=True, exist_ok=True)

    if config.RESULTS_KEY in results:
      print(f"\nGenerating visualizations in: {viz_dir}")
      for result in results[config.RESULTS_KEY]:
        if result.get("success") and config.EXTRACTION_RESULT_KEY in result:
          model_name = result["model"].replace("/", "_").replace(":", "_")
          text_type = result["text_type"]
          viz_name = f"{model_name}_{text_type}"

          jsonl_path = viz_dir / f"{viz_name}.jsonl"
          lio.save_annotated_documents(
              [result[config.EXTRACTION_RESULT_KEY]],
              output_name=jsonl_path.name,
              output_dir=str(viz_dir),
          )

          html_content = visualize(str(jsonl_path))
          html_path = viz_dir / f"{viz_name}.html"
          with open(html_path, "w") as f:
            f.write(getattr(html_content, "data", html_content))

    # Remove extraction result objects before saving JSON
    for result in results.get(config.RESULTS_KEY, []):
      result.pop(config.EXTRACTION_RESULT_KEY, None)

    with open(json_path, "w") as f:
      json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    plot_created = plotting.create_diverse_plots(results, json_path)

    if plot_created:
      print(f"Plot saved to: {json_path.with_suffix('.png')}")
    else:
      print(f"Warning: Failed to create plot for {json_path.name}")

  def run_diverse_benchmark(self, models: list[str] | None = None):
    """Run diverse text type benchmark."""
    self.print_header()

    tokenization_results = self.benchmark_tokenization()
    diverse_results = self.test_diverse_text_types(models)

    results = {
        "tokenization": tokenization_results,
        config.RESULTS_KEY: diverse_results,
    }

    self.save_results(results)


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(description="LangExtract Benchmark Suite")

  parser.add_argument(
      "--model",
      type=str,
      default=None,
      help=f"Model to use (default: {config.MODELS.default_model})",
  )

  parser.add_argument(
      "--compare",
      action="store_true",
      help="Generate comparison plots from existing benchmark results",
  )

  args = parser.parse_args()

  # Handle comparison mode
  if args.compare:
    results_dir = Path("benchmark_results")
    json_files = sorted(results_dir.glob("benchmark_*.json"))

    if len(json_files) < 2:
      print(
          "Need at least 2 benchmark results for comparison, found"
          f" {len(json_files)}"
      )
      return

    print(f"Found {len(json_files)} benchmark results to compare")

    # Use last 10 results or all if less than 10
    files_to_compare = json_files[-10:]
    comparison_path = (
        results_dir
        / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )

    plotting.create_comparison_plots(files_to_compare, comparison_path)
    print(f"\nComparison plot saved to: {comparison_path}")
    return

  model_to_test = args.model or config.MODELS.default_model
  if "gemini" in model_to_test.lower() and not GEMINI_API_KEY:
    print(
        f"Error: {model_to_test} requires GEMINI_API_KEY or LANGEXTRACT_API_KEY"
    )
    return

  runner = BenchmarkRunner()
  runner.run_diverse_benchmark([args.model] if args.model else None)


if __name__ == "__main__":
  main()
