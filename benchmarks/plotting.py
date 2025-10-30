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

"""Visualization generation for benchmark results.

Creates multi-panel plots showing tokenization performance, extraction metrics,
and cross-language comparisons.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import config

matplotlib.use("Agg")
plt.style.use(config.DISPLAY.plot_style)


def create_diverse_plots(results: dict[str, Any], filepath: Path) -> bool:
  """Generate comprehensive benchmark visualization.

  Args:
    results: Benchmark results dictionary with tokenization and extraction data.
    filepath: Output path for PNG file.

  Returns:
    True if plot created successfully, False on error.
  """
  try:
    fig = plt.figure(figsize=(15, 10))

    # Create 2x3 grid: tokenization metrics (top), extraction metrics (bottom)
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])  # Tokenization throughput
    ax2 = fig.add_subplot(gs[0, 1])  # Token density by language
    ax3 = fig.add_subplot(gs[0, 2])  # Entity extraction counts
    ax4 = fig.add_subplot(gs[1, 0])  # Processing speed
    ax5 = fig.add_subplot(gs[1, 1])  # Summary metrics
    ax6 = fig.add_subplot(gs[1, 2])  # Unused

    fig.suptitle(
        f"LangExtract Benchmark - {results['timestamp']}", fontsize=14, y=0.98
    )

    _plot_tokenization_throughput(ax1, results)
    _plot_tokenization_rate(ax2, results)
    _plot_extraction_density(ax3, results)
    _plot_processing_speed(ax4, results)
    _plot_summary_table(ax5, results)
    ax6.axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    plot_path = filepath.with_suffix(".png")
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {plot_path}")
    return True

  except (IOError, OSError) as e:
    print(f"Warning: Could not create benchmark plot: {e}")
    return False


def _plot_tokenization_throughput(ax, results):
  """Plot tokenization throughput (tokens per second) on log scale."""
  if (
      config.TOKENIZATION_KEY not in results
      or not results[config.TOKENIZATION_KEY]
  ):
    ax.text(0.5, 0.5, "No tokenization data", ha="center", va="center")
    ax.set_title("Tokenization Throughput")
    return

  sizes = [r["words"] for r in results[config.TOKENIZATION_KEY]]
  speeds = [r["tokens_per_sec"] for r in results[config.TOKENIZATION_KEY]]

  ax.semilogx(sizes, speeds, "b-o", linewidth=2, markersize=8)
  ax.set_xlabel("Number of Words (log scale)")
  ax.set_ylabel("Tokens per Second")
  ax.set_title("Tokenization Throughput")
  ax.grid(True, alpha=0.3)

  max_speed = max(speeds)
  ax.set_ylim(0, max_speed * 1.15)

  y_ticks = [0, 100000, 200000, 300000, 400000]
  ax.set_yticks(y_ticks)
  ax.set_yticklabels([f"{int(y/1000)}K" if y > 0 else "0" for y in y_ticks])

  for x, y in zip(sizes, speeds):
    label = f"{y/1000:.0f}K"
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

  ax.set_xticks([100, 1000, 10000])
  ax.set_xticklabels(["10²", "10³", "10⁴"])


def _plot_tokenization_rate(ax, results):
  """Plot tokenization rate by text type."""
  if config.RESULTS_KEY not in results:
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title("Tokenization Rate")
    return

  text_types = []
  tok_per_char = []

  for result in results[config.RESULTS_KEY]:
    if config.TOKENIZATION_KEY in result and result.get("success", False):
      text_type = result.get("text_type", "unknown")
      if text_type not in text_types:
        text_types.append(text_type)
        tpc = result[config.TOKENIZATION_KEY]["tokens_per_char"]
        tok_per_char.append(tpc)

  if not text_types:
    ax.text(0.5, 0.5, "No tokenization data", ha="center", va="center")
    ax.set_title("Tokenization Rate")
    return

  x = np.arange(len(text_types))
  bars = ax.bar(x, tok_per_char, color="#2196f3", alpha=0.7)

  for bar_rect, val in zip(bars, tok_per_char):
    ax.text(
        bar_rect.get_x() + bar_rect.get_width() / 2,
        val + 0.005,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

  ax.set_xlabel("Text Type")
  ax.set_ylabel("Tokens per Character")
  ax.set_title("Tokenization Rate")
  ax.set_xticks(x)
  ax.set_xticklabels([t.capitalize() for t in text_types])
  ax.grid(True, alpha=0.3, axis="y")
  ax.set_ylim(0, max(0.30, max(tok_per_char) * 1.2) if tok_per_char else 0.30)


def _plot_extraction_density(ax, results):
  """Plot entity extraction density."""
  if config.RESULTS_KEY not in results:
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title("Extraction Density")
    return

  text_types = []
  densities = []

  for result in results[config.RESULTS_KEY]:
    if result.get("success", False):
      text_type = result.get("text_type", "unknown")
      if text_type not in text_types:
        text_types.append(text_type)

        char_count = 1000
        if config.TOKENIZATION_KEY in result:
          char_count = result[config.TOKENIZATION_KEY].get("num_chars", 1000)

        entity_count = result.get("entity_count", 0)
        density = (entity_count * 1000) / char_count
        densities.append(density)

  if not text_types:
    ax.text(0.5, 0.5, "No successful extractions", ha="center", va="center")
    ax.set_title("Extraction Density")
    return

  x = np.arange(len(text_types))
  bars = ax.bar(x, densities, color="#4caf50", alpha=0.7)

  for bar_rect, val in zip(bars, densities):
    ax.text(
        bar_rect.get_x() + bar_rect.get_width() / 2,
        val,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

  ax.set_xlabel("Text Type")
  ax.set_ylabel("Entities per 1K Characters")
  ax.set_title("Extraction Density")
  ax.set_xticks(x)
  ax.set_xticklabels([t.capitalize() for t in text_types])
  ax.grid(True, alpha=0.3, axis="y")


def _plot_processing_speed(ax, results):
  """Plot processing speed normalized by text size."""
  if config.RESULTS_KEY not in results:
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title("Processing Speed")
    return

  text_types = []
  speeds = []

  for result in results[config.RESULTS_KEY]:
    if result.get("success", False):
      text_type = result.get("text_type", "unknown")
      if text_type not in text_types:
        text_types.append(text_type)

        char_count = 1000
        if config.TOKENIZATION_KEY in result:
          char_count = result[config.TOKENIZATION_KEY].get("num_chars", 1000)

        time_seconds = result.get("time_seconds", 0)
        speed = (time_seconds * 1000) / char_count
        speeds.append(speed)

  if not text_types:
    ax.text(0.5, 0.5, "No timing data", ha="center", va="center")
    ax.set_title("Processing Speed")
    return

  x = np.arange(len(text_types))
  bars = ax.bar(x, speeds, color="#ff9800", alpha=0.7)

  for bar_rect, val in zip(bars, speeds):
    ax.text(
        bar_rect.get_x() + bar_rect.get_width() / 2,
        val,
        f"{val:.1f}s",
        ha="center",
        va="bottom",
        fontsize=9,
    )

  ax.set_xlabel("Text Type")
  ax.set_ylabel("Seconds per 1K Characters")
  ax.set_title("Processing Speed")
  ax.set_xticks(x)
  ax.set_xticklabels([t.capitalize() for t in text_types])
  ax.grid(True, alpha=0.3, axis="y")


def _plot_summary_table(ax, results):
  """Create a summary of key findings."""
  ax.axis("off")

  if config.RESULTS_KEY not in results:
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title("Key Metrics")
    return

  summary_lines = []
  summary_lines.append("Key Metrics")
  summary_lines.append("-" * 20)
  summary_lines.append("")

  success_count = sum(
      1 for r in results.get(config.RESULTS_KEY, []) if r.get("success")
  )
  total_count = len(results.get(config.RESULTS_KEY, []))

  if total_count > 0:
    summary_lines.append("Tests Run:")
    summary_lines.append(f"  {success_count} successful")
    summary_lines.append(f"  {total_count - success_count} failed")
    summary_lines.append("")

  if success_count > 0:
    avg_time = (
        sum(
            r.get("time_seconds", 0)
            for r in results.get(config.RESULTS_KEY, [])
            if r.get("success")
        )
        / success_count
    )
    summary_lines.append(f"Avg Time: {avg_time:.1f}s")

  summary_text = "\n".join(summary_lines)
  ax.text(
      0.5,
      0.5,
      summary_text,
      ha="center",
      va="center",
      fontsize=10,
      family="monospace",
  )

  ax.set_title("Key Metrics", fontweight="bold", y=0.9)


def create_comparison_plots(json_files: list[Path], output_path: Path) -> None:
  """Create comparison plots from multiple benchmark JSON files.

  Args:
    json_files: List of paths to benchmark JSON files to compare.
    output_path: Path where the comparison plot should be saved.
  """
  if len(json_files) < 2:
    print("Need at least 2 JSON files for comparison")
    return

  all_results = []
  for json_file in json_files:
    try:
      with open(json_file, "r") as f:
        data = json.load(f)
        data["filename"] = json_file.stem
        all_results.append(data)
    except (IOError, OSError, json.JSONDecodeError) as e:
      print(f"Error loading {json_file}: {e}")
      continue

  if len(all_results) < 2:
    print("Could not load enough valid JSON files for comparison")
    return

  plt.figure(figsize=(18, 12))

  ax1 = plt.subplot(2, 3, (1, 2))
  _plot_tokenization_comparison(ax1, all_results)

  ax2 = plt.subplot(2, 3, 3)
  _plot_entity_comparison(ax2, all_results)

  ax3 = plt.subplot(2, 3, 4)
  _plot_time_comparison(ax3, all_results)

  ax4 = plt.subplot(2, 3, 5)
  _plot_success_rate_comparison(ax4, all_results)

  ax5 = plt.subplot(2, 3, 6)
  _plot_timeline(ax5, all_results)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  plt.suptitle(
      f"LangExtract Benchmark Comparison - {timestamp}",
      fontsize=14,
      fontweight="bold",
  )
  plt.tight_layout(rect=[0, 0.01, 1, 0.95])
  plt.subplots_adjust(hspace=0.45, wspace=0.35, top=0.93)
  plt.savefig(output_path, dpi=100, bbox_inches="tight")
  plt.close()
  print(f"Comparison plot saved to: {output_path}")


def _plot_entity_comparison(ax, all_results):
  """Plot entity count comparison across runs."""
  runs = []
  languages = ["english", "french", "spanish", "japanese"]
  language_data = []

  for result in all_results:
    run_name = result["filename"].replace("benchmark_", "")[:10]
    runs.append(run_name)

    run_counts = {lang: 0 for lang in languages}
    if config.RESULTS_KEY in result:
      for res in result[config.RESULTS_KEY]:
        lang = res.get("text_type", "")
        if lang in languages and res.get("success"):
          run_counts[lang] = res.get("entity_count", 0)

    language_data.append(run_counts)

  x = np.arange(len(runs))
  width = 0.2

  for i, lang in enumerate(languages):
    counts = [data[lang] for data in language_data]
    bars = ax.bar(x + i * width, counts, width, label=lang.capitalize())

    for bar_rect, count in zip(bars, counts):
      if count > 0:
        ax.text(
            bar_rect.get_x() + bar_rect.get_width() / 2,
            bar_rect.get_height() + 0.5,
            str(count),
            ha="center",
            fontsize=7,
        )

  ax.set_xlabel("Run")
  ax.set_ylabel("Entity Count")
  title = "Entities Extracted by Language\n"
  subtitle = "Number of unique character names found per language"
  ax.set_title(title, fontweight="bold", fontsize=10)
  ax.text(
      0.5,
      1.01,
      subtitle,
      transform=ax.transAxes,
      ha="center",
      fontsize=7,
      style="italic",
      color="#666666",
      va="bottom",
  )
  ax.set_xticks(x + width * 1.5)
  ax.set_xticklabels(runs, rotation=45, ha="right")
  ax.legend(loc="upper left", fontsize=8)
  ax.grid(True, alpha=0.3)
  ax.set_ylim(0, ax.get_ylim()[1] * 1.1)


def _plot_time_comparison(ax, all_results):
  """Plot processing time comparison."""
  runs = []
  avg_times = []

  for result in all_results:
    run_name = result["filename"].replace("benchmark_", "")[:10]
    runs.append(run_name)

    if config.RESULTS_KEY in result:
      times = [
          r.get("time_seconds", 0)
          for r in result[config.RESULTS_KEY]
          if r.get("success")
      ]
      avg_time = sum(times) / len(times) if times else 0
      avg_times.append(avg_time)
    else:
      avg_times.append(0)

  x_pos = np.arange(len(runs))
  bars = ax.bar(x_pos, avg_times, color="skyblue", edgecolor="navy", alpha=0.7)

  ax.set_xlabel("Run")
  ax.set_ylabel("Average Time (seconds)")
  title = "Average Processing Time\n"
  subtitle = "Mean extraction time across all language tests"
  ax.set_title(title, fontweight="bold", fontsize=10)
  ax.text(
      0.5,
      1.01,
      subtitle,
      transform=ax.transAxes,
      ha="center",
      fontsize=7,
      style="italic",
      color="#666666",
      va="bottom",
  )
  ax.set_xticks(x_pos)
  ax.set_xticklabels(runs, rotation=45, ha="right")
  ax.grid(True, alpha=0.3)

  for bar_rect, time in zip(bars, avg_times):
    if time > 0:
      ax.text(
          bar_rect.get_x() + bar_rect.get_width() / 2,
          bar_rect.get_height() + 0.1,
          f"{time:.1f}s",
          ha="center",
          fontsize=8,
      )

  if max(avg_times) > 0:
    ax.set_ylim(0, max(avg_times) * 1.2)


def _plot_tokenization_comparison(ax, all_results):
  """Plot tokenization throughput comparison as line graphs."""

  for i, result in enumerate(all_results):
    run_name = result["filename"].replace("benchmark_", "")[:10]

    if config.TOKENIZATION_KEY in result and result[config.TOKENIZATION_KEY]:
      sizes = [r["words"] for r in result[config.TOKENIZATION_KEY]]
      speeds = [r["tokens_per_sec"] for r in result[config.TOKENIZATION_KEY]]

      ax.semilogx(
          sizes,
          speeds,
          "o-",
          linewidth=2,
          markersize=6,
          label=run_name,
          alpha=0.8,
      )

      for x, y in zip(sizes, speeds):
        if i == 0:  # Only label first run to avoid overlap
          label = f"{y/1000:.0f}K"
          ax.annotate(
              label,
              xy=(x, y),
              xytext=(0, 5),
              textcoords="offset points",
              ha="center",
              fontsize=7,
          )

  ax.set_xlabel("Number of Words (log scale)")
  ax.set_ylabel("Tokens per Second")
  title = "Tokenization Throughput Comparison\n"
  subtitle = "Speed of text tokenization at different document sizes"
  ax.set_title(title, fontweight="bold", fontsize=10)
  ax.text(
      0.5,
      1.01,
      subtitle,
      transform=ax.transAxes,
      ha="center",
      fontsize=7,
      style="italic",
      color="#666666",
      va="bottom",
  )
  ax.grid(True, alpha=0.3)
  ax.legend(loc="best", fontsize=8)

  ax.set_xticks([100, 1000, 10000])
  ax.set_xticklabels(["10²", "10³", "10⁴"])

  _, ymax = ax.get_ylim()
  ax.set_ylim(0, ymax * 1.1)


def _plot_success_rate_comparison(ax, all_results):
  """Plot success rate comparison."""
  runs = []
  success_rates = []

  for result in all_results:
    run_name = result["filename"].replace("benchmark_", "")[:10]
    runs.append(run_name)

    if config.RESULTS_KEY in result:
      total = len(result[config.RESULTS_KEY])
      success = sum(1 for r in result[config.RESULTS_KEY] if r.get("success"))
      rate = (success / total * 100) if total > 0 else 0
      success_rates.append(rate)
    else:
      success_rates.append(0)

  x_pos = np.arange(len(runs))
  colors = [
      "green" if rate == 100 else "orange" if rate >= 75 else "red"
      for rate in success_rates
  ]
  bars = ax.bar(x_pos, success_rates, color=colors, alpha=0.7)

  ax.set_xlabel("Run")
  ax.set_ylabel("Success Rate (%)")
  title = "Extraction Success Rate\n"
  subtitle = "Percentage of language tests completed without errors"
  ax.set_title(title, fontweight="bold", fontsize=10)
  ax.text(
      0.5,
      1.01,
      subtitle,
      transform=ax.transAxes,
      ha="center",
      fontsize=7,
      style="italic",
      color="#666666",
      va="bottom",
  )
  ax.set_ylim(0, 105)
  ax.set_xticks(x_pos)
  ax.set_xticklabels(runs, rotation=45, ha="right")
  ax.axhline(y=100, color="green", linestyle="--", alpha=0.3)
  ax.grid(True, alpha=0.3)

  for bar_rect, rate in zip(bars, success_rates):
    ax.text(
        bar_rect.get_x() + bar_rect.get_width() / 2,
        bar_rect.get_height() + 1,
        f"{rate:.0f}%",
        ha="center",
        fontsize=8,
    )


def _plot_token_rate_by_language(ax, all_results):
  """Plot tokenization rates by language."""
  languages = ["english", "french", "spanish", "japanese"]
  latest_result = all_results[-1]

  token_rates = []
  colors = []

  if config.RESULTS_KEY in latest_result:
    for lang in languages:
      lang_results = [
          r
          for r in latest_result[config.RESULTS_KEY]
          if r.get("text_type") == lang and r.get("success")
      ]
      if lang_results and config.TOKENIZATION_KEY in lang_results[0]:
        rate = lang_results[0][config.TOKENIZATION_KEY].get(
            "tokens_per_char", 0
        )
        token_rates.append(rate)
        colors.append(
            "red" if rate < 0.1 else "orange" if rate < 0.2 else "green"
        )
      else:
        token_rates.append(0)
        colors.append("gray")

  ax.bar(languages, token_rates, color=colors, alpha=0.7)
  ax.set_xlabel("Language")
  ax.set_ylabel("Tokens per Character")
  ax.set_title("Tokenization Density (Latest Run)")
  ax.set_xticks(range(len(languages)))
  ax.set_xticklabels([l.capitalize() for l in languages])
  ax.grid(True, alpha=0.3)

  for i, (lang, rate) in enumerate(zip(languages, token_rates)):
    ax.text(i, rate + 0.01, f"{rate:.3f}", ha="center", fontsize=8)


def _plot_timeline(ax, all_results):
  """Plot metrics over time if timestamps available."""
  timestamps = []
  entity_totals = []

  for result in all_results:
    filename = result["filename"]
    if "timestamp" in result:
      timestamps.append(result["timestamp"])
    else:
      # Try to parse from filename (format: benchmark_YYYYMMDD_HHMMSS)
      parts = filename.split("_")
      if len(parts) >= 3:
        timestamps.append(f"{parts[-2]}_{parts[-1]}")
      else:
        timestamps.append(filename[:10])

    if config.RESULTS_KEY in result:
      total_entities = sum(
          r.get("entity_count", 0)
          for r in result[config.RESULTS_KEY]
          if r.get("success")
      )
      entity_totals.append(total_entities)
    else:
      entity_totals.append(0)

  x_pos = np.arange(len(timestamps))
  ax.plot(x_pos, entity_totals, "o-", color="blue", linewidth=2, markersize=8)
  ax.set_xlabel("Run")
  ax.set_ylabel("Total Entities")
  title = "Total Entities Over Time\n"
  subtitle = "Sum of all entities extracted across all languages"
  ax.set_title(title, fontweight="bold", fontsize=10)
  ax.text(
      0.5,
      1.01,
      subtitle,
      transform=ax.transAxes,
      ha="center",
      fontsize=7,
      style="italic",
      color="#666666",
      va="bottom",
  )
  ax.set_xticks(x_pos)
  ax.set_xticklabels([t[-6:] for t in timestamps], rotation=45, ha="right")
  ax.grid(True, alpha=0.3)

  for i, total in enumerate(entity_totals):
    ax.text(i, total + 1, str(total), ha="center", fontsize=8)

  if entity_totals:
    min_val = min(0, min(entity_totals) - 5)
    max_val = max(entity_totals) + 5
    ax.set_ylim(min_val, max_val)
