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

"""Progress and visualization utilities for LangExtract.

This module provides progress bars and completion messages for terminal display.
Note that these utilities use direct print() with ANSI color codes for visual
appeal in interactive terminals. For production logging (e.g., to files), use
the unified logging system via langextract._logging.get_logger().

Configuration:
- progress_enabled: Controls whether progress bars and print_* messages are shown.
  Can be set via:
  - langextract.configure(progress_enabled=False)
  - Environment variable LANGEXTRACT_PROGRESS_ENABLED=0
  - Default: True
"""
from __future__ import annotations

import sys
from typing import Any
import urllib.parse

import tqdm

from langextract._logging import get_logger

# ANSI color codes for terminal output
BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Google Blue color for progress bars
GOOGLE_BLUE = "#4285F4"

logger = get_logger(__name__)


def _is_progress_enabled() -> bool:
  """Check if progress display is enabled.

  Returns:
    True if progress bars and print_* messages should be shown.
  """
  try:
    from langextract._config import get_global_config

    config = get_global_config()
    return config.progress_enabled
  except Exception:
    return True


def _strip_ansi(text: str) -> str:
  """Strip ANSI color codes from text.

  Args:
    text: Text containing ANSI codes.

  Returns:
    Text without ANSI codes.
  """
  import re

  ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
  return ansi_escape.sub("", text)


def create_download_progress_bar(
    total_size: int, url: str, ncols: int = 100, max_url_length: int = 50
) -> tqdm.tqdm:
  """Create a styled progress bar for downloads.

  Args:
    total_size: Total size in bytes.
    url: The URL being downloaded.
    ncols: Number of columns for the progress bar.
    max_url_length: Maximum length to show for the URL.

  Returns:
    A configured tqdm progress bar.
  """
  # Truncate URL if too long, keeping the domain and end
  if len(url) > max_url_length:
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc or parsed.hostname or "unknown"

    path_parts = parsed.path.strip("/").split("/")
    filename = path_parts[-1] if path_parts and path_parts[-1] else "file"

    available = max_url_length - len(domain) - len(filename) - 5
    if available > 0:
      url_display = f"{domain}/.../{filename}"
    else:
      url_display = url[: max_url_length - 3] + "..."
  else:
    url_display = url

  disable = not _is_progress_enabled()

  return tqdm.tqdm(
      total=total_size,
      unit="B",
      unit_scale=True,
      desc=(
          f"{BLUE}{BOLD}LangExtract{RESET}: Downloading"
          f" {GREEN}{url_display}{RESET}"
      ),
      bar_format=(
          "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
          " [{elapsed}<{remaining}, {rate_fmt}]"
      ),
      colour=GOOGLE_BLUE,
      ncols=ncols,
      disable=disable,
  )


def create_extraction_progress_bar(
    iterable: Any, model_info: str | None = None, disable: bool = False
) -> tqdm.tqdm:
  """Create a styled progress bar for extraction.

  Args:
    iterable: The iterable to wrap with progress bar.
    model_info: Optional model information to display (e.g., "gemini-1.5-pro").
    disable: Whether to disable the progress bar (overrides config).

  Returns:
    A configured tqdm progress bar.
  """
  desc = format_extraction_progress(model_info)

  effective_disable = disable or not _is_progress_enabled()

  return tqdm.tqdm(
      iterable,
      desc=desc,
      bar_format="{desc} [{elapsed}]",
      disable=effective_disable,
      dynamic_ncols=True,
  )


def print_download_complete(
    char_count: int, word_count: int, filename: str
) -> None:
  """Print a styled download completion message.

  This function outputs to stdout with ANSI colors for terminal display.
  It also logs the same information (without ANSI codes) at INFO level.

  Args:
    char_count: Number of characters downloaded.
    word_count: Number of words downloaded.
    filename: Name of the downloaded file.
  """
  message = (
      f"{GREEN}✓{RESET} Downloaded {BOLD}{char_count:,}{RESET} characters "
      f"({BOLD}{word_count:,}{RESET} words) from {BLUE}{filename}{RESET}"
  )

  logger.info(
      "Downloaded %d characters (%d words) from %s",
      char_count,
      word_count,
      filename,
  )

  if _is_progress_enabled():
    print(message, flush=True, file=sys.stdout)


def print_extraction_complete() -> None:
  """Print a generic extraction completion message.

  This function outputs to stdout with ANSI colors for terminal display.
  It also logs the same information (without ANSI codes) at INFO level.
  """
  message = f"{GREEN}✓{RESET} Extraction processing complete"

  logger.info("Extraction processing complete")

  if _is_progress_enabled():
    print(message, flush=True, file=sys.stdout)


def print_extraction_summary(
    num_extractions: int,
    unique_classes: int,
    elapsed_time: float | None = None,
    chars_processed: int | None = None,
    num_chunks: int | None = None,
) -> None:
  """Print a styled extraction summary with optional performance metrics.

  This function outputs to stdout with ANSI colors for terminal display.
  It also logs the same information (without ANSI codes) at INFO level.

  Args:
    num_extractions: Total number of extractions.
    unique_classes: Number of unique extraction classes.
    elapsed_time: Optional elapsed time in seconds.
    chars_processed: Optional number of characters processed.
    num_chunks: Optional number of chunks processed.
  """
  main_message = (
      f"{GREEN}✓{RESET} Extracted {BOLD}{num_extractions}{RESET} entities "
      f"({BOLD}{unique_classes}{RESET} unique types)"
  )

  logger.info("Extracted %d entities (%d unique types)", num_extractions, unique_classes)

  metrics: list[str] = []
  log_metrics: list[str] = []

  if elapsed_time is not None:
    # Time
    metrics.append(f"Time: {BOLD}{elapsed_time:.2f}s{RESET}")
    log_metrics.append(f"Time: {elapsed_time:.2f}s")

    # Speed
    if chars_processed is not None and elapsed_time > 0:
      speed = chars_processed / elapsed_time
      metrics.append(f"Speed: {BOLD}{speed:,.0f}{RESET} chars/sec")
      log_metrics.append(f"Speed: {speed:,.0f} chars/sec")

    if num_chunks is not None:
      metrics.append(f"Chunks: {BOLD}{num_chunks}{RESET}")
      log_metrics.append(f"Chunks: {num_chunks}")

  if log_metrics:
    logger.info("Performance: %s", ", ".join(log_metrics))

  if _is_progress_enabled():
    print(main_message, flush=True, file=sys.stdout)
    for metric in metrics:
      print(f"  {CYAN}•{RESET} {metric}", flush=True, file=sys.stdout)


def create_save_progress_bar(
    output_path: str, disable: bool = False
) -> tqdm.tqdm:
  """Create a progress bar for saving documents.

  Args:
    output_path: The output file path.
    disable: Whether to disable the progress bar.

  Returns:
    A configured tqdm progress bar.
  """
  filename = output_path.split("/")[-1]
  effective_disable = disable or not _is_progress_enabled()

  return tqdm.tqdm(
      desc=(
          f"{BLUE}{BOLD}LangExtract{RESET}: Saving to {GREEN}{filename}{RESET}"
      ),
      unit=" docs",
      disable=effective_disable,
  )


def create_load_progress_bar(
    file_path: str, total_size: int | None = None, disable: bool = False
) -> tqdm.tqdm:
  """Create a progress bar for loading documents.

  Args:
    file_path: The file path being loaded.
    total_size: Optional total file size in bytes.
    disable: Whether to disable the progress bar.

  Returns:
    A configured tqdm progress bar.
  """
  filename = file_path.split("/")[-1]
  effective_disable = disable or not _is_progress_enabled()

  if total_size:
    return tqdm.tqdm(
        total=total_size,
        desc=(
            f"{BLUE}{BOLD}LangExtract{RESET}: Loading {GREEN}{filename}{RESET}"
        ),
        unit="B",
        unit_scale=True,
        disable=effective_disable,
    )
  else:
    return tqdm.tqdm(
        desc=(
            f"{BLUE}{BOLD}LangExtract{RESET}: Loading {GREEN}{filename}{RESET}"
        ),
        unit=" docs",
        disable=effective_disable,
    )


def print_save_complete(num_docs: int, file_path: str) -> None:
  """Print a save completion message.

  This function outputs to stdout with ANSI colors for terminal display.
  It also logs the same information (without ANSI codes) at INFO level.

  Args:
    num_docs: Number of documents saved.
    file_path: Path to the saved file.
  """
  filename = file_path.split("/")[-1]
  message = (
      f"{GREEN}✓{RESET} Saved {BOLD}{num_docs}{RESET} documents to"
      f" {GREEN}{filename}{RESET}"
  )

  logger.info("Saved %d documents to %s", num_docs, filename)

  if _is_progress_enabled():
    print(message, flush=True, file=sys.stdout)


def print_load_complete(num_docs: int, file_path: str) -> None:
  """Print a load completion message.

  This function outputs to stdout with ANSI colors for terminal display.
  It also logs the same information (without ANSI codes) at INFO level.

  Args:
    num_docs: Number of documents loaded.
    file_path: Path to the loaded file.
  """
  filename = file_path.split("/")[-1]
  message = (
      f"{GREEN}✓{RESET} Loaded {BOLD}{num_docs}{RESET} documents from"
      f" {GREEN}{filename}{RESET}"
  )

  logger.info("Loaded %d documents from %s", num_docs, filename)

  if _is_progress_enabled():
    print(message, flush=True, file=sys.stdout)


def get_model_info(language_model: Any) -> str | None:
  """Extract model information from a language model instance.

  Args:
    language_model: A language model instance.

  Returns:
    A string describing the model, or None if not available.
  """
  if hasattr(language_model, "model_id"):
    return language_model.model_id

  if hasattr(language_model, "model_url"):
    return language_model.model_url

  return None


def format_extraction_stats(current_chars: int, processed_chars: int) -> str:
  """Format extraction progress statistics with colors.

  Args:
    current_chars: Number of characters in current batch.
    processed_chars: Total number of characters processed so far.

  Returns:
    Formatted string with colored statistics.
  """
  current_str = f"{GREEN}{current_chars:,}{RESET}"
  processed_str = f"{GREEN}{processed_chars:,}{RESET}"
  return f"current={current_str} chars, processed={processed_str} chars"


def create_extraction_postfix(current_chars: int, processed_chars: int) -> str:
  """Create a formatted postfix string for extraction progress.

  Args:
    current_chars: Number of characters in current batch.
    processed_chars: Total number of characters processed so far.

  Returns:
    Formatted string with statistics.
  """
  current_str = f"{GREEN}{current_chars:,}{RESET}"
  processed_str = f"{GREEN}{processed_chars:,}{RESET}"
  return f"current={current_str} chars, processed={processed_str} chars"


def format_extraction_progress(
    model_info: str | None,
    current_chars: int | None = None,
    processed_chars: int | None = None,
) -> str:
  """Format the complete extraction progress bar description.

  Args:
    model_info: Optional model information (e.g., "gemini-2.0-flash").
    current_chars: Number of characters in current batch (optional).
    processed_chars: Total number of characters processed so far (optional).

  Returns:
    Formatted description string.
  """
  if model_info:
    desc = f"{BLUE}{BOLD}LangExtract{RESET}: model={GREEN}{model_info}{RESET}"
  else:
    desc = f"{BLUE}{BOLD}LangExtract{RESET}: Processing"

  if current_chars is not None and processed_chars is not None:
    current_str = f"{GREEN}{current_chars:,}{RESET}"
    processed_str = f"{GREEN}{processed_chars:,}{RESET}"
    desc += f", current={current_str} chars, processed={processed_str} chars"

  return desc


def create_pass_progress_bar(
    total_passes: int, disable: bool = False
) -> tqdm.tqdm:
  """Create a progress bar for sequential extraction passes.

  Args:
    total_passes: Total number of sequential passes.
    disable: Whether to disable the progress bar.

  Returns:
    A configured tqdm progress bar.
  """
  effective_disable = disable or not _is_progress_enabled()
  desc = f"{BLUE}{BOLD}LangExtract{RESET}: Extraction passes"

  return tqdm.tqdm(
      total=total_passes,
      desc=desc,
      bar_format=(
          "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
      ),
      disable=effective_disable,
      colour=GOOGLE_BLUE,
      ncols=100,
  )
