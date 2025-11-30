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

"""Helper functions for benchmark text retrieval and analysis."""

import subprocess
from typing import Any
import urllib.error
import urllib.request

from benchmarks import config
from langextract.core import tokenizer


def download_text(url: str) -> str:
  """Download text from URL.

  Args:
    url: URL to download from.

  Returns:
    Downloaded text content.
  """
  try:
    with urllib.request.urlopen(url) as response:
      return response.read().decode("utf-8")
  except (urllib.error.URLError, urllib.error.HTTPError) as e:
    raise RuntimeError(f"Could not download from {url}: {e}") from e


def extract_text_content(full_text: str) -> str:
  """Extract main content from Gutenberg text.

  Skips headers and footers by taking middle 60% of text.

  Args:
    full_text: Full text including Gutenberg headers.

  Returns:
    Extracted main content.
  """
  start_marker = "*** START OF"
  end_marker = "*** END OF"

  start_idx = full_text.upper().find(start_marker)
  end_idx = full_text.upper().find(end_marker)

  if start_idx != -1 and end_idx != -1:
    content_start = full_text.find("\n", start_idx) + 1

    # Handle markers with trailing asterisks (e.g., "*** START ... ***").
    line_end = full_text.find("***", start_idx + 3)
    if (
        line_end != -1 and line_end < content_start + 100
    ):  # Ensure marker is on same line.
      content_start = full_text.find("\n", line_end) + 1

    return full_text[content_start:end_idx].strip()

  text_length = len(full_text)
  start = int(text_length * 0.2)
  end = int(text_length * 0.8)
  return full_text[start:end].strip()


def get_text_from_gutenberg(text_type: config.TextTypes) -> str:
  """Get text from Project Gutenberg for given language.

  Args:
    text_type: Type of text (language).

  Returns:
    Text sample from Gutenberg.
  """
  url = config.GUTENBERG_TEXTS[text_type]
  full_text = download_text(url)
  content = extract_text_content(full_text)

  mid_point = len(content) // 2
  start_chunk = max(0, mid_point - 2500)
  return content[start_chunk : start_chunk + 5000].strip()


def get_optimal_text_size(text: str, model_id: str) -> str:
  """Get optimal text size for model.

  Args:
    text: Original text.
    model_id: Model identifier.

  Returns:
    Text truncated to optimal size.
  """
  if (
      ":" in model_id
      or "gemma" in model_id.lower()
      or "llama" in model_id.lower()
  ):
    max_chars = 500  # Smaller context for local models.
  else:
    max_chars = 5000

  return text[:max_chars]


def get_extraction_example(text_type: config.TextTypes) -> dict[str, str]:  # pylint: disable=unused-argument
  """Get extraction example configuration.

  Args:
    text_type: Type of text.

  Returns:
    Dictionary with prompt configuration.
  """
  return {
      "prompt": "Extract all character names from this text",
  }


def get_git_info() -> dict[str, str]:
  """Get current git branch and commit info.

  Returns:
    Dictionary with branch and commit info.
  """
  try:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    if status:
      commit += "-dirty"

    return {"branch": branch, "commit": commit}
  except subprocess.CalledProcessError:
    return {"branch": "unknown", "commit": "unknown"}


def analyze_tokenization(
    text: str, tokenizer_inst: tokenizer.Tokenizer | None = None
) -> dict[str, Any]:
  """Analyze tokenization of given text.

  Args:
    text: Text to analyze.
    tokenizer_inst: Tokenizer instance to use (default: RegexTokenizer).

  Returns:
    Dictionary with tokenization metrics.
  """
  if tokenizer_inst:
    tokenized = tokenizer_inst.tokenize(text)
  else:
    tokenized = tokenizer.tokenize(text)
  num_tokens = len(tokenized.tokens)
  num_chars = len(text)
  tokens_per_char = num_tokens / num_chars if num_chars > 0 else 0

  return {
      "num_tokens": num_tokens,
      "num_chars": num_chars,
      "tokens_per_char": tokens_per_char,
  }


def format_tokenization_summary(analysis: dict[str, Any]) -> str:
  """Format tokenization analysis as summary string.

  Args:
    analysis: Tokenization analysis dict.

  Returns:
    Formatted summary string.
  """
  return (
      f"{analysis['num_tokens']} tokens, "
      f"{analysis['tokens_per_char']:.3f} tok/char"
  )
