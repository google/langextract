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

"""Benchmark for fuzzy alignment in the resolver.

Measures wall-time and correctness of _fuzzy_align_extraction across
realistic input sizes. Run from repo root:

  python benchmarks/fuzzy_benchmark.py
  python benchmarks/fuzzy_benchmark.py --sizes planted_contiguous,perf_1k
  python benchmarks/fuzzy_benchmark.py --sizes large --runs 1
  python benchmarks/fuzzy_benchmark.py --tokenizer unicode
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time

from langextract import resolver as resolver_lib
from langextract.core import data
from langextract.core import tokenizer as tokenizer_lib

_WORD_POOL = [
    "patient",
    "diagnosed",
    "with",
    "diabetes",
    "hypertension",
    "medication",
    "prescribed",
    "daily",
    "chronic",
    "condition",
    "treatment",
    "history",
    "symptoms",
    "blood",
    "pressure",
    "glucose",
    "insulin",
    "kidney",
    "liver",
    "cardiac",
    "pulmonary",
    "neurological",
    "assessment",
    "examination",
    "laboratory",
    "results",
    "normal",
    "elevated",
    "decreased",
    "follow",
    "appointment",
    "scheduled",
    "monitor",
    "progress",
    "clinical",
    "evaluation",
    "imaging",
    "therapy",
    "dosage",
    "adverse",
    "reaction",
    "prognosis",
    "referral",
    "discharge",
    "admission",
    "surgery",
    "recovery",
    "emergency",
    "outpatient",
    "inpatient",
    "consultation",
    "diagnosis",
    "pathology",
    "specimen",
    "biopsy",
    "cultures",
    "antibiotics",
    "analgesic",
    "sedation",
    "ventilation",
    "intubation",
    "catheter",
    "drainage",
    "infusion",
]


def _generate_source_text(n_tokens: int, seed: int = 42) -> str:
  """Generates deterministic source text from _WORD_POOL."""
  rng = random.Random(seed)
  words = [rng.choice(_WORD_POOL) for _ in range(n_tokens)]
  return " ".join(words)


def _plant_span(source: str, target: str, position: int) -> str:
  """Inserts target text at approximately token position in source."""
  words = source.split()
  target_words = target.split()
  pos = min(position, len(words))
  words[pos : pos + len(target_words)] = target_words
  return " ".join(words)


def _plant_gapped(source: str, tokens: list[str], start: int, gap: int) -> str:
  """Inserts tokens with gaps between them in source."""
  words = source.split()
  for i, token in enumerate(tokens):
    pos = min(start + i * (gap + 1), len(words) - 1)
    words[pos] = token
  return " ".join(words)


def _make_extraction(text: str) -> data.Extraction:
  return data.Extraction(
      extraction_class="entity",
      extraction_text=text,
  )


def _build_cases() -> dict[str, dict]:
  """Builds benchmark cases with planted spans for correctness oracles."""
  cases = {}

  # --- Planted correctness cases (small, fast) ---

  base_200 = _generate_source_text(200, seed=42)

  # Contiguous positive: plant exact 3-token span at known position.
  planted_source = _plant_span(base_200, "metformin hydrochloride tablet", 50)
  cases["planted_contiguous"] = {
      "description": "3-token planted contiguous match in 200 tokens",
      "source": planted_source,
      "extraction_text": "metformin hydrochloride tablet",
      "expect_match": True,
      "expect_token_interval": (50, 53),
      "expect_char_interval": (451, 481),
      "expect_substring": "metformin hydrochloride tablet",
  }

  # Fuzzy positive: extraction has stemming variation.
  cases["planted_fuzzy"] = {
      "description": "3-token fuzzy match (stemming) in 200 tokens",
      "source": planted_source,
      "extraction_text": "metformins hydrochlorides tablets",
      "expect_match": True,
      "expect_token_interval": (50, 53),
      "expect_char_interval": (451, 481),
      "expect_substring": "metformin hydrochloride tablet",
  }

  # Gapped positive: extraction tokens scattered with noise between them.
  gapped_source = _plant_gapped(
      _generate_source_text(200, seed=99),
      ["metformin", "hydrochloride", "tablet"],
      start=40,
      gap=3,
  )
  cases["planted_gapped"] = {
      "description": "3-token gapped match (gap=3) in 200 tokens",
      "source": gapped_source,
      "extraction_text": "metformin hydrochloride tablet",
      "expect_match": True,
      "expect_token_interval": (40, 49),
      "expect_char_interval": (371, 461),
      "expect_substring": (
          "metformin pulmonary antibiotics assessment"
          " hydrochloride hypertension pressure with tablet"
      ),
  }

  # Near-miss negative: tokens not present in source.
  cases["planted_negative"] = {
      "description": "3-token near-miss negative in 200 tokens",
      "source": base_200,
      "extraction_text": "warfarin coumadin anticoagulant",
      "expect_match": False,
  }

  # --- Perf stress case (in-vocabulary extraction, keeps overlap filter hot) ---

  source_perf = _generate_source_text(1000, seed=42)
  cases["perf_1k"] = {
      "description": "5-token in-vocab extraction, 1000-token source (perf)",
      "source": source_perf,
      "extraction_text": "patient diagnosed chronic condition treatment",
  }

  # --- Scale cases (opt-in) ---

  source_large = _generate_source_text(5000, seed=42)
  cases["large"] = {
      "description": "5-token in-vocab extraction, 5000-token source (opt-in)",
      "source": source_large,
      "extraction_text": "patient diagnosed chronic condition treatment",
  }

  source_stress = _generate_source_text(10000, seed=42)
  cases["stress"] = {
      "description": "5-token in-vocab extraction, 10000-token source (opt-in)",
      "source": source_stress,
      "extraction_text": "patient diagnosed chronic condition treatment",
  }

  return cases


_DEFAULT_SIZES = (
    "planted_contiguous,planted_fuzzy,planted_gapped,planted_negative,perf_1k"
)


def _get_metadata(tokenizer_name: str, seed: int, threshold: float) -> dict:
  """Collects run metadata for reproducibility."""
  git_sha = "unknown"
  try:
    git_sha = (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )
  except (subprocess.CalledProcessError, FileNotFoundError):
    pass

  return {
      "python_version": platform.python_version(),
      "platform": platform.platform(),
      "tokenizer": tokenizer_name,
      "seed": seed,
      "fuzzy_alignment_threshold": threshold,
      "git_sha": git_sha,
  }


def _run_single(
    aligner: resolver_lib.WordAligner,
    source_text: str,
    extraction_text: str,
    tokenizer: tokenizer_lib.Tokenizer,
    threshold: float,
) -> dict:
  """Runs a single fuzzy alignment and returns timing + result."""
  resolver_lib._normalize_token.cache_clear()

  tokenized = tokenizer.tokenize(source_text)
  source_tokens = [t.lower() for t in _tokenize_words(source_text, tokenizer)]
  extraction = _make_extraction(extraction_text)

  start = time.perf_counter()
  result = aligner._fuzzy_align_extraction(
      extraction=extraction,
      source_tokens=source_tokens,
      tokenized_text=tokenized,
      token_offset=0,
      char_offset=0,
      fuzzy_alignment_threshold=threshold,
      tokenizer_impl=tokenizer,
  )
  elapsed = time.perf_counter() - start

  matched_substring = None
  if result and result.char_interval:
    start_pos = result.char_interval.start_pos
    end_pos = result.char_interval.end_pos
    matched_substring = source_text[start_pos:end_pos]

  return {
      "elapsed_ms": round(elapsed * 1000, 2),
      "matched": result is not None,
      "alignment_status": result.alignment_status.value if result else None,
      "token_interval": (
          f"{result.token_interval.start_index}"
          f"-{result.token_interval.end_index}"
          if result and result.token_interval
          else None
      ),
      "char_interval": (
          f"{result.char_interval.start_pos}-{result.char_interval.end_pos}"
          if result and result.char_interval
          else None
      ),
      "matched_substring": matched_substring,
  }


def _tokenize_words(text: str, tokenizer: tokenizer_lib.Tokenizer) -> list[str]:
  """Extracts word strings from tokenized text."""
  tokenized = tokenizer.tokenize(text)
  return [
      text[t.char_interval.start_pos : t.char_interval.end_pos]
      for t in tokenized.tokens
  ]


def main():
  parser = argparse.ArgumentParser(
      description="Benchmark fuzzy alignment performance"
  )
  parser.add_argument(
      "--sizes",
      default=_DEFAULT_SIZES,
      help="Comma-separated case names (default: planted + perf_1k)",
  )
  parser.add_argument(
      "--runs", type=int, default=3, help="Number of runs per case"
  )
  parser.add_argument(
      "--tokenizer",
      choices=["regex", "unicode"],
      default="regex",
      help="Tokenizer backend (default: regex)",
  )
  parser.add_argument(
      "--threshold",
      type=float,
      default=0.75,
      help="Fuzzy alignment threshold (default: 0.75)",
  )
  parser.add_argument("--json-output", help="Write results to JSON file")
  args = parser.parse_args()

  cases = _build_cases()
  selected = [s.strip() for s in args.sizes.split(",")]

  if args.tokenizer == "unicode":
    tokenizer = tokenizer_lib.UnicodeTokenizer()
  else:
    tokenizer = tokenizer_lib.RegexTokenizer()

  aligner = resolver_lib.WordAligner()
  metadata = _get_metadata(args.tokenizer, 42, args.threshold)

  results = {"_metadata": metadata}
  print(f"Fuzzy alignment benchmark ({args.runs} runs per case)\n")
  print(f"  tokenizer: {args.tokenizer}")
  print(f"  threshold: {args.threshold}")
  print(f"  git: {metadata['git_sha']}\n")

  for name in selected:
    if name not in cases:
      print(f"  {name}: unknown case, skipping\n")
      continue

    case = cases[name]
    source = case["source"]
    extraction_text = case["extraction_text"]
    expect_match = case.get("expect_match")
    n_source_tokens = len(_tokenize_words(source, tokenizer))

    print(f"  {name}: {case['description']}", flush=True)
    print(f"    source tokens: {n_source_tokens}", flush=True)

    expect_token = case.get("expect_token_interval")
    expect_char = case.get("expect_char_interval")
    expect_sub = case.get("expect_substring")

    timings = []
    last_result = None
    correctness = "n/a"
    for i in range(args.runs):
      print(f"    run {i + 1}/{args.runs}...", end="", flush=True)
      result = _run_single(
          aligner, source, extraction_text, tokenizer, args.threshold
      )
      timings.append(result["elapsed_ms"])
      last_result = result
      print(f" {result['elapsed_ms']:.1f}ms", flush=True)

      # Check oracle on every run. All configured expectations are checked.
      if expect_match is not None:
        if result["matched"] != expect_match:
          correctness = "FAIL"
          print(f"    FAIL: expected matched={expect_match}", flush=True)
        if expect_token and result["token_interval"]:
          expected = f"{expect_token[0]}-{expect_token[1]}"
          if result["token_interval"] != expected:
            correctness = "FAIL"
            print(
                f"    FAIL: token_interval {result['token_interval']}"
                f" != {expected}",
                flush=True,
            )
        if expect_char and result["char_interval"]:
          expected = f"{expect_char[0]}-{expect_char[1]}"
          if result["char_interval"] != expected:
            correctness = "FAIL"
            print(
                f"    FAIL: char_interval {result['char_interval']}"
                f" != {expected}",
                flush=True,
            )
        if expect_sub and result["matched_substring"] != expect_sub:
          correctness = "FAIL"
          print("    FAIL: substring mismatch", flush=True)

    if correctness != "FAIL":
      correctness = "PASS" if expect_match is not None else "n/a"

    avg_ms = sum(timings) / len(timings)
    min_ms = min(timings)
    max_ms = max(timings)

    print(f"    avg: {avg_ms:.1f}ms  min: {min_ms:.1f}ms  max: {max_ms:.1f}ms")
    print(f"    matched: {last_result['matched']}  correctness: {correctness}")
    if last_result["matched_substring"]:
      sub = last_result["matched_substring"]
      if len(sub) > 80:
        sub = sub[:80] + "..."
      print(f"    substring: {sub!r}")
    print(flush=True)

    results[name] = {
        "description": case["description"],
        "source_tokens": n_source_tokens,
        "runs": args.runs,
        "avg_ms": round(avg_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "matched": last_result["matched"],
        "correctness": correctness,
        "token_interval": last_result["token_interval"],
        "char_interval": last_result["char_interval"],
        "matched_substring": last_result["matched_substring"],
    }

  if args.json_output:
    with open(args.json_output, "w") as f:
      json.dump(results, f, indent=2)
    print(f"Results written to {args.json_output}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
