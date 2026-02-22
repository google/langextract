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

"""Fast fuzzy alignment helpers.

These utilities are designed for cases where exhaustive sliding-window fuzzy
alignment becomes too slow (e.g., long example texts used for prompt validation).
They operate on *normalized token strings* to find a best-effort span in the
source tokens that maximizes the number of extraction tokens matched in order.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence


def build_token_positions(tokens: Sequence[str]) -> dict[str, list[int]]:
  """Build an inverted index of token -> sorted list of positions."""
  positions: dict[str, list[int]] = {}
  for i, tok in enumerate(tokens):
    positions.setdefault(tok, []).append(i)
  return positions


def _iter_limited_positions(
    positions: Sequence[int], *, max_items: int
) -> Sequence[int]:
  if max_items <= 0 or len(positions) <= max_items:
    return positions

  if max_items == 1:
    return [positions[len(positions) // 2]]

  stride = max(1, len(positions) // max_items)
  sampled = list(positions[0::stride])
  if len(sampled) > max_items:
    sampled = sampled[:max_items]
  if sampled[-1] != positions[-1] and len(sampled) < max_items:
    sampled.append(positions[-1])
  return sampled


def greedy_subsequence_match_positions(
    extraction_tokens: Sequence[str],
    token_positions: Mapping[str, Sequence[int]],
    *,
    start_at: int = 0,
) -> list[int]:
  """Greedily match extraction tokens as a subsequence of source tokens.

  Args:
    extraction_tokens: Tokens to match (in order).
    token_positions: Map of token -> sorted positions in the source token list.
    start_at: Earliest source index that the first match may use.

  Returns:
    List of matched source indices (monotonic increasing). May be empty.
  """
  if not extraction_tokens:
    return []

  matched: list[int] = []
  current = max(-1, start_at - 1)

  for tok in extraction_tokens:
    pos_list = token_positions.get(tok)
    if not pos_list:
      continue
    j = bisect.bisect_right(pos_list, current)
    if j >= len(pos_list):
      continue
    current = pos_list[j]
    matched.append(current)

  return matched


def generate_candidate_starts(
    extraction_tokens: Sequence[str],
    token_positions: Mapping[str, Sequence[int]],
    *,
    max_candidates: int,
    max_anchor_occurrences: int,
    max_anchors: int = 3,
) -> list[int]:
  """Generate candidate start indices based on rare anchor tokens.

  Candidates are derived by aligning occurrences of a small number of
  low-frequency extraction tokens ("anchors") to their positions in the
  extraction token list.
  """
  if max_candidates <= 0:
    return [0]

  token_to_extraction_indices: dict[str, list[int]] = {}
  for j, tok in enumerate(extraction_tokens):
    token_to_extraction_indices.setdefault(tok, []).append(j)

  # Prefer anchors that appear in the source and are rare there.
  anchors: list[tuple[int, str]] = []
  for tok in token_to_extraction_indices:
    freq = len(token_positions.get(tok, ()))
    if freq > 0:
      anchors.append((freq, tok))
  anchors.sort(key=lambda x: x[0])

  candidate_set: set[int] = {0}
  candidates: list[int] = [0]
  if len(candidates) >= max_candidates:
    return candidates

  for _, anchor in anchors[:max_anchors]:
    extraction_idxs = token_to_extraction_indices[anchor]
    source_positions = token_positions.get(anchor, ())
    limited_positions = _iter_limited_positions(
        source_positions, max_items=max_anchor_occurrences
    )
    for src_i in limited_positions:
      for ex_j in extraction_idxs:
        if len(candidates) >= max_candidates:
          return candidates
        start = src_i - ex_j
        start = start if start > 0 else 0
        if start in candidate_set:
          continue
        candidate_set.add(start)
        candidates.append(start)
        if len(candidates) >= max_candidates:
          return candidates

  return candidates


def find_best_fuzzy_span(
    source_tokens: Sequence[str],
    extraction_tokens: Sequence[str],
    *,
    max_candidates: int,
    max_anchor_occurrences: int,
) -> tuple[int, int, int] | None:
  """Find best source span for an extraction by subsequence matching.

  The "best" span maximizes the number of matched extraction tokens (in order).
  Ties are broken by preferring smaller spans, then earlier spans.

  Returns:
    Tuple of (start_index, end_index_exclusive, matched_tokens) or None.
  """
  if not extraction_tokens or not source_tokens:
    return None

  token_positions = build_token_positions(source_tokens)
  candidates = generate_candidate_starts(
      extraction_tokens,
      token_positions,
      max_candidates=max_candidates,
      max_anchor_occurrences=max_anchor_occurrences,
  )

  best: tuple[int, int, int] | None = None
  for start in candidates:
    matched_positions = greedy_subsequence_match_positions(
        extraction_tokens, token_positions, start_at=start
    )
    if not matched_positions:
      continue
    span_start = matched_positions[0]
    span_end = matched_positions[-1] + 1
    matched = len(matched_positions)

    if best is None:
      best = (span_start, span_end, matched)
      continue

    best_start, best_end, best_matched = best
    if matched > best_matched:
      best = (span_start, span_end, matched)
      continue
    if matched == best_matched:
      span_len = span_end - span_start
      best_len = best_end - best_start
      if span_len < best_len or (
          span_len == best_len and span_start < best_start
      ):
        best = (span_start, span_end, matched)

  return best
