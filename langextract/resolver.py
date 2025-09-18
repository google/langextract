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

"""Library for resolving LLM output.

In the context of this module, a "resolver" is a component designed to parse and
transform the textual output of an LLM into structured data.
"""

from __future__ import annotations

import abc
import collections
from collections.abc import Iterator, Mapping, Sequence
import difflib
import functools
import itertools
import operator
from typing import Final

from absl import logging

from langextract.core import data
from langextract.core import exceptions
from langextract.core import format_handler as fh
from langextract.core import schema
from langextract.core import tokenizer

_FUZZY_ALIGNMENT_MIN_THRESHOLD = 0.75

# Default suffix for extraction index keys (e.g., "entity_index")
DEFAULT_INDEX_SUFFIX = "_index"  # Suffix for index fields in extraction sorting

ALIGNMENT_PARAM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "enable_fuzzy_alignment",
        "fuzzy_alignment_threshold",
        "accept_match_lesser",
    }
)


class AbstractResolver(abc.ABC):
    """Resolves LLM text outputs into structured data."""

    # TODO: Review value and requirements for abstract class.
    def __init__(
        self,
        fence_output: bool = True,
        constraint: schema.Constraint = schema.Constraint(),
        format_type: data.FormatType = data.FormatType.JSON,
    ):
        """Initializes the BaseResolver.

        Delimiters are used for parsing text blocks, and are used primarily for
        models that do not have constrained-decoding support.

        Args:
          fence_output: Whether to expect/generate fenced output (```json or
            ```yaml). When True, the model is prompted to generate fenced output and
            the resolver expects it. When False, raw JSON/YAML is expected. If your
            model utilizes schema constraints, this can generally be set to False
            unless the constraint also accounts for code fence delimiters.
          constraint: Applies constraint when decoding the output. Defaults to no
            constraint.
          format_type: The format type for the output (JSON or YAML).
        """
        self._fence_output = fence_output
        self._constraint = constraint
        self._format_type = format_type

    @property
    def fence_output(self) -> bool:
        """Returns whether fenced output is expected."""
        return self._fence_output

    @fence_output.setter
    def fence_output(self, fence_output: bool) -> None:
        """Sets whether fenced output is expected.

        Args:
          fence_output: Whether to expect fenced output.
        """
        self._fence_output = fence_output

    @property
    def format_type(self) -> data.FormatType:
        """Returns the format type."""
        return self._format_type

    @format_type.setter
    def format_type(self, new_format_type: data.FormatType) -> None:
        """Sets a new format type."""
        self._format_type = new_format_type

    @abc.abstractmethod
    def resolve(
        self,
        input_text: str,
        **kwargs,
    ) -> Sequence[data.Extraction]:
        """Run resolve function on input text.

        Args:
            input_text: The input text to be processed.
            **kwargs: Additional arguments for subclass implementations.

        Returns:
            Annotated text in the form of Extractions.
        """

    @abc.abstractmethod
    def align(
        self,
        extractions: Sequence[data.Extraction],
        source_text: str,
        token_offset: int,
        char_offset: int | None = None,
        enable_fuzzy_alignment: bool = True,
        fuzzy_alignment_threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD,
        accept_match_lesser: bool = True,
        **kwargs,
    ) -> Iterator[data.Extraction]:
        """Aligns extractions with source text, setting token/char intervals and alignment status.

        Uses exact matching first (difflib), then fuzzy alignment fallback if
        enabled.

        Alignment Status Results:
        - MATCH_EXACT: Perfect token-level match
        - MATCH_LESSER: Partial exact match (extraction longer than matched text)
        - MATCH_FUZZY: Best overlap window meets threshold (≥
        fuzzy_alignment_threshold)
        - None: No alignment found

        Args:
          extractions: Annotated extractions to align with the source text.
          source_text: The text in which to align the extractions.
          token_offset: The token_offset corresponding to the starting token index
            of the chunk.
          char_offset: The char_offset corresponding to the starting character index
            of the chunk.
          enable_fuzzy_alignment: Whether to use fuzzy alignment when exact matching
            fails.
          fuzzy_alignment_threshold: Minimum token overlap ratio for fuzzy alignment
            (0-1).
          accept_match_lesser: Whether to accept partial exact matches (MATCH_LESSER
            status).
          **kwargs: Additional keyword arguments for provider-specific alignment.

        Yields:
          Aligned extractions with updated token intervals and alignment status.
        """


class ResolverParsingError(exceptions.LangExtractError):
    """Error raised when content cannot be parsed as the given format."""


class Resolver(AbstractResolver):
    """Resolver for YAML/JSON-based information extraction.

    By default, extractions are returned in the order they appear in the model
    output. To enable index-based sorting, set extraction_index_suffix to a
    value like "_index" (the DEFAULT_INDEX_SUFFIX constant). This will sort
    extractions by fields ending with that suffix (e.g., "entity_index").

    Uses FormatHandler for parsing model output into extractions.
    """

    def __init__(
        self,
        format_handler: fh.FormatHandler | None = None,
        extraction_index_suffix: str | None = None,
        **kwargs,  # Collect legacy parameters
    ):
        """Constructor.

        Args:
          format_handler: The format handler that knows how to parse output.
          extraction_index_suffix: Suffix identifying index keys that determine the
            ordering of extractions.
          **kwargs: Legacy parameters (fence_output, format_type, etc.) for backward
            compatibility. These will be used to create a FormatHandler if one is not
            provided. Support for these parameters will be removed in v2.0.0.
        """
        constraint = kwargs.pop("constraint", None)
        extraction_attributes_suffix = kwargs.pop("extraction_attributes_suffix", None)

        if format_handler is None:
            if kwargs or extraction_attributes_suffix is not None:
                handler_kwargs = dict(kwargs)
                if extraction_attributes_suffix is not None:
                    handler_kwargs["attribute_suffix"] = extraction_attributes_suffix
                format_handler = fh.FormatHandler.from_kwargs(**handler_kwargs)
                for param in [
                    "fence_output",
                    "format_type",
                    "strict_fences",
                    "require_extractions_key",
                    "attribute_suffix",
                ]:
                    kwargs.pop(param, None)
            else:
                format_handler = fh.FormatHandler()

        if kwargs:
            raise TypeError(
                f"got an unexpected keyword argument '{list(kwargs.keys())[0]}'"
            )

        constraint = constraint or schema.Constraint()
        super().__init__(
            fence_output=format_handler.use_fences,
            format_type=format_handler.format_type,
            constraint=constraint,
        )
        self.format_handler = format_handler
        self.extraction_index_suffix = extraction_index_suffix
        self._constraint = constraint

    def resolve(
        self,
        input_text: str,
        suppress_parse_errors: bool = False,
        **kwargs,
    ) -> Sequence[data.Extraction]:
        """Runs resolve function on text with YAML/JSON extraction data.

        Args:
            input_text: The input text to be processed.
            suppress_parse_errors: Log errors and continue pipeline.
            **kwargs: Additional keyword arguments.

        Returns:
            Annotated text in the form of a sequence of data.Extraction objects.

        Raises:
            ResolverParsingError: If the content within the string cannot be parsed
            due to formatting errors, or if the parsed content is not as expected.
        """
        logging.info("Starting resolver process for input text.")
        logging.debug("Input Text: %s", input_text)

        try:
            constraint = getattr(self, "_constraint", schema.Constraint())
            strict = getattr(constraint, "strict", False)
            extraction_data = self.format_handler.parse_output(
                input_text, strict=strict
            )
            logging.debug("Parsed content: %s", extraction_data)

        except exceptions.FormatError as e:
            if suppress_parse_errors:
                logging.exception(
                    "Failed to parse input_text: %s, error: %s", input_text, e
                )
                return []
            raise ResolverParsingError(str(e)) from e

        processed_extractions = self.extract_ordered_extractions(extraction_data)

        logging.debug("Completed the resolver process.")

        return processed_extractions

    def align(
        self,
        extractions: Sequence[data.Extraction],
        source_text: str,
        token_offset: int,
        char_offset: int | None = None,
        enable_fuzzy_alignment: bool = True,
        fuzzy_alignment_threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD,
        accept_match_lesser: bool = True,
        **kwargs,
    ) -> Iterator[data.Extraction]:
        """Aligns annotated extractions with source text.

        This uses WordAligner which is based on Python's difflib SequenceMatcher to
        match tokens in the source text with tokens from the annotated extractions.
        If
        the extraction order is significantly different from the source text order,
        difflib may skip some matches, leaving certain extractions unmatched.

        Args:
          extractions: Annotated extractions.
          source_text: The text chunk in which to align the extractions.
          token_offset: The starting token index of the chunk.
          char_offset: The starting character index of the chunk.
          enable_fuzzy_alignment: Whether to enable fuzzy alignment fallback.
          fuzzy_alignment_threshold: Minimum overlap ratio required for fuzzy
            alignment.
          accept_match_lesser: Whether to accept partial exact matches (MATCH_LESSER
            status).
          **kwargs: Additional parameters.

        Yields:
            Iterator on aligned extractions.
        """
        logging.info("Starting alignment process for provided chunk text.")

        if not extractions:
            logging.debug(
                "No extractions found in the annotated text; exiting alignment"
                " process."
            )
            return
        else:
            extractions_group = [extractions]

        aligner = WordAligner()
        aligned_yaml_extractions = aligner.align_extractions(
            extractions_group,
            source_text,
            token_offset,
            char_offset or 0,
            enable_fuzzy_alignment=enable_fuzzy_alignment,
            fuzzy_alignment_threshold=fuzzy_alignment_threshold,
            accept_match_lesser=accept_match_lesser,
        )
        logging.debug(
            "Aligned extractions count: %d",
            sum(len(group) for group in aligned_yaml_extractions),
        )

        for extraction in itertools.chain(*aligned_yaml_extractions):
            logging.debug("Yielding aligned extraction: %s", extraction)
            yield extraction

        logging.info("Completed alignment process for the provided source_text.")

    def string_to_extraction_data(
        self,
        input_string: str,
    ) -> Sequence[Mapping[str, fh.ExtractionValueType]]:
        """Parses a YAML or JSON-formatted string into extraction data.

        This method is kept for backward compatibility with tests.
        It delegates to the FormatHandler for actual parsing.

        Args:
            input_string: A string containing YAML or JSON content.

        Returns:
            Sequence[Mapping[str, fh.ExtractionValueType]]: A sequence of parsed objects.

        Raises:
            ResolverParsingError: If the content within the string cannot be parsed.
            ValueError: If the input is invalid or does not contain expected format.
        """
        if not input_string or not isinstance(input_string, str):
            logging.error("Input string must be a non-empty string.")
            raise ValueError("Input string must be a non-empty string.")

        try:
            constraint = getattr(self, "_constraint", schema.Constraint())
            strict = getattr(constraint, "strict", False)
            return self.format_handler.parse_output(input_string, strict=strict)

        except exceptions.FormatError as e:
            raise ResolverParsingError(str(e)) from e

        except Exception as e:
            logging.exception("Failed to parse content.")
            raise ResolverParsingError("Failed to parse content.") from e

    def extract_ordered_extractions(
        self,
        extraction_data: Sequence[Mapping[str, fh.ExtractionValueType]],
    ) -> Sequence[data.Extraction]:
        """Extracts and orders extraction data based on their associated indexes.

        This function processes a list of dictionaries, each containing pairs of
        extraction class keys and their corresponding values, along with optionally
        associated index keys (identified by the index_suffix). It sorts these pairs
        by their indices in ascending order and excludes pairs without an index key,
        returning a list of lists of tuples (extraction_class: str, extraction_text:
        str).

        Args:
            extraction_data: A list of dictionaries. Each dictionary contains pairs
              of extraction class keys and their values, along with optional index
              keys.

        Returns:
            Extractions sorted by the index attribute or by order of appearance. If
            two
            extractions have the same index, their group order dictates the sorting
            order.
        Raises:
            ValueError: If the extraction text is not a string or integer, or if the
            index is not an integer.
        """
        logging.info("Starting to extract and order extractions from data.")

        if not extraction_data:
            logging.debug("Received empty extraction data.")

        processed_extractions = []
        extraction_index = 0
        index_suffix = self.extraction_index_suffix
        attributes_suffix = self.format_handler.attribute_suffix

        for group_index, group in enumerate(extraction_data):
            for extraction_class, extraction_value in group.items():
                if index_suffix and extraction_class.endswith(index_suffix):
                    if not isinstance(extraction_value, int):
                        logging.error(
                            "Index must be an integer. Found: %s",
                            type(extraction_value),
                        )
                        raise ValueError("Index must be an integer.")
                    continue

                if attributes_suffix and extraction_class.endswith(attributes_suffix):
                    if not isinstance(extraction_value, (dict, type(None))):
                        logging.error(
                            "Attributes must be a dict or None. Found: %s",
                            type(extraction_value),
                        )
                        raise ValueError(
                            "Extraction value must be a dict or None for attributes."
                        )
                    continue

                if not isinstance(extraction_value, (str, int, float)):
                    logging.error(
                        "Extraction text must be a string, integer, or float. Found: %s",
                        type(extraction_value),
                    )
                    raise ValueError(
                        "Extraction text must be a string, integer, or float."
                    )

                if not isinstance(extraction_value, str):
                    extraction_value = str(extraction_value)

                if index_suffix:
                    index_key = extraction_class + index_suffix
                    extraction_index = group.get(index_key, None)
                    if extraction_index is None:
                        logging.debug(
                            "No index value for %s. Skipping extraction.",
                            extraction_class,
                        )
                        continue
                else:
                    extraction_index += 1

                attributes = None
                if attributes_suffix:
                    attributes_key = extraction_class + attributes_suffix
                    attributes = group.get(attributes_key, None)

                processed_extractions.append(
                    data.Extraction(
                        extraction_class=extraction_class,
                        extraction_text=extraction_value,
                        extraction_index=extraction_index,
                        group_index=group_index,
                        attributes=attributes,
                    )
                )

        processed_extractions.sort(key=operator.attrgetter("extraction_index"))
        logging.info("Completed extraction and ordering of extractions.")
        return processed_extractions


class WordAligner:
    """Aligns words between two sequences of tokens using Python's difflib."""

    def __init__(self):
        """Initialize the WordAligner with difflib SequenceMatcher."""
        self.matcher = difflib.SequenceMatcher(autojunk=False)
        self.source_tokens: Sequence[str] | None = None
        self.extraction_tokens: Sequence[str] | None = None

    def _set_seqs(
        self,
        source_tokens: Sequence[str] | Iterator[str],
        extraction_tokens: Sequence[str] | Iterator[str],
    ):
        """Sets the source and extraction tokens for alignment.

        Args:
          source_tokens: A nonempty sequence or iterator of word-level tokens from
            source text.
          extraction_tokens: A nonempty sequence or iterator of extraction tokens in
            order for matching to the source.
        """

        if isinstance(source_tokens, Iterator):
            source_tokens = list(source_tokens)
        if isinstance(extraction_tokens, Iterator):
            extraction_tokens = list(extraction_tokens)

        if not source_tokens or not extraction_tokens:
            raise ValueError("Source tokens and extraction tokens cannot be empty.")

        self.source_tokens = source_tokens
        self.extraction_tokens = extraction_tokens
        self.matcher.set_seqs(a=source_tokens, b=extraction_tokens)

    def _get_matching_blocks(self) -> Sequence[tuple[int, int, int]]:
        """Utilizes difflib SequenceMatcher and returns matching blocks of tokens.

        Returns:
          Sequence of matching blocks between source_tokens (S) and
          extraction_tokens
          (E). Each block (i, j, n) conforms to: S[i:i+n] == E[j:j+n], guaranteed to
          be monotonically increasing in j. Final entry is a dummy with value
          (len(S), len(E), 0).
        """
        if self.source_tokens is None or self.extraction_tokens is None:
            raise ValueError(
                "Source tokens and extraction tokens must be set before getting"
                " matching blocks."
            )
        return self.matcher.get_matching_blocks()

    def _fuzzy_align_extraction(
        self,
        extraction: data.Extraction,
        source_tokens: list[str],
        tokenized_text: tokenizer.TokenizedText,
        token_offset: int,
        char_offset: int,
        fuzzy_alignment_threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD,
    ) -> data.Extraction | None:
        """Fuzzy-align an extraction using difflib.SequenceMatcher on tokens.

        The algorithm scans every candidate window in `source_tokens` and selects
        the window with the highest SequenceMatcher `ratio`. It uses an efficient
        token-count intersection as a fast pre-check to discard windows that cannot
        meet the alignment threshold. A match is accepted when the ratio is ≥
        `fuzzy_alignment_threshold`. This only runs on unmatched extractions, which
        is usually a small subset of the total extractions.

        Args:
          extraction: The extraction to align.
          source_tokens: The tokens from the source text.
          tokenized_text: The tokenized source text.
          token_offset: The token offset of the current chunk.
          char_offset: The character offset of the current chunk.
          fuzzy_alignment_threshold: The minimum ratio for a fuzzy match.

        Returns:
          The aligned data.Extraction if successful, None otherwise.
        """

        extraction_tokens = list(_tokenize_with_lowercase(extraction.extraction_text))
        # Work with lightly stemmed tokens so pluralisation doesn't block alignment
        extraction_tokens_norm = [_normalize_token(t) for t in extraction_tokens]

        if not extraction_tokens:
            return None

        logging.debug(
            "Fuzzy aligning %r (%d tokens)",
            extraction.extraction_text,
            len(extraction_tokens),
        )

        best_ratio = 0.0
        best_span: tuple[int, int] | None = None  # (start_idx, window_size)

        len_e = len(extraction_tokens)
        max_window = len(source_tokens)

        extraction_counts = collections.Counter(extraction_tokens_norm)
        min_overlap = int(len_e * fuzzy_alignment_threshold)

        matcher = difflib.SequenceMatcher(autojunk=False, b=extraction_tokens_norm)

        for window_size in range(len_e, max_window + 1):
            if window_size > len(source_tokens):
                break

            # Initialize for sliding window
            window_deque = collections.deque(source_tokens[0:window_size])
            window_counts = collections.Counter(
                [_normalize_token(t) for t in window_deque]
            )

            for start_idx in range(len(source_tokens) - window_size + 1):
                # Optimization: check if enough overlapping tokens exist before expensive
                # sequence matching. This is an upper bound on the match count.
                if (extraction_counts & window_counts).total() >= min_overlap:
                    window_tokens_norm = [_normalize_token(t) for t in window_deque]
                    matcher.set_seq1(window_tokens_norm)
                    matches = sum(size for _, _, size in matcher.get_matching_blocks())
                    if len_e > 0:
                        ratio = matches / len_e
                    else:
                        ratio = 0.0
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_span = (start_idx, window_size)

                # Slide the window to the right
                if start_idx + window_size < len(source_tokens):
                    # Remove the leftmost token from the count
                    old_token = window_deque.popleft()
                    old_token_norm = _normalize_token(old_token)
                    window_counts[old_token_norm] -= 1
                    if window_counts[old_token_norm] == 0:
                        del window_counts[old_token_norm]

                    # Add the new rightmost token to the deque and count
                    new_token = source_tokens[start_idx + window_size]
                    window_deque.append(new_token)
                    new_token_norm = _normalize_token(new_token)
                    window_counts[new_token_norm] += 1

        if best_span and best_ratio >= fuzzy_alignment_threshold:
            start_idx, window_size = best_span

            try:
                extraction.token_interval = tokenizer.TokenInterval(
                    start_index=start_idx + token_offset,
                    end_index=start_idx + window_size + token_offset,
                )

                start_token = tokenized_text.tokens[start_idx]
                end_token = tokenized_text.tokens[start_idx + window_size - 1]
                extraction.char_interval = data.CharInterval(
                    start_pos=char_offset + start_token.char_interval.start_pos,
                    end_pos=char_offset + end_token.char_interval.end_pos,
                )

                extraction.alignment_status = data.AlignmentStatus.MATCH_FUZZY
                return extraction
            except IndexError:
                logging.exception(
                    "Index error while setting intervals during fuzzy alignment."
                )
                return None

        return None

    def align_extractions(
        self,
        extraction_groups: Sequence[Sequence[data.Extraction]],
        source_text: str,
        token_offset: int = 0,
        char_offset: int = 0,
        delim: str = "\u241F",  # Unicode Symbol for unit separator
        enable_fuzzy_alignment: bool = True,
        fuzzy_alignment_threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD,
        accept_match_lesser: bool = True,
    ) -> Sequence[Sequence[data.Extraction]]:
        """Aligns extractions with their positions in the source text.

        This method takes a sequence of extractions and the source text, aligning
        each extraction with its corresponding position in the source text. It
        returns a sequence of extractions along with token intervals indicating the
        start and
        end positions of each extraction in the source text. If an extraction cannot
        be
        aligned, its token interval is set to None.

        Args:
          extraction_groups: A sequence of sequences, where each inner sequence
            contains an Extraction object.
          source_text: The source text against which extractions are to be aligned.
          token_offset: The offset to add to the start and end indices of the token
            intervals.
          char_offset: The offset to add to the start and end positions of the
            character intervals.
          delim: Token used to separate multi-token extractions.
          enable_fuzzy_alignment: Whether to use fuzzy alignment when exact matching
            fails.
          fuzzy_alignment_threshold: Minimum token overlap ratio for fuzzy alignment
            (0-1).
          accept_match_lesser: Whether to accept partial exact matches (MATCH_LESSER
            status).

        Returns:
          A sequence of extractions aligned with the source text, including token
          intervals.
        """
        logging.debug(
            "WordAligner: Starting alignment of extractions with the source text."
            " Extraction groups to align: %s",
            extraction_groups,
        )
        if not extraction_groups:
            logging.info("No extraction groups provided; returning empty list.")
            return []

        source_tokens = list(_tokenize_with_lowercase(source_text))

        delim_len = len(list(_tokenize_with_lowercase(delim)))
        if delim_len != 1:
            raise ValueError(f"Delimiter {delim!r} must be a single token.")

        logging.debug("Using delimiter %r for extraction alignment", delim)

        extraction_tokens = _tokenize_with_lowercase(
            f" {delim} ".join(
                extraction.extraction_text
                for extraction in itertools.chain(*extraction_groups)
            )
        )

        self._set_seqs(source_tokens, extraction_tokens)

        index_to_extraction_group = {}
        extraction_index = 0
        for group_index, group in enumerate(extraction_groups):
            logging.debug(
                "Processing extraction group %d with %d extractions.",
                group_index,
                len(group),
            )
            for extraction in group:
                # Validate delimiter doesn't appear in extraction text
                if delim in extraction.extraction_text:
                    raise ValueError(
                        f"Delimiter {delim!r} appears inside extraction text"
                        f" {extraction.extraction_text!r}. This would corrupt alignment"
                        " mapping."
                    )

                index_to_extraction_group[extraction_index] = (extraction, group_index)
                extraction_text_tokens = list(
                    _tokenize_with_lowercase(extraction.extraction_text)
                )
                extraction_index += len(extraction_text_tokens) + delim_len

        aligned_extraction_groups: list[list[data.Extraction]] = [
            [] for _ in extraction_groups
        ]
        tokenized_text = tokenizer.tokenize(source_text)

        # Track which extractions were aligned in the exact matching phase
        aligned_extractions = []
        exact_matches = 0
        lesser_matches = 0

        # Exact matching phase
        for i, j, n in self._get_matching_blocks()[:-1]:
            extraction, _ = index_to_extraction_group.get(j, (None, None))
            if extraction is None:
                logging.debug(
                    "No clean start index found for extraction index=%d iterating"
                    " Difflib matching_blocks",
                    j,
                )
                continue

            extraction.token_interval = tokenizer.TokenInterval(
                start_index=i + token_offset,
                end_index=i + n + token_offset,
            )

            try:
                start_token = tokenized_text.tokens[i]
                end_token = tokenized_text.tokens[i + n - 1]
                extraction.char_interval = data.CharInterval(
                    start_pos=char_offset + start_token.char_interval.start_pos,
                    end_pos=char_offset + end_token.char_interval.end_pos,
                )
            except IndexError as e:
                raise IndexError(
                    "Failed to align extraction with source text. Extraction token"
                    f" interval {extraction.token_interval} does not match source text"
                    f" tokens {tokenized_text.tokens}."
                ) from e

            extraction_text_len = len(
                list(_tokenize_with_lowercase(extraction.extraction_text))
            )
            if extraction_text_len < n:
                raise ValueError(
                    "Delimiter prevents blocks greater than extraction length: "
                    f"extraction_text_len={extraction_text_len}, block_size={n}"
                )
            if extraction_text_len == n:
                extraction.alignment_status = data.AlignmentStatus.MATCH_EXACT
                exact_matches += 1
                aligned_extractions.append(extraction)
            else:
                # Partial match (extraction longer than matched text)
                if accept_match_lesser:
                    extraction.alignment_status = data.AlignmentStatus.MATCH_LESSER
                    lesser_matches += 1
                    aligned_extractions.append(extraction)
                else:
                    # Reset intervals when not accepting lesser matches
                    extraction.token_interval = None
                    extraction.char_interval = None
                    extraction.alignment_status = None

        # Collect unaligned extractions
        unaligned_extractions = []
        for extraction, _ in index_to_extraction_group.values():
            if extraction not in aligned_extractions:
                unaligned_extractions.append(extraction)

        # Apply fuzzy alignment to remaining extractions
        if enable_fuzzy_alignment and unaligned_extractions:
            logging.debug(
                "Starting fuzzy alignment for %d unaligned extractions",
                len(unaligned_extractions),
            )
            for extraction in unaligned_extractions:
                aligned_extraction = self._fuzzy_align_extraction(
                    extraction,
                    source_tokens,
                    tokenized_text,
                    token_offset,
                    char_offset,
                    fuzzy_alignment_threshold,
                )
                if aligned_extraction:
                    aligned_extractions.append(aligned_extraction)
                    logging.debug(
                        "Fuzzy alignment successful for extraction: %s",
                        extraction.extraction_text,
                    )

        for extraction, group_index in index_to_extraction_group.values():
            aligned_extraction_groups[group_index].append(extraction)

        logging.debug("Final aligned extraction groups: %s", aligned_extraction_groups)
        return aligned_extraction_groups


def _tokenize_with_lowercase(text: str) -> Iterator[str]:
    """Extract and lowercase tokens from the input text into words.

    This function utilizes the tokenizer module to tokenize text and yields
    lowercased words.

    Args:
      text (str): The text to be tokenized.

    Yields:
      Iterator[str]: An iterator over tokenized words.
    """
    tokenized_pb2 = tokenizer.tokenize(text)
    original_text = tokenized_pb2.text
    for token in tokenized_pb2.tokens:
        start = token.char_interval.start_pos
        end = token.char_interval.end_pos
        token_str = original_text[start:end]
        token_str = token_str.lower()
        yield token_str


@functools.lru_cache(maxsize=10000)
def _normalize_token(token: str) -> str:
    """Lowercases and applies light pluralisation stemming."""
    token = token.lower()
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token
