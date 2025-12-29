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

"""Provides functionality for annotating medical text using a language model.

The annotation process involves tokenizing the input text, generating prompts
for the language model, and resolving the language model's output into
structured annotations.

Usage example:
    annotator = Annotator(language_model, prompt_template)
    annotated_documents = annotator.annotate_documents(documents, resolver)
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Iterator
import time
from typing import DefaultDict

from absl import logging

from langextract import chunking
from langextract import progress
from langextract import prompting
from langextract import resolver as resolver_lib
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import format_handler as fh
from langextract.core import tokenizer as tokenizer_lib


def _merge_non_overlapping_extractions(
    all_extractions: list[Iterable[data.Extraction]],
) -> list[data.Extraction]:
  """Merges extractions from multiple extraction passes.

  When extractions from different passes overlap in their character positions,
  the extraction from the earlier pass is kept (first-pass wins strategy).
  Only non-overlapping extractions from later passes are added to the result.

  Args:
    all_extractions: List of extraction iterables from different sequential
      extraction passes, ordered by pass number.

  Returns:
    List of merged extractions with overlaps resolved in favor of earlier
    passes.
  """
  if not all_extractions:
    return []

  if len(all_extractions) == 1:
    return list(all_extractions[0])

  merged_extractions = list(all_extractions[0])

  for pass_extractions in all_extractions[1:]:
    for extraction in pass_extractions:
      overlaps = False
      if extraction.char_interval is not None:
        for existing_extraction in merged_extractions:
          if existing_extraction.char_interval is not None:
            if _extractions_overlap(extraction, existing_extraction):
              overlaps = True
              break

      if not overlaps:
        merged_extractions.append(extraction)

  return merged_extractions


def _extractions_overlap(
    extraction1: data.Extraction, extraction2: data.Extraction
) -> bool:
  """Checks if two extractions overlap based on their character intervals.

  Args:
    extraction1: First extraction to compare.
    extraction2: Second extraction to compare.

  Returns:
    True if the extractions overlap, False otherwise.
  """
  if extraction1.char_interval is None or extraction2.char_interval is None:
    return False

  start1, end1 = (
      extraction1.char_interval.start_pos,
      extraction1.char_interval.end_pos,
  )
  start2, end2 = (
      extraction2.char_interval.start_pos,
      extraction2.char_interval.end_pos,
  )

  if start1 is None or end1 is None or start2 is None or end2 is None:
    return False

  # Two intervals overlap if one starts before the other ends
  return start1 < end2 and start2 < end1


def _document_chunk_iterator(
    documents: Iterable[data.Document],
    max_char_buffer: int,
    restrict_repeats: bool = True,
    tokenizer: tokenizer_lib.Tokenizer | None = None,
) -> Iterator[chunking.TextChunk]:
  """Iterates over documents to yield text chunks along with the document ID.

  Args:
    documents: A sequence of Document objects.
    max_char_buffer: The maximum character buffer size for the ChunkIterator.
    restrict_repeats: Whether to restrict the same document id from being
      visited more than once.
    tokenizer: Optional tokenizer instance.

  Yields:
    TextChunk containing document ID for a corresponding document.

  Raises:
    InvalidDocumentError: If restrict_repeats is True and the same document ID
      is visited more than once. Valid documents prior to the error will be
      returned.
  """
  visited_ids = set()
  for document in documents:
    if tokenizer:
      tokenized_text = tokenizer.tokenize(document.text or "")
    else:
      tokenized_text = document.tokenized_text
    document_id = document.document_id
    if restrict_repeats and document_id in visited_ids:
      raise exceptions.InvalidDocumentError(
          f"Document id {document_id} is already visited."
      )
    chunk_iter = chunking.ChunkIterator(
        text=tokenized_text,
        max_char_buffer=max_char_buffer,
        document=document,
        tokenizer_impl=tokenizer or tokenizer_lib.RegexTokenizer(),
    )
    visited_ids.add(document_id)

    yield from chunk_iter


class Annotator:
  """Annotates documents with extractions using a language model."""

  def __init__(
      self,
      language_model: base_model.BaseLanguageModel,
      prompt_template: prompting.PromptTemplateStructured,
      format_type: data.FormatType = data.FormatType.YAML,
      attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
      fence_output: bool = False,
      format_handler: fh.FormatHandler | None = None,
  ):
    """Initializes Annotator.

    Args:
      language_model: Model which performs language model inference.
      prompt_template: Structured prompt template where the answer is expected
        to be formatted text (YAML or JSON).
      format_type: The format type for the output (YAML or JSON).
      attribute_suffix: Suffix to append to attribute keys in the output.
      fence_output: Whether to expect/generate fenced output (```json or
        ```yaml). When True, the model is prompted to generate fenced output and
        the resolver expects it. When False, raw JSON/YAML is expected.
        Defaults to False. If format_handler is provided, it takes precedence.
      format_handler: Optional FormatHandler for managing format-specific logic.
    """
    self._language_model = language_model

    if format_handler is None:
      format_handler = fh.FormatHandler(
          format_type=format_type,
          use_wrapper=True,
          wrapper_key=data.EXTRACTIONS_KEY,
          use_fences=fence_output,
          attribute_suffix=attribute_suffix,
      )

    self._prompt_generator = prompting.QAPromptGenerator(
        template=prompt_template,
        format_handler=format_handler,
    )

    logging.debug(
        "Annotator initialized with format_handler: %s", format_handler
    )

  def annotate_documents(
      self,
      documents: Iterable[data.Document],
      resolver: resolver_lib.AbstractResolver | None = None,
      max_char_buffer: int = 200,
      batch_length: int = 1,
      debug: bool = True,
      extraction_passes: int = 1,
      context_window_chars: int | None = None,
      show_progress: bool = True,
      tokenizer: tokenizer_lib.Tokenizer | None = None,
      **kwargs,
  ) -> Iterator[data.AnnotatedDocument]:
    """Annotates a sequence of documents with NLP extractions.

      Breaks documents into chunks, processes them into prompts and performs
      batched inference, mapping annotated extractions back to the original
      document. Batch processing is determined by batch_length, and can operate
      across documents for optimized throughput.

    Args:
      documents: Documents to annotate. Each document is expected to have a
        unique document_id.
      resolver: Resolver to use for extracting information from text.
      max_char_buffer: Max number of characters that we can run inference on.
        The text will be broken into chunks up to this length.
      batch_length: Number of chunks to process in a single batch.
      debug: Whether to populate debug fields.
      extraction_passes: Number of sequential extraction attempts to improve
        recall by finding additional entities. Defaults to 1, which performs
        standard single extraction.
        Values > 1 reprocess tokens multiple times, potentially increasing
        costs with the potential for a more thorough extraction.
      context_window_chars: Number of characters from the previous chunk to
        include as context for the current chunk. Helps with coreference
        resolution across chunk boundaries. Defaults to None (disabled).
      show_progress: Whether to show progress bar. Defaults to True.
      tokenizer: Optional tokenizer to use. If None, uses default tokenizer.
      **kwargs: Additional arguments passed to LanguageModel.infer and Resolver.

    Yields:
      Resolved annotations from input documents.

    Raises:
      ValueError: If there are no scored outputs during inference.
    """
    if resolver is None:
      resolver = resolver_lib.Resolver(format_type=data.FormatType.YAML)

    if extraction_passes == 1:
      yield from self._annotate_documents_single_pass(
          documents,
          resolver,
          max_char_buffer,
          batch_length,
          debug,
          show_progress,
          context_window_chars=context_window_chars,
          tokenizer=tokenizer,
          **kwargs,
      )
    else:
      yield from self._annotate_documents_sequential_passes(
          documents,
          resolver,
          max_char_buffer,
          batch_length,
          debug,
          extraction_passes,
          show_progress,
          context_window_chars=context_window_chars,
          tokenizer=tokenizer,
          **kwargs,
      )

  def _annotate_documents_single_pass(
      self,
      documents: Iterable[data.Document],
      resolver: resolver_lib.AbstractResolver,
      max_char_buffer: int,
      batch_length: int,
      debug: bool,
      show_progress: bool = True,
      context_window_chars: int | None = None,
      tokenizer: tokenizer_lib.Tokenizer | None = None,
      **kwargs,
  ) -> Iterator[data.AnnotatedDocument]:
    """Single-pass annotation with stable ordering and streaming emission.

    Streams input without full materialization, maintains correct attribution
    across batches, and emits completed documents immediately to minimize
    peak memory usage. Handles generators from both infer() and align().

    When context_window_chars is set, includes text from the previous chunk as
    context for coreference resolution across chunk boundaries.
    """
    doc_order: list[str] = []
    doc_text_by_id: dict[str, str] = {}
    per_doc: DefaultDict[str, list[data.Extraction]] = collections.defaultdict(
        list
    )
    next_emit_idx = 0

    def _capture_docs(src: Iterable[data.Document]) -> Iterator[data.Document]:
      """Captures document order and text lazily as chunks are produced."""
      for document in src:
        document_id = document.document_id
        if document_id in doc_text_by_id:
          raise exceptions.InvalidDocumentError(
              f"Duplicate document_id: {document_id}"
          )
        doc_order.append(document_id)
        doc_text_by_id[document_id] = document.text or ""
        yield document

    def _emit_docs_iter(
        keep_last_doc: bool,
    ) -> Iterator[data.AnnotatedDocument]:
      """Yields documents that are guaranteed complete.

      Args:
        keep_last_doc: If True, retains the most recently started document
          for additional extractions. If False, emits all remaining documents.
      """
      nonlocal next_emit_idx
      limit = max(0, len(doc_order) - 1) if keep_last_doc else len(doc_order)
      while next_emit_idx < limit:
        document_id = doc_order[next_emit_idx]
        yield data.AnnotatedDocument(
            document_id=document_id,
            extractions=per_doc.get(document_id, []),
            text=doc_text_by_id.get(document_id, ""),
        )
        per_doc.pop(document_id, None)
        doc_text_by_id.pop(document_id, None)
        next_emit_idx += 1

    chunk_iter = _document_chunk_iterator(
        _capture_docs(documents), max_char_buffer, tokenizer=tokenizer
    )
    batches = chunking.make_batches_of_textchunk(chunk_iter, batch_length)

    model_info = progress.get_model_info(self._language_model)
    batch_iter = progress.create_extraction_progress_bar(
        batches, model_info=model_info, disable=not show_progress
    )

    chars_processed = 0

    prompt_builder = prompting.ContextAwarePromptBuilder(
        generator=self._prompt_generator,
        context_window_chars=context_window_chars,
    )

    try:
      for batch in batch_iter:
        if not batch:
          continue

        prompts = [
            prompt_builder.build_prompt(
                chunk.chunk_text, chunk.document_id, chunk.additional_context
            )
            for chunk in batch
        ]

        if show_progress:
          current_chars = sum(
              len(text_chunk.chunk_text) for text_chunk in batch
          )
          try:
            batch_iter.set_description(
                progress.format_extraction_progress(
                    model_info,
                    current_chars=current_chars,
                    processed_chars=chars_processed,
                )
            )
          except AttributeError:
            pass

        outputs = self._language_model.infer(batch_prompts=prompts, **kwargs)
        if not isinstance(outputs, list):
          outputs = list(outputs)

        for text_chunk, scored_outputs in zip(batch, outputs):
          if not isinstance(scored_outputs, list):
            scored_outputs = list(scored_outputs)
          if not scored_outputs:
            raise exceptions.InferenceOutputError(
                "No scored outputs from language model."
            )

          resolved_extractions = resolver.resolve(
              scored_outputs[0].output, debug=debug, **kwargs
          )

          token_offset = (
              text_chunk.token_interval.start_index
              if text_chunk.token_interval
              else 0
          )
          char_offset = (
              text_chunk.char_interval.start_pos
              if text_chunk.char_interval
              else 0
          )

          aligned_extractions = resolver.align(
              resolved_extractions,
              text_chunk.chunk_text,
              token_offset,
              char_offset,
              tokenizer_inst=tokenizer,
              **kwargs,
          )

          for extraction in aligned_extractions:
            per_doc[text_chunk.document_id].append(extraction)

          if show_progress and text_chunk.char_interval is not None:
            chars_processed += (
                text_chunk.char_interval.end_pos
                - text_chunk.char_interval.start_pos
            )

        yield from _emit_docs_iter(keep_last_doc=True)

    finally:
      batch_iter.close()

    yield from _emit_docs_iter(keep_last_doc=False)

  def _annotate_documents_sequential_passes(
      self,
      documents: Iterable[data.Document],
      resolver: resolver_lib.AbstractResolver,
      max_char_buffer: int,
      batch_length: int,
      debug: bool,
      extraction_passes: int,
      show_progress: bool = True,
      context_window_chars: int | None = None,
      tokenizer: tokenizer_lib.Tokenizer | None = None,
      **kwargs,
  ) -> Iterator[data.AnnotatedDocument]:
    """Sequential extraction passes logic for improved recall."""

    logging.info(
        "Starting sequential extraction passes for improved recall with %d"
        " passes.",
        extraction_passes,
    )

    document_list = list(documents)

    document_extractions_by_pass: dict[str, list[list[data.Extraction]]] = {}
    document_texts: dict[str, str] = {}
    # Preserve text up-front so we can emit documents even if later passes
    # produce no extractions.
    for _doc in document_list:
      document_texts[_doc.document_id] = _doc.text or ""

    for pass_num in range(extraction_passes):
      logging.info(
          "Starting extraction pass %d of %d", pass_num + 1, extraction_passes
      )

      for annotated_doc in self._annotate_documents_single_pass(
          document_list,
          resolver,
          max_char_buffer,
          batch_length,
          debug=(debug and pass_num == 0),
          show_progress=show_progress if pass_num == 0 else False,
          context_window_chars=context_window_chars,
          tokenizer=tokenizer,
          **kwargs,
      ):
        doc_id = annotated_doc.document_id

        if doc_id not in document_extractions_by_pass:
          document_extractions_by_pass[doc_id] = []
          # Keep first-seen text (already pre-filled above).

        document_extractions_by_pass[doc_id].append(
            annotated_doc.extractions or []
        )

    # Emit results strictly in original input order.
    for doc in document_list:
      doc_id = doc.document_id
      all_pass_extractions = document_extractions_by_pass.get(doc_id, [])
      merged_extractions = _merge_non_overlapping_extractions(
          all_pass_extractions
      )

      if debug:
        total_extractions = sum(
            len(extractions) for extractions in all_pass_extractions
        )
        logging.info(
            "Document %s: Merged %d extractions from %d passes into "
            "%d non-overlapping extractions.",
            doc_id,
            total_extractions,
            extraction_passes,
            len(merged_extractions),
        )

      yield data.AnnotatedDocument(
          document_id=doc_id,
          extractions=merged_extractions,
          text=document_texts.get(doc_id, doc.text or ""),
      )

    logging.info("Sequential extraction passes completed.")

  def annotate_text(
      self,
      text: str,
      resolver: resolver_lib.AbstractResolver | None = None,
      max_char_buffer: int = 200,
      batch_length: int = 1,
      additional_context: str | None = None,
      debug: bool = True,
      extraction_passes: int = 1,
      context_window_chars: int | None = None,
      show_progress: bool = True,
      tokenizer: tokenizer_lib.Tokenizer | None = None,
      **kwargs,
  ) -> data.AnnotatedDocument:
    """Annotates text with NLP extractions for text input.

    Args:
      text: Source text to annotate.
      resolver: Resolver to use for extracting information from text.
      max_char_buffer: Max number of characters that we can run inference on.
        The text will be broken into chunks up to this length.
      batch_length: Number of chunks to process in a single batch.
      additional_context: Additional context to supplement prompt instructions.
      debug: Whether to populate debug fields.
      extraction_passes: Number of sequential extraction passes to improve
        recall by finding additional entities. Defaults to 1, which performs
        standard single extraction. Values > 1 reprocess tokens multiple times,
        potentially increasing costs.
      context_window_chars: Number of characters from the previous chunk to
        include as context for coreference resolution. Defaults to None
        (disabled).
      show_progress: Whether to show progress bar. Defaults to True.
      tokenizer: Optional tokenizer instance.
      **kwargs: Additional arguments for inference and resolver_lib.

    Returns:
      Resolved annotations from text for document.
    """
    if resolver is None:
      resolver = resolver_lib.Resolver(
          format_type=data.FormatType.YAML,
      )

    start_time = time.time() if debug else None

    documents = [
        data.Document(
            text=text,
            document_id=None,
            additional_context=additional_context,
        )
    ]

    annotations = list(
        self.annotate_documents(
            documents=documents,
            resolver=resolver,
            max_char_buffer=max_char_buffer,
            batch_length=batch_length,
            debug=debug,
            extraction_passes=extraction_passes,
            context_window_chars=context_window_chars,
            show_progress=show_progress,
            tokenizer=tokenizer,
            **kwargs,
        )
    )
    assert (
        len(annotations) == 1
    ), f"Expected 1 annotation but got {len(annotations)} annotations."

    if debug and annotations[0].extractions:
      elapsed_time = time.time() - start_time if start_time else None
      num_extractions = len(annotations[0].extractions)
      unique_classes = len(
          set(e.extraction_class for e in annotations[0].extractions)
      )
      num_chunks = len(text) // max_char_buffer + (
          1 if len(text) % max_char_buffer else 0
      )

      progress.print_extraction_summary(
          num_extractions,
          unique_classes,
          elapsed_time=elapsed_time,
          chars_processed=len(text),
          num_chunks=num_chunks,
      )

    return data.AnnotatedDocument(
        document_id=annotations[0].document_id,
        extractions=annotations[0].extractions,
        text=annotations[0].text,
    )
