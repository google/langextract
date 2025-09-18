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

from collections.abc import Iterable, Iterator
import itertools
import time

from absl import logging

from langextract import chunking
from langextract import progress
from langextract import prompting
from langextract import resolver as resolver_lib
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import format_handler as fh


class DocumentRepeatError(exceptions.LangExtractError):
    """Exception raised when identical document ids are present."""


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
) -> Iterator[chunking.TextChunk]:
    """Iterates over documents to yield text chunks along with the document ID.

    Args:
      documents: A sequence of Document objects.
      max_char_buffer: The maximum character buffer size for the ChunkIterator.
      restrict_repeats: Whether to restrict the same document id from being
        visited more than once.

    Yields:
      TextChunk containing document ID for a corresponding document.

    Raises:
      DocumentRepeatError: If restrict_repeats is True and the same document ID
        is visited more than once. Valid documents prior to the error will be
        returned.
    """
    visited_ids = set()
    for document in documents:
        tokenized_text = document.tokenized_text
        document_id = document.document_id
        if restrict_repeats and document_id in visited_ids:
            raise DocumentRepeatError(f"Document id {document_id} is already visited.")
        chunk_iter = chunking.ChunkIterator(
            text=tokenized_text,
            max_char_buffer=max_char_buffer,
            document=document,
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

        logging.debug("Annotator initialized with format_handler: %s", format_handler)

    def annotate_documents(
        self,
        documents: Iterable[data.Document],
        resolver: resolver_lib.AbstractResolver | None = None,
        max_char_buffer: int = 200,
        batch_length: int = 1,
        debug: bool = True,
        extraction_passes: int = 1,
        show_progress: bool = True,
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
          show_progress: Whether to show progress bar. Defaults to True.
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
        **kwargs,
    ) -> Iterator[data.AnnotatedDocument]:
        """Single-pass annotation logic (original implementation)."""

        logging.info("Starting document annotation.")
        doc_iter, doc_iter_for_chunks = itertools.tee(documents, 2)
        curr_document = next(doc_iter, None)
        if curr_document is None:
            logging.warning("No documents to process.")
            return

        annotated_extractions: list[data.Extraction] = []
        chunk_iter = _document_chunk_iterator(doc_iter_for_chunks, max_char_buffer)

        batches = chunking.make_batches_of_textchunk(chunk_iter, batch_length)

        model_info = progress.get_model_info(self._language_model)

        progress_bar = progress.create_extraction_progress_bar(
            batches, model_info=model_info, disable=not show_progress
        )

        chars_processed = 0

        for index, batch in enumerate(progress_bar):
            logging.info("Processing batch %d with length %d", index, len(batch))

            batch_prompts: list[str] = []
            for text_chunk in batch:
                batch_prompts.append(
                    self._prompt_generator.render(
                        question=text_chunk.chunk_text,
                        additional_context=text_chunk.additional_context,
                    )
                )

            # Show what we're currently processing
            if debug and progress_bar:
                batch_size = sum(len(chunk.chunk_text) for chunk in batch)
                desc = progress.format_extraction_progress(
                    model_info,
                    current_chars=batch_size,
                    processed_chars=chars_processed,
                )
                progress_bar.set_description(desc)

            batch_scored_outputs = self._language_model.infer(
                batch_prompts=batch_prompts,
                **kwargs,
            )

            # Update total processed
            if debug:
                for chunk in batch:
                    if chunk.document_text:
                        char_interval = chunk.char_interval
                        chars_processed += (
                            char_interval.end_pos - char_interval.start_pos
                        )

                # Update progress bar with final processed count
                if progress_bar:
                    batch_size = sum(len(chunk.chunk_text) for chunk in batch)
                    desc = progress.format_extraction_progress(
                        model_info,
                        current_chars=batch_size,
                        processed_chars=chars_processed,
                    )
                    progress_bar.set_description(desc)

            for text_chunk, scored_outputs in zip(batch, batch_scored_outputs):
                logging.debug("Processing chunk: %s", text_chunk)
                if not scored_outputs:
                    logging.error(
                        "No scored outputs for chunk with ID %s.",
                        text_chunk.document_id,
                    )
                    raise exceptions.InferenceOutputError(
                        "No scored outputs from language model."
                    )
                while curr_document.document_id != text_chunk.document_id:
                    logging.info(
                        "Completing annotation for document ID %s.",
                        curr_document.document_id,
                    )
                    annotated_doc = data.AnnotatedDocument(
                        document_id=curr_document.document_id,
                        extractions=annotated_extractions,
                        text=curr_document.text,
                    )
                    yield annotated_doc
                    annotated_extractions.clear()

                    curr_document = next(doc_iter, None)
                    assert curr_document is not None, (
                        f"Document should be defined for {text_chunk} per"
                        " _document_chunk_iterator(...) specifications."
                    )

                top_inference_result = scored_outputs[0].output
                logging.debug("Top inference result: %s", top_inference_result)

                annotated_chunk_extractions = resolver.resolve(
                    top_inference_result, debug=debug, **kwargs
                )
                chunk_text = text_chunk.chunk_text
                token_offset = text_chunk.token_interval.start_index
                char_offset = text_chunk.char_interval.start_pos

                aligned_extractions = resolver.align(
                    annotated_chunk_extractions,
                    chunk_text,
                    token_offset,
                    char_offset,
                    **kwargs,
                )

                annotated_extractions.extend(aligned_extractions)

        progress_bar.close()

        if debug:
            progress.print_extraction_complete()

        if curr_document is not None:
            logging.info(
                "Finalizing annotation for document ID %s.", curr_document.document_id
            )
            annotated_doc = data.AnnotatedDocument(
                document_id=curr_document.document_id,
                extractions=annotated_extractions,
                text=curr_document.text,
            )

            yield annotated_doc

        logging.info("Document annotation completed.")

    def _annotate_documents_sequential_passes(
        self,
        documents: Iterable[data.Document],
        resolver: resolver_lib.AbstractResolver,
        max_char_buffer: int,
        batch_length: int,
        debug: bool,
        extraction_passes: int,
        show_progress: bool = True,
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
                **kwargs,
            ):
                doc_id = annotated_doc.document_id

                if doc_id not in document_extractions_by_pass:
                    document_extractions_by_pass[doc_id] = []
                    document_texts[doc_id] = annotated_doc.text or ""

                document_extractions_by_pass[doc_id].append(
                    annotated_doc.extractions or []
                )

        for doc_id, all_pass_extractions in document_extractions_by_pass.items():
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
                text=document_texts[doc_id],
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
        show_progress: bool = True,
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
          show_progress: Whether to show progress bar. Defaults to True.
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
                documents,
                resolver,
                max_char_buffer,
                batch_length,
                debug,
                extraction_passes,
                show_progress,
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
